# Algorithms-DataModels

CSC 501 final project on data-driven forecasting of newly released car MSRP. The project combines SQL analysis, Neo4j graph modeling, and machine learning to study how vehicle attributes influence price.

## Project Overview

The core question of the project is how structured vehicle attributes such as manufacturer, production year, engine volume, mileage, fuel type, drivetrain, and safety features relate to car price. The workflow in this repository has three parts:

- relational analysis with SQL;
- graph analysis with Neo4j;
- predictive modeling with regression and boosted/tree-based models.

## Repository Structure

- `doc/`: report assets and figures used in the project write-up.
- `doc/Data-Driven Forecasting- Newly Released Car MSRP.pdf`: main paper for the project.
- `doc/visualization Q3.png`: Neo4j visualization used for the manufacturer-to-model graph example.
- `doc/visualization Q4.png`: Neo4j visualization used for the high-price feature co-occurrence graph.
- `db-20251115T062130Z-1-001/db/car.db`: SQLite database copy.
- `sql-20251115T062131Z-1-001/sql/car.sql`: SQL queries used in the analysis section.
- `sql-20251115T062131Z-1-001/sql/BrandCountry Mapping.xlsx`: helper mapping used for brand-region grouping.
- `Neo4j-20251115T062133Z-1-001/Neo4j/car_neo4j_create_table.txt`: Neo4j CSV import script.
- `Dataset/car_neo4j_query.txt`: Neo4j queries for graph analysis and co-occurrence construction.
- `csv-20251115T062129Z-1-001/csv/car_price_prediction_RAW.csv`: raw dataset snapshot.
- `csv-20251115T062129Z-1-001/csv/car_price_prediction_CLEANED.csv`: cleaned dataset used by the training script.
- `model/train_model.py`: main ML training and evaluation pipeline.
- `model/artifacts/`: saved models, plots, metrics, and predictions from training runs.
- `code/RF-car_price_prediction.ipynb`: notebook experimentation.

## Dataset

The project uses the Kaggle Car Price Prediction Challenge dataset referenced in the paper. The cleaned CSV has 18 columns:

`ID, Price, Levy, Manufacturer, Model, Prod_year, Category, Leather_interior, Fuel_type, Engine_volume, Mileage, Cylinders, Gear_box_type, Drive_wheels, Doors, Wheel, Color, Airbags`

The cleaned CSV currently in the repository contains 19,234 data rows. The paper reports 19,237 rows, so the small difference likely comes from later cleanup or deduplication in the project files. The ML pipeline also removes duplicate `ID` values and incomplete rows before training.

## Questions Studied

The project and paper focus on three main analytical questions:

1. How does the share of gasoline vehicles change before and after threshold years such as 2000, 2010, and 2015?
2. How do average car prices differ across brand-origin groups such as Asian, American, and European?
3. Which vehicle features tend to co-occur in higher-priced cars when represented as a graph?

## SQL Component

The relational analysis uses SQLite and the queries in `sql-20251115T062131Z-1-001/sql/car.sql`.

The main SQL results described in the draft report are:

- gasoline share before 2000 vs after 2000;
- average price by `Brand_Country`;
- expanded grouped summaries by manufacturer and brand region.

The SQLite database used for these queries is stored at `db-20251115T062130Z-1-001/db/car.db`.

## Neo4j Component

The graph analysis uses two repository files:

- `Neo4j-20251115T062133Z-1-001/Neo4j/car_neo4j_create_table.txt`: imports the CSV into Neo4j and creates `Car`, `Manufacturer`, and `Model` nodes with `MAKES` and `INSTANCES` relationships.
- `Dataset/car_neo4j_query.txt`: contains the follow-up graph queries, including the high-price feature co-occurrence analysis.

The paper figures in `doc/visualization Q3.png` and `doc/visualization Q4.png` appear to come from this Neo4j workflow.

## How To Reproduce The Neo4j Visualization

To generate the Neo4j visualizations shown in the paper:

1. Start a local Neo4j database and open Neo4j Browser.
2. Place `car_price_prediction_CLEANED.csv` in Neo4j's `import` directory.
3. Run the import script from `Neo4j-20251115T062133Z-1-001/Neo4j/car_neo4j_create_table.txt`.
4. For the simpler manufacturer/model graph, run:

```cypher
MATCH p=(:Manufacturer {name: "MERCEDES-BENZ"})-[:MAKES]->(:Model)
RETURN p;
```

5. For the high-price co-occurrence graph, run the queries in `Dataset/car_neo4j_query.txt` that:
   - convert `Price` to a numeric property;
   - compute the top-20% price threshold;
   - build `CO_OCCUR_HIGHPRICE` relationships between feature nodes.
6. After the co-occurrence edges are created, use Neo4j Browser to return the connected feature pairs and adjust the visualization styling.

Example query to display the co-occurrence network after relationships are created:

```cypher
MATCH (a)-[r:CO_OCCUR_HIGHPRICE]->(b)
RETURN a, r, b
ORDER BY r.count DESC
LIMIT 100;
```

To make the graph look closer to the paper figure:

- size relationships or filter on larger `r.count` values so weak co-occurrences do not dominate the view;
- use the Browser style panel to color node labels by feature type;
- keep only a limited number of strongest edges on screen;
- export the final Browser view as an image.

The exact visual layout will vary slightly because Neo4j Browser uses a force-directed layout.

## Machine Learning Pipeline

`model/train_model.py` is the main implementation for price prediction. It:

- loads the cleaned CSV;
- cleans and normalizes numeric fields such as `Levy`, `Mileage`, and `Engine_volume`;
- creates `BrandCountry` from `Manufacturer`;
- winsorizes selected numeric columns;
- engineers features such as `Mileage_log`, `Engine_volume_log`, `Vehicle_age`, `Mileage_per_year`, `Engine_per_cylinder`, `Is_premium_brand`, and interaction terms;
- preprocesses numeric and categorical variables with imputation, scaling, and one-hot encoding;
- splits the data into train and test subsets;
- evaluates predictive performance with `MSE`, `RMSE`, `MAE`, `R²`, and `MAPE`;
- saves predictions, plots, metrics, and serialized model artifacts.

## Supported Models

The training script supports four `--model-type` options:

- `ridge`
- `neural`
- `lightgbm`
- `catboost`

The implementation in the repository is broader than the earlier draft report, which mainly discussed linear regression.

## Running The Model

From the repository root:

```bash
python3 model/train_model.py
```

Default data path:

```text
csv-20251115T062129Z-1-001/csv/car_price_prediction_CLEANED.csv
```

Example runs:

```bash
python3 model/train_model.py --model-type ridge
python3 model/train_model.py --model-type neural
python3 model/train_model.py --model-type lightgbm
python3 model/train_model.py --model-type catboost
```

Main dependencies inferred from the script are `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `joblib`, `jax`, `flax`, `optax`, `lightgbm`, and `catboost`.

## Generated Outputs

Training runs write to `model/artifacts/`, including:

- `metrics.json`
- `test_predictions.csv`
- `pred_vs_actual.png`
- `residual_histogram.png`
- `residuals_vs_pred.png`
- `cv_mse_bar.png`
- serialized `joblib` model files

## Latest Saved Results

The current `model/artifacts/metrics.json` corresponds to a CatBoost GPU run with:

- learning rate `0.05`
- depth `8`
- iterations `2000`
- `l2_leaf_reg` `5.0`
- `bagging_temperature` `1.0`
- `border_count` `128`

Saved test metrics:

- `RMSE`: `10578.02`
- `MAE`: `5747.86`
- `R²`: `0.5667`
- `MAPE`: `33.91%`

Baseline train-mean metrics:

- `RMSE`: `16073.38`
- `MAE`: `11757.39`
- `R²`: `-0.0005`

Non-outlier test performance after removing the top 2% most expensive cars:

- `RMSE`: `8406.44`
- `MAE`: `5077.20`
- `R²`: `0.6110`

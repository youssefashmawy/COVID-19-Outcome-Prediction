# README

## Project Overview

This project involves building, evaluating, and comparing different machine learning models for binary classification. The primary objective is to predict a target variable using features from the provided dataset (`data.csv`). The models explored include K-Nearest Neighbors (KNN), Logistic Regression, Gaussian Naive Bayes, Decision Trees, and Support Vector Machine (SVM).

The project is structured in two phases:
1. **Phase 1**: Initial model training and evaluation using KNN, Logistic Regression, and Gaussian Naive Bayes.
2. **Phase 2**: Further exploration and model comparison using Decision Trees and SVM, alongside Phase 1 models.

## Dependencies

The project requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install the required dependencies with:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Data Preparation

The dataset is loaded and split into features (`X`) and the target variable (`y`). The data is then split into training and test sets with a 75:25 ratio using `train_test_split` from `sklearn.model_selection`.

## Model Training and Evaluation

### Phase 1

Three models were initially trained and evaluated:
1. **K-Nearest Neighbors (KNN)**: Hyperparameters such as the number of neighbors (`n_neighbors`) and weights (`uniform` or `distance`) were tuned using `GridSearchCV`.
2. **Logistic Regression**: The model was trained with different regularization strengths (`C`) and penalties (`l2`). The data was standardized to address convergence issues.
3. **Gaussian Naive Bayes**: A basic model with no hyperparameter tuning.

For each model, performance metrics such as Precision, Recall, F1-score, and ROC-AUC were calculated and compared. The ROC curves were plotted to visualize the performance of each model.

### Phase 2

Two additional models were trained and compared:
1. **Decision Trees**: Hyperparameters such as `criterion`, `max_depth`, `min_samples_split`, and `max_leaf_nodes` were tuned using `GridSearchCV`.
2. **Support Vector Machine (SVM)**: Hyperparameters including the regularization parameter (`C`), kernel type, gamma, and degree were tuned.

The best models from Phase 1 were loaded for comparison with the newly trained models. The ROC curves for all models were plotted together to identify the most suitable model based on ROC-AUC.

## Results

### Model Comparison

Based on the evaluation metrics:
- **KNN** performed best overall in terms of ROC-AUC and F1-score in Phase 1.
- **Decision Tree** and **SVM** models were introduced in Phase 2, with competitive performance.

The project concludes that **KNN** remains the most suitable model for this specific dataset and classification task.

### Model Summary

A summary of all the models with their best hyperparameters and performance metrics (Precision, Recall, F1-score, and ROC-AUC) is presented in a table format.

## Saving and Loading Models

The trained models are saved using `joblib` for later use:
```python
joblib.dump(model, 'model_name.pkl')
```
You can load a model for prediction with:
```python
model = joblib.load('model_name.pkl')
```

## Visualizations

- ROC curves for all models.
- Confusion matrices for each model.
- Data summaries and performance tables.

## How to Run

1. Ensure the required dependencies are installed.
2. Load the dataset (`data.csv`) in the project directory.
3. Run the Jupyter Notebook or Python script to train the models, evaluate them, and generate the visualizations.

## Future Improvements

- Implement more advanced models like Random Forests, XGBoost, or Neural Networks.
- Perform feature engineering to improve model performance.
- Experiment with techniques like SMOTE for handling imbalanced datasets.
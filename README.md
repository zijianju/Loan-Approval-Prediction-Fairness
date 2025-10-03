# Responsible AI Audit: Loan Approval Prediction

This project builds and audits machine learning models for **loan default prediction** using a Kaggle credit risk dataset.  
Beyond achieving high predictive performance, the focus is on **responsible AI** practices: interpretability, fairness analysis, and robustness checks.


## 🔍 Project Overview
- **Goal:** Predict loan default risk and audit subgroup fairness.
- **Dataset:** ~91,000 records × 26 features (demographics, loan details, credit history).
- **Target:** `loan_status` (1 = default, 0 = non-default).
- **Models:** LightGBM and CatBoost with Stratified K-Fold cross-validation.


## Workflow
1. **Exploratory Data Analysis (EDA)**
   - Distributions (age, income, interest rate).
   - Stacked bars for default vs. loan intent & home ownership.
   - Correlation heatmap of engineered features.

2. **Feature Engineering**
   - Loan-to-income ratio, debt-to-credit ratio, interest-to-loan ratio.
   - Interaction terms (e.g., interest rate × employment).
   - Per-employee/per-year normalization.

3. **Modeling**
   - LightGBM & CatBoost with tuned hyperparameters.
   - Evaluation metric: **ROC-AUC**.
   - Blended probabilities for final predictions.

4. **Responsible AI Audit**
   - **Stability check:** Gaussian noise injection (σ≈1%) → <1% accuracy drop.
   - **Fairness slices:** Subgroup metrics by *home ownership* and *loan grade*.
   - **Local interpretability:** LIME explanations for individual predictions.


## Tech Stack
- **Languages:** Python (pandas, numpy, matplotlib, seaborn)
- **ML Libraries:** scikit-learn, LightGBM, CatBoost, XGBoost
- **Explainability & Fairness:** LIME, Fairlearn
- **Environment:** Jupyter Notebook


## Data Source
https://www.kaggle.com/competitions/playground-series-s4e10/data

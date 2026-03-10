---
name: courier-analyst
description: Intelligent file analysis with subagent delegation and artifact delivery. Analyzes uploaded files (CSV, logs, code, documents), produces structured reports with insights, and sends results back to the user via send_file.
version: 1.0.0
author: eren-karakus0
license: MIT
metadata:
  hermes:
    tags: [Analysis, Files, Delegation, Artifacts, Productivity]
    platforms: [cli, telegram, discord]
---

# Courier Analyst

Analyze uploaded files, produce structured reports with visualizations, and deliver artifacts back to the user via send_file.

## When to Use

Activate this skill when:
- A user sends a file (CSV, log, JSON, code archive, document) and asks for analysis
- The user wants insights, statistics, patterns, or quality checks on data
- The user needs a structured report generated from raw data
- The user wants charts or visualizations from data

## Required Toolsets

This skill requires: `terminal`, `file`, `file_transfer`

## CRITICAL Rules

1. NEVER use `python -c "..."` for multi-line code — ALWAYS use `write_file` to create a .py script first, then run it with `terminal`
2. Use `python` command (NOT `python3`)
3. Always add `matplotlib.use('Agg')` BEFORE `import matplotlib.pyplot as plt`
4. Use `send_file` for EACH output file separately (one call per file)
5. Save output files in the current working directory (not /workspace/ or /tmp/)

## Workflow

### Step 1: Read the file
Use `read_file` to examine the uploaded file and understand its structure.

### Step 2: Write analysis script
Use `write_file` to create a Python script (e.g. `analyze_data.py`). The script should:
- Read the data file
- Compute statistics
- Generate charts (save as .png)
- Write a markdown report (save as .md)

### Step 3: Run the script
Use `terminal` with command: `python analyze_data.py`
If it fails, read the error, fix the script with `write_file`, and re-run.

### Step 4: Deliver results
Use `send_file` once for EACH output file:
- `send_file` for the analysis report (.md)
- `send_file` for each chart (.png)

## CSV Analysis Template

When analyzing CSV data, use `write_file` to create this script (adapt column names to match the actual data):

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Read data
df = pd.read_csv('INPUT_FILE.csv')
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")
print(df.describe())

# --- Bar Chart: sales by region ---
fig, ax = plt.subplots(figsize=(10, 6))
group_data = df.groupby('REGION_COLUMN')['VALUE_COLUMN'].sum().sort_values(ascending=False)
group_data.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_title('Total Sales by Region', fontsize=14, fontweight='bold')
ax.set_xlabel('Region')
ax.set_ylabel('Total Sales ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('region_sales_chart.png', dpi=150)
plt.close()
print("Saved region_sales_chart.png")

# --- Pie Chart: sales by category ---
fig, ax = plt.subplots(figsize=(8, 8))
cat_data = df.groupby('CATEGORY_COLUMN')['VALUE_COLUMN'].sum()
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']
cat_data.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors[:len(cat_data)], startangle=140)
ax.set_title('Sales Distribution by Category', fontsize=14, fontweight='bold')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('category_chart.png', dpi=150)
plt.close()
print("Saved category_chart.png")

# --- Markdown Report ---
total_sales = df['VALUE_COLUMN'].sum()
avg_sales = df['VALUE_COLUMN'].mean()
report = f"""# Data Analysis Report: INPUT_FILE.csv

## Overview
- **Total Rows**: {len(df)}
- **Columns**: {len(df.columns)}
- **Total Sales**: ${total_sales:,.2f}
- **Average Sale**: ${avg_sales:,.2f}

## Sales by Region
"""
for region, val in group_data.items():
    pct = val / total_sales * 100
    report += f"- **{region}**: ${val:,.2f} ({pct:.1f}%)\n"

report += "\n## Sales by Category\n"
for cat, val in cat_data.items():
    pct = val / total_sales * 100
    report += f"- **{cat}**: ${val:,.2f} ({pct:.1f}%)\n"

report += f"""
## Data Quality
- Missing values: {df.isnull().sum().sum()} total
- Duplicate rows: {df.duplicated().sum()}

## Key Findings
1. Top region: {group_data.index[0]} (${group_data.iloc[0]:,.2f})
2. Top category: {cat_data.idxmax()} (${cat_data.max():,.2f})
3. {len(df)} transactions analyzed
"""

with open('analysis_report.md', 'w') as f:
    f.write(report)
print("Saved analysis_report.md")
print("DONE - all files generated successfully")
```

Adapt the column names (REGION_COLUMN, VALUE_COLUMN, CATEGORY_COLUMN, INPUT_FILE) to match the actual CSV columns found in Step 1.

## Log Analysis Template

For log files, create a script that:
- Counts log levels (ERROR, WARN, INFO)
- Identifies error patterns and frequencies
- Creates a timeline chart of errors over time
- Writes a markdown report

## Example Interaction

**User sends:** `sales_data.csv` with message "analyze this"

**Agent does:**
1. `read_file` → sees columns: date, region, category, amount, status
2. `write_file` → creates `analyze_data.py` adapted from the CSV template above
3. `terminal` → runs `python analyze_data.py` → output says "DONE"
4. `send_file(path="region_sales_chart.png", caption="Sales by Region")` → delivered
5. `send_file(path="category_chart.png", caption="Sales by Category")` → delivered
6. `send_file(path="analysis_report.md", caption="Analysis Report")` → delivered

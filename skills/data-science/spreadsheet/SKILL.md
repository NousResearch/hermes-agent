---
name: spreadsheet
description: "Use this skill any time a spreadsheet file is involved — .xlsx, .csv, .tsv, or .xls. This includes: reading, parsing, analyzing, creating, editing, or transforming tabular data; applying formulas; creating pivot tables; generating charts; diffing spreadsheets; or converting between formats. Trigger whenever the user mentions spreadsheet, Excel, CSV, workbook, or references a tabular data file."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [spreadsheet, excel, csv, data, pandas, openpyxl, pivot, chart]
    related_skills: [jupyter-live-kernel]
---

# Spreadsheet Skill

## Quick Reference

| Task | Command |
|------|---------|
| Read spreadsheet to JSON | `python scripts/read_sheet.py data.xlsx` |
| Read specific sheet | `python scripts/read_sheet.py data.xlsx --sheet Sales` |
| Read as markdown table | `python scripts/read_sheet.py data.xlsx --format markdown` |
| Write JSON to spreadsheet | `python scripts/write_sheet.py output.xlsx --input data.json` |
| Write JSON from stdin | `echo '[{"a":1}]' \| python scripts/write_sheet.py output.xlsx` |
| Set a formula | `python scripts/apply_formula.py data.xlsx --cell B10 --formula '=SUM(B1:B9)'` |
| Create pivot table | `python scripts/pivot.py data.xlsx --rows Category --values Revenue --aggfunc sum` |
| Embed a chart | `python scripts/chart.py data.xlsx --type bar --x Month --y Revenue` |
| Diff two files | `python scripts/diff.py old.xlsx new.xlsx` |

**All scripts** are in this skill's `scripts/` directory. Run them via the terminal tool with `python scripts/<script>.py`.

---

## Prerequisites

```bash
# Install required packages (use uv if available, otherwise pip)
uv pip install openpyxl pandas xlsxwriter 2>/dev/null || pip install openpyxl pandas xlsxwriter
```

These packages are only needed for xlsx operations. CSV reading/writing works with no extra dependencies.

---

## Reading Data

```bash
# Read entire workbook (first sheet by default) → JSON array of row objects
python scripts/read_sheet.py report.xlsx

# Read a specific sheet
python scripts/read_sheet.py report.xlsx --sheet "Q3 Data"

# Read a cell range
python scripts/read_sheet.py report.xlsx --range A1:D20

# Output as markdown table (great for displaying to user)
python scripts/read_sheet.py report.xlsx --format markdown

# Read CSV
python scripts/read_sheet.py data.csv

# Read TSV
python scripts/read_sheet.py data.tsv
```

Output is JSON by default — an array of objects where keys are column headers.

---

## Writing Data

```bash
# Write from a JSON file
python scripts/write_sheet.py output.xlsx --input data.json

# Pipe JSON from stdin
echo '[{"Name": "Alice", "Score": 95}, {"Name": "Bob", "Score": 87}]' | python scripts/write_sheet.py output.xlsx

# Write to a named sheet
python scripts/write_sheet.py output.xlsx --input data.json --sheet Results

# Write CSV instead of xlsx
python scripts/write_sheet.py output.csv --input data.json
```

The JSON input should be an array of objects with consistent keys.

---

## Formulas

```bash
# Set a single formula
python scripts/apply_formula.py report.xlsx --cell B10 --formula '=SUM(B1:B9)'

# Set formula on a specific sheet
python scripts/apply_formula.py report.xlsx --cell C5 --formula '=AVERAGE(C1:C4)' --sheet Analysis

# Batch-apply formulas from JSON
python scripts/apply_formula.py report.xlsx --batch '[{"cell": "B10", "formula": "=SUM(B1:B9)"}, {"cell": "C10", "formula": "=AVERAGE(C1:C9)"}]'
```

Formulas are written as-is (Excel syntax). They are NOT evaluated by the script — Excel or another spreadsheet app will compute values on open.

---

## Pivot Tables

```bash
# Basic pivot: sum of Revenue grouped by Category
python scripts/pivot.py sales.xlsx --rows Category --values Revenue --aggfunc sum

# Multi-dimensional: Category rows, Month columns
python scripts/pivot.py sales.xlsx --rows Category --cols Month --values Revenue --aggfunc sum

# Save result to a new file
python scripts/pivot.py sales.xlsx --rows Category --values Revenue --aggfunc mean --output pivot_result.xlsx

# Available aggregations: sum, mean, count, min, max, median
```

Output is JSON by default. Use `--output` to write to a new spreadsheet.

---

## Charts

```bash
# Bar chart
python scripts/chart.py data.xlsx --type bar --x Month --y Revenue --title "Monthly Revenue"

# Line chart saved to a different file
python scripts/chart.py data.xlsx --type line --x Date --y Price --output chart.xlsx

# Pie chart
python scripts/chart.py data.xlsx --type pie --x Category --y Amount

# Scatter plot
python scripts/chart.py data.xlsx --type scatter --x Weight --y Height

# Supported types: bar, line, pie, scatter, area
```

Charts are embedded into the xlsx file on a new sheet named "Chart". Use `--output` to write to a separate file instead of modifying the input.

---

## Diffing

```bash
# Compare two files (cell-level diff)
python scripts/diff.py old.xlsx new.xlsx

# Compare specific sheets
python scripts/diff.py old.xlsx new.xlsx --sheet "Q3 Data"

# Output as markdown
python scripts/diff.py old.xlsx new.xlsx --format markdown
```

Shows added, removed, and changed cells with old/new values.

---

## Common Patterns

### Read → Transform → Write

```bash
# 1. Read data
python scripts/read_sheet.py input.xlsx > /tmp/data.json

# 2. Transform with Python/jq/etc.
# (use terminal or write a quick script)

# 3. Write back
python scripts/write_sheet.py output.xlsx --input /tmp/data.json
```

### Analyze and Report

```bash
# Read the data
python scripts/read_sheet.py sales.xlsx --format markdown

# Create a pivot summary
python scripts/pivot.py sales.xlsx --rows Region --values Revenue --aggfunc sum --output summary.xlsx

# Add a chart
python scripts/chart.py summary.xlsx --type bar --x Region --y Revenue --title "Revenue by Region"
```

### Best Practices

- **Always read before modifying** — check the sheet structure and column names first
- **CSV for simple data** — if the user doesn't need formatting, formulas, or charts, prefer CSV
- **Large files** — for files with >100k rows, warn the user that operations may be slow and consider reading a subset with `--range`
- **Preserve originals** — write to a new file rather than overwriting the input unless explicitly asked
- **Column names** — the scripts use the first row as headers by default; if the file has no headers, the agent should note this

---

## Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `openpyxl` | Read/write xlsx with formatting | `pip install openpyxl` |
| `pandas` | Pivot tables, aggregation, CSV | `pip install pandas` |
| `xlsxwriter` | Chart embedding | `pip install xlsxwriter` |

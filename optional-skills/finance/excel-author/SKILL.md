---
name: excel-author
description: Build auditable Excel workbooks headless with openpyxl — blue/black/green cell conventions, formulas over hardcodes, named ranges, balance checks, sensitivity tables. Use for financial models, audit outputs, reconciliations.
version: 1.0.0
author: Anthropic (adapted by Nous Research)
license: Apache-2.0
metadata:
  hermes:
    tags: [excel, openpyxl, finance, spreadsheet, modeling]
    related_skills: [pptx-author, dcf-model, comps-analysis, lbo-model, 3-statement-model]
---

# excel-author

Produce an .xlsx file on disk using `openpyxl`. Follow the banker-grade conventions below so the model is auditable, flexible, and reviewable by someone other than the person who built it.

Adapted from Anthropic's `xlsx-author` and `audit-xls` skills in the [anthropics/financial-services](https://github.com/anthropics/financial-services) repo. The MCP / Office-JS / Cowork-specific branches of the originals are dropped — this skill assumes headless Python.

## Output contract

- Write to `./out/<name>.xlsx`. Create `./out/` if it does not exist.
- Return the relative path in your final message so downstream tools can pick it up.
- One logical model per file. Do not append to an existing workbook unless explicitly asked.

## Setup

```bash
pip install "openpyxl>=3.0"
```

## Core conventions (non-negotiable)

### Blue / black / green cell color
- **Blue** (`Font(color="0000FF")`) — hardcoded input a human entered. Revenue drivers, WACC inputs, terminal growth, market data.
- **Black** (default) — formula. Every derived cell is a live Excel formula.
- **Green** (`Font(color="006100")`) — link to another sheet or external file.

A reviewer can then scan the sheet and immediately see what's an assumption vs. what's computed.

### Formulas over hardcodes
Every calculation cell MUST be a formula string, never a number computed in Python and pasted as a value.

```python
# WRONG — silent bug waiting to happen
ws["D20"] = revenue_prior_year * (1 + growth)

# CORRECT — flexes when the user changes the assumption
ws["D20"] = "=D19*(1+$B$8)"
```

The only hardcoded numbers permitted:
1. Raw historical inputs (actual revenues, reported EBITDA, etc.)
2. Assumption drivers the user is meant to flex (growth rates, WACC inputs, terminal g)
3. Current market data (share price, debt balance) — with a cell comment documenting source + date

If you catch yourself computing a value in Python and writing the result, stop.

### Named ranges for cross-sheet references
Use named ranges for any figure referenced from another sheet, a deck, or a memo.

```python
from openpyxl.workbook.defined_name import DefinedName
wb.defined_names["WACC"] = DefinedName("WACC", attr_text="Inputs!$C$8")
# then elsewhere:
calc["D30"] = "=D29/WACC"
```

### Balance checks tab
Include a `Checks` tab that ties everything and surfaces TRUE/FALSE:
- Balance sheet balances (assets = liabilities + equity)
- Cash flow ties to period-over-period cash change on the BS
- Sum-of-parts ties to consolidated totals
- No rogue hardcodes inside calc ranges

Example:
```python
checks = wb.create_sheet("Checks")
checks["A2"] = "BS balances"
checks["B2"] = "=IS!D20-IS!D21-IS!D22"
checks["C2"] = "=ABS(B2)<0.01"  # TRUE/FALSE
```

### Cell comments on every hardcoded input
Add the comment AS you create the cell, not later.

```python
from openpyxl.comments import Comment
ws["C2"] = 1_250_000_000
ws["C2"].font = Font(color="0000FF")
ws["C2"].comment = Comment("Source: 10-K FY2024, p.47, revenue line", "analyst")
```

Format: `Source: [System/Document], [Date], [Reference], [URL if applicable]`.

Never defer sourcing. Never write `TODO: add source`.

## Skeleton: typical financial model

```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.comments import Comment
from openpyxl.utils import get_column_letter
from pathlib import Path

BLUE = Font(color="0000FF")
BLACK = Font(color="000000")
GREEN = Font(color="006100")
BOLD = Font(bold=True)
HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(color="FFFFFF", bold=True)

wb = Workbook()

# --- Inputs tab ---
inp = wb.active
inp.title = "Inputs"
inp["A1"] = "MARKET DATA & KEY INPUTS"
inp["A1"].font = HEADER_FONT
inp["A1"].fill = HEADER_FILL
inp.merge_cells("A1:C1")

inp["B3"] = "Revenue FY2024"
inp["C3"] = 1_250_000_000
inp["C3"].font = BLUE
inp["C3"].comment = Comment("Source: 10-K FY2024 p.47", "model")

inp["B4"] = "Growth Rate"
inp["C4"] = 0.12
inp["C4"].font = BLUE

# --- Calc tab ---
calc = wb.create_sheet("DCF")
calc["B2"] = "Projected Revenue"
calc["C2"] = "=Inputs!C3*(1+Inputs!C4)"   # formula, black

# --- Checks tab ---
chk = wb.create_sheet("Checks")
chk["A2"] = "BS balances"
chk["B2"] = "=ABS(BS!D20-BS!D21-BS!D22)<0.01"

Path("./out").mkdir(exist_ok=True)
wb.save("./out/model.xlsx")
```

## Section headers with merged cells

openpyxl quirk: when you merge, set the value on the top-left cell and style the full range separately.

```python
ws["A7"] = "CASH FLOW PROJECTION"
ws["A7"].font = HEADER_FONT
ws.merge_cells("A7:H7")
for col in range(1, 9):  # A..H
    ws.cell(row=7, column=col).fill = HEADER_FILL
```

## Sensitivity tables

Build with loops, not hardcoded formulas per cell. Rules:

- **Odd number of rows/cols** (5×5 or 7×7) — guarantees a true center cell.
- **Center cell = base case.** The middle row/col header must equal the model's actual WACC and terminal g so the center output equals the base-case implied share price. That's the sanity check.
- **Highlight the center cell** with medium-blue fill (`"BDD7EE"`) and bold.
- Populate every cell with a full recalculation formula — never an approximation.

```python
# 5x5 WACC (rows) x terminal growth (cols) sensitivity
wacc_axis = [0.08, 0.085, 0.09, 0.095, 0.10]        # center row = base 9.0%
term_axis = [0.02, 0.025, 0.03, 0.035, 0.04]        # center col = base 3.0%

start_row = 40
ws.cell(row=start_row, column=1).value = "Implied Share Price ($)"
ws.cell(row=start_row, column=1).font = BOLD

for j, g in enumerate(term_axis):
    ws.cell(row=start_row+1, column=2+j).value = g
    ws.cell(row=start_row+1, column=2+j).font = BLUE

for i, w in enumerate(wacc_axis):
    r = start_row + 2 + i
    ws.cell(row=r, column=1).value = w
    ws.cell(row=r, column=1).font = BLUE
    for j, g in enumerate(term_axis):
        c = 2 + j
        # Full DCF recalc formula (simplified for illustration).
        # In a real model this references the full projection block.
        ws.cell(row=r, column=c).value = (
            f"=SUMPRODUCT(FCF_range,1/(1+{w})^year_offset) + "
            f"FCF_terminal*(1+{g})/({w}-{g})/(1+{w})^terminal_year"
        )

# Highlight center cell (base case)
center = ws.cell(row=start_row+2+len(wacc_axis)//2,
                 column=2+len(term_axis)//2)
center.fill = PatternFill("solid", fgColor="BDD7EE")
center.font = BOLD
```

## Recalculating before delivery

openpyxl writes formula strings but does not compute them. Excel recalculates on open, but downstream consumers (auto-check scripts, CI) need computed values.

Run LibreOffice or a dedicated recalc step before delivery:

```bash
# LibreOffice headless recalc
libreoffice --headless --calc --convert-to xlsx ./out/model.xlsx --outdir ./out/
```

Or use a Python recalc helper (see `scripts/recalc.py` in this skill).

## Model layout planning

Before writing any formula:
1. Define ALL section row positions
2. Write ALL headers and labels
3. Write ALL section dividers and blank rows
4. THEN write formulas using the locked row positions

This prevents the cascading-formula-breakage pattern where inserting a header row after formulas are written shifts every downstream reference.

## Verify step-by-step with the user

For large models (DCFs, 3-statement, LBO), stop and show the user intermediate artifacts before continuing. Catching a wrong margin assumption before you've built downstream sensitivity tables saves an hour.

Checkpoint pattern:
- After Inputs block → show raw inputs, confirm before projecting
- After Revenue projections → confirm top line + growth
- After FCF build → confirm the full schedule
- After WACC → confirm inputs
- After valuation → confirm the equity bridge
- THEN build sensitivity tables

## When NOT to use this skill

- Users in a live Excel session with an Office MCP available — drive their live workbook instead.
- Pure tabular data export with no formulas — `csv` or `pandas.to_excel` is simpler.
- Dashboards / charts with heavy interactivity — use a real BI tool.

## Attribution

Conventions (blue/black/green, formulas-over-hardcodes, named ranges, sensitivity rules) adapted from Anthropic's Claude for Financial Services plugin suite, Apache-2.0 licensed. Original: https://github.com/anthropics/financial-services/tree/main/plugins/vertical-plugins/financial-analysis/skills/xlsx-author

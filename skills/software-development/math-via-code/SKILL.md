---
name: math-via-code
description: "Offloads arithmetic to code execution for accuracy."
version: 2.0.0
author: Tom Mulkins (@tommulkins)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [math, accuracy, arithmetic, computation, financial-modeling, code-execution]
    related_skills: [ocr-and-documents, google-workspace]
---

# Math Via Code Skill

LLM arithmetic is unreliable — especially past ~150k context tokens. This skill enforces a hard rule: all non-trivial math goes through `execute_code` or a Python script run via `terminal`. The only exception is a single operation on two numbers (e.g. `5 * 3`). It does not replace domain judgment; it only forbids in-head arithmetic.

## When to Use

- Any computation involving 3+ numbers, even "simple" addition
- Financial analysis: SDE, P&L, margins, royalties, annualizations
- Growth rates, ratios, percentages, weighted averages
- Unit conversions, projections, multi-facility rollups
- Reading data from files or APIs and computing derived values
- Whenever the user provides numbers and expects a computed result

**Don't use for:** Trivial single-operation arithmetic on two numbers (`5 * 3`, `100 - 20`).

## Prerequisites

- `execute_code` available for pure-Python math (no third-party packages)
- Project Python environment via `terminal` when packages like `openpyxl` or `pandas` are required
- Real source data (conversation numbers, workbook, API) — never fabricated test figures

## How to Run

1. Capture inputs as variables (read programmatically from files/APIs; never re-type from memory).
2. Write the calculation in Python.
3. Run via `execute_code` (stdlib only) or `terminal` (venv Python + packages).
4. Assert invariants and, when a source total exists, cross-check against it.
5. Report inputs, outputs, and assertion pass/fail.

```python
# Example: SDE from raw inputs
revenue = 327158
direct_costs = 192193
gp = revenue - direct_costs

overhead_items = [960, 1351.92, 9000, 1440, 1137.24]
overhead_royalties = round(revenue * 0.07, 2)
total_overhead = sum(overhead_items) + overhead_royalties

owner_salary = 4500 * 12
owner_payroll_tax = 370 * 12
owner_auto = 364 * 12
owner_comp = owner_salary + owner_payroll_tax + owner_auto

net_profit = gp - total_overhead - owner_comp
sde = net_profit + owner_comp

assert abs(gp + direct_costs - revenue) < 0.01, "GP + DOC should equal revenue"
assert abs(sde - (net_profit + owner_comp)) < 0.01, "SDE = net + add-back"
assert abs(overhead_royalties - round(revenue * 0.07, 2)) < 0.01, "Royalties = 7%"
```

When a script needs external packages or complex quoting, write it to the system temp dir and run the file:

```python
import tempfile
from pathlib import Path

script = Path(tempfile.gettempdir()) / "math_via_code_calc.py"
script.write_text(code, encoding="utf-8")
# then: terminal → python path/to/venv/bin/python script
```

## Quick Reference

| Calculation | Formula |
|-------------|---------|
| Gross margin | `(revenue - cogs) / revenue * 100` |
| SDE | `net_profit + owner_salary + payroll_tax + auto + other_addbacks` |
| Annualized | `ytd_total / months_elapsed * 12` |
| Growth rate | `(current - prior) / prior * 100` |
| Royalties | `revenue * royalty_rate` |
| Multi-facility rollup | `sum(facility[i] for i in facilities)` |
| Weighted average | `sum(val * weight) / sum(weights)` |

| Tool | When |
|------|------|
| `execute_code` | Pure-Python math, no external packages |
| `terminal` | Need pandas, openpyxl, or other installed packages |
| `read_file` / programmatic loaders | Pull numbers from files — never hand-transcribe |

## Procedure

### 1. Determine inputs

Capture numbers as variables. Source does not change the rule — compute in code, verify against source.

- **Conversation:** Assign directly to variables.
- **File (Excel, CSV, PDF):** Read programmatically; do not copy-paste partial numbers.
- **Web / API:** Extract into variables, then compute.

### 2. Write and run the calculation

Use `execute_code` for in-memory stdlib math. Use `terminal` with the project's venv when packages are required.

### 3. Verify (mandatory)

**Self-check assertions** (same names as the calculation variables):

```python
assert abs(gp + direct_costs - revenue) < 0.01, "GP + DOC should equal revenue"
assert abs(sde - (net_profit + owner_comp)) < 0.01, "SDE = net + add-back"
assert abs(overhead_royalties - round(revenue * 0.07, 2)) < 0.01, "Royalties = 7%"
```

**Cross-check against source** (document, workbook, or API total):

```python
assert abs(computed_sde - source_sde) < 0.01, \
    "SDE mismatch: computed={}, source={}".format(computed_sde, source_sde)
```

### 4. Report

- Show inputs and computed outputs
- Show assertion results (pass/fail)
- Flag if computed values differ from source

### Excel / .xlsx specifics

1. Use the project Python environment (not the sandbox) when openpyxl/pandas are needed.
2. Always `data_only=True` so you get computed values, not formulas:
   ```python
   wb = openpyxl.load_workbook(path, data_only=True)
   ```
3. Strip whitespace from labels — Excel cells often have leading spaces:
   ```python
   label = cell.value.strip()  # NOT cell.value
   ```
4. Write multi-line scripts under `tempfile.gettempdir()` and run the file; avoid complex f-strings in subprocess.
5. Verify facility sums match totals — sum individuals and compare to the "All Customers" row.

## Pitfalls

1. **Sandbox Python missing packages.** Use the project venv via `terminal` for pandas/openpyxl.
2. **Excel labels have leading/trailing spaces.** Always `.strip()` before comparison.
3. **`data_only=False` returns formulas not values.** Pass `data_only=True` to `load_workbook`.
4. **f-strings with escaped quotes break in subprocess.** Write a temp file via `tempfile.gettempdir()`, then run it.
5. **Rounding too early.** Keep full precision during computation; round only for display.
6. **Hand-transcribing numbers.** Read the source programmatically.
7. **Made-up "test" data.** Always use real source numbers.
8. **"Quick" mental math on 3+ numbers.** Don't — write code.

## Verification

- [ ] All arithmetic executed via `execute_code` or a Python script — nothing computed in-head
- [ ] Inputs captured as variables, not re-typed from memory
- [ ] Self-check assertions pass (or cross-check against source values)
- [ ] Rounded for display only; full precision used during computation
- [ ] If data came from a document/workbook/API, computed result compared back to source
- [ ] Temp scripts use `tempfile.gettempdir()`, not a hardcoded POSIX-only temp path

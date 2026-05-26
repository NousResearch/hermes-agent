---
name: excel-xlsx
triggers:
  - Excel
  - XLSX
  - spreadsheet
  - openpyxl
  - pandas excel
  - workbook
  - formula preservation
  - date serial
description: >
  Best-practice instructions for AI agents handling Excel/XLSX spreadsheet
  operations. Covers workflow selection (pandas vs openpyxl), date serial
  handling (1900 bug, 1904 epoch), formula preservation, data type protection
  (long IDs, leading zeros), workbook structure preservation, write-only/read-only
  modes for large files, 10 common traps, and the pandas↔openpyxl bridge pattern.
  Source: clawhub.ai/ivangdavila/excel-xlsx
user-invocable: false
---

# Excel / XLSX Skill

Best-practice instructions for AI agents handling spreadsheet operations.
Source: clawhub.ai/ivangdavila/excel-xlsx

## Workflow Selection

- Use `pandas` for analysis, reshaping, and CSV-like tasks.
- Use `openpyxl` when formulas, styles, sheets, comments, merged cells, or workbook preservation matter.
- When in doubt, ask: "Does the output need to look like a spreadsheet, or just contain the data?"

## Date Handling

- Excel stores dates as serial numbers (1 = 1900-01-01).
- The 1900 date system has a known bug: it treats 1900 as a leap year (serial 60 = Feb 29, 1900, which doesn't exist).
- macOS Excel may use the 1904 date system (serial 0 = 1904-01-01). Check `workbook.epoch` in openpyxl.
- Always convert dates explicitly. Never assume a numeric column is or isn't a date without checking.

```python
from openpyxl.utils.datetime import from_excel
from datetime import datetime

# Reading: convert serial to datetime
dt = from_excel(serial_number)

# Writing: use datetime objects, not strings
ws["A1"] = datetime(2026, 3, 31)
ws["A1"].number_format = "YYYY-MM-DD"
```

## Formula Preservation

- Write formulas as strings: `ws["B2"] = "=SUM(A1:A10)"`
- Never hardcode values that should be formulas.
- After writing, validate formulas load correctly:

```python
from openpyxl import load_workbook
wb = load_workbook("output.xlsx")
ws = wb.active
for row in ws.iter_rows():
    for cell in row:
        if cell.data_type == "f":
            print(f"{cell.coordinate}: {cell.value}")
```

- If the recipient needs current calculated values (not just formula strings), use `data_only=True` to read cached values, or recalculate via Excel/LibreOffice.

## Data Type Protection

- Excel truncates numbers beyond 15 significant digits. Store long IDs (barcodes, account numbers) as text:

```python
from openpyxl.utils import get_column_letter
for row in ws.iter_rows(min_row=2):
    row[0].number_format = "@"  # Text format
    row[0].value = str(long_id)
```

- Leading zeros are stripped by default. Prefix with apostrophe or use text format.
- pandas `read_excel(dtype=str)` prevents type inference corruption.

## Workbook Structure Preservation

When modifying an existing workbook:

```python
# Always load with full preservation
wb = load_workbook("template.xlsx", rich_text=True, data_only=False)

# Work on specific sheets, don't create new workbook
ws = wb["Sheet1"]

# Save to SAME file or new file — never wb = Workbook() if preserving
wb.save("output.xlsx")
```

- Merged cells: check `ws.merged_cells.ranges` before writing.
- Hidden sheets/rows/columns: preserve unless explicitly asked to change.
- Conditional formatting: survives load/save in openpyxl but test after modifications.

## File Scaling

For large workbooks (>50k rows):

```python
# Writing: use write-only mode
from openpyxl import Workbook
wb = Workbook(write_only=True)
ws = wb.create_sheet()
for row_data in data_generator():
    ws.append(row_data)
wb.save("large_output.xlsx")

# Reading: use read-only mode
wb = load_workbook("large_input.xlsx", read_only=True)
ws = wb.active
for row in ws.iter_rows(values_only=True):
    process(row)
wb.close()  # Important in read-only mode
```

## Common Traps

1. **Column indexing**: openpyxl is 1-based, pandas is 0-based.
2. **Sheet names**: max 31 characters, no `[]:*?/\` characters.
3. **Empty rows**: `ws.max_row` may overcount if cells were deleted but formatting remains. Use `ws.calculate_dimension()`.
4. **Number formats**: `"0.00"` is a number format, `"@"` is text. Don't confuse with Python format strings.
5. **CSV round-trip**: Saving as CSV loses formulas, styles, and multi-sheet structure. Always warn before CSV export from XLSX.
6. **Macros**: openpyxl cannot read/write `.xlsm` macros. Use `keep_vba=True` to preserve them passively.
7. **Named ranges**: Check `wb.defined_names` before assuming cell references.
8. **Encoding**: Excel uses UTF-8 but some older files may be Latin-1. pandas handles this; openpyxl may not.
9. **Memory**: Loading large files without `read_only=True` can exhaust memory.
10. **Timezone**: Excel dates are timezone-naive. Convert explicitly if your data has timezones.

## pandas ↔ openpyxl Bridge

```python
import pandas as pd
from openpyxl import load_workbook

# Read with pandas, write styled output with openpyxl
df = pd.read_excel("input.xlsx", sheet_name="Data")
# ... transform with pandas ...

# Write to new sheet in existing workbook
wb = load_workbook("input.xlsx")
ws = wb.create_sheet("Analysis")
for r_idx, row in enumerate(df.itertuples(index=False), start=2):
    for c_idx, value in enumerate(row, start=1):
        ws.cell(row=r_idx, column=c_idx, value=value)
# Add headers
for c_idx, col_name in enumerate(df.columns, start=1):
    ws.cell(row=1, column=c_idx, value=col_name)
wb.save("output.xlsx")
```

## Dependencies

- `openpyxl` — workbook preservation, formulas, styles
- `pandas` — data analysis, reshaping, CSV-like operations
- Both should be installed: `pip install openpyxl pandas`

---
name: xlsx
description: "Create, read, edit Excel .xlsx spreadsheets and CSVs."
version: 1.0.0
author: Anthropic (adapted by Nous Research)
license: Proprietary. LICENSE.txt has complete terms
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Excel, XLSX, Spreadsheets, Office, Productivity]
    category: productivity
    related_skills: [docx, pdf, powerpoint]
---

# XLSX Skill

Create, read, and edit Excel workbooks — formulas, formatting, charts, data cleaning, and format conversion. Every formula-bearing output must be recalculated and error-free before delivery.

## When to Use

Use this skill any time a spreadsheet file is the primary input or output: opening, reading, editing, or fixing an existing .xlsx, .xlsm, .xltx, .csv, or .tsv file; creating a new spreadsheet from scratch or from other data; converting between tabular formats; cleaning messy tabular data into a proper spreadsheet. Trigger whenever the user references a spreadsheet file by name or path — even casually. Do NOT trigger when the deliverable is a Word document (`docx` skill), HTML report, standalone script, or Google Sheets API integration. For finance-grade modeling conventions (DCF, LBO, three-statement), the optional `excel-author` skill adds stricter standards on top of this one.

## Prerequisites

```bash
pip install openpyxl pandas "markitdown[xlsx]"
which soffice || sudo apt install -y libreoffice   # formula recalculation (scripts/recalc.py)
```

macOS: `brew install libreoffice`.

## Quick Reference

| Task | Approach |
|---|---|
| **Create** or **edit** with formulas/formatting | `openpyxl` — see gotchas below |
| **Bulk data** in or out | `pandas` (`read_excel`, `to_excel`) |
| **Quick look** at a sheet | `markitdown file.xlsx` — `## SheetName` per sheet; reads `.xlsm` too. No cell coordinates, so don't plan edits from it. (`read_file` also auto-extracts .xlsx) |
| **Read** a model (formulas *and* values) | two `load_workbook` passes — see gotchas |

> Script paths below are relative to this skill's directory.

## Requirements for every output

- **Professional font** (Arial, Times New Roman) throughout, unless the user says otherwise.
- **Zero formula errors.** Never ship while `recalc.py` reports `errors_found`. If you think an error predates you, prove it: load the *original* with `data_only=True` and look at that cell. An error you introduced looks exactly like one you inherited.
- **Use formulas, never hardcoded results.** Write `sheet['B10'] = '=SUM(B2:B9)'`, not the Python-computed total. The sheet must recalculate when its inputs change.
- **Follow the user's spec literally.** Exact tab names, exact column headers, and the formula they spelled out. A redesign that computes something else fails, however elegant.
- **Document every assumption and hardcoded number** where the reader will see it — a cell comment, or an adjacent cell at a table's end. Cite a real source when one exists; when the number came from the user, say so plainly.
- **A workbook *you create* for someone to fill in** needs a short legend naming which cells to edit, and one example row of realistic values showing the expected format. Never add such a row to a file you were asked to edit.
- **Editing an existing file: match its conventions exactly.** They override every guideline here. Find its designated input cells first — a distinct font color, fill, or shading marks them — write only there, and leave every existing formula untouched.

## Recalculate (mandatory whenever the file contains formulas)

openpyxl writes formulas as strings with **no cached values**. Until you recalculate, every formula cell reads back as `None` to anything reading cached values — `pandas`, `load_workbook(data_only=True)`, and most previewers.

```bash
python scripts/recalc.py output.xlsx [timeout_seconds]   # default 30
```

LibreOffice computes every formula, the file is **rewritten in place**, and you get JSON: `status` (`success` | `errors_found`), `total_formulas`, `total_errors`, and an `error_summary` naming up to 100 cells per error type (`locations_truncated` says how many it withheld — trust `total_errors`, not the length of the list). Fix what it names and run it again. **JSON with an `error` key instead of a `status` means nothing was recalculated**, and only that case exits non-zero — `errors_found` exits 0, so never treat a clean exit as a clean workbook.

**A green recalc proves your formulas *evaluate*, not that they are *right*.** An off-by-one range or a reference to the wrong row yields a clean, error-free file with wrong numbers. Write 2–3 formulas first and check they pull the values you expect, before building out a grid.

**A workbook that links to another file loses those links** if you re-save it with openpyxl and then recalculate. Such a formula reads `='[1]Returns Analysis'!$B$2` — the `[1]` is an index into the workbook's external-reference list, naming a *separate file on disk*, not a sheet. That file is rarely present, so the cell's cached value is the only thing holding its data. openpyxl strips that value on save; LibreOffice then has to resolve the reference for real, fails, writes `#NAME?`, and deletes every link. `recalc.py` refuses to run in that state — copy those cells' values out of the original before you save over them (`--force` overrides, and accepts the loss).

## Choosing formulas that survive verification

LibreOffice implements fewer functions than Excel, and one it cannot evaluate becomes a literal `#NAME?` baked into the file you deliver.

- **Prefer Excel-2007-era functions** — `SUMIFS`, `INDEX`, `MATCH`, `IFERROR`, `SUMPRODUCT` — which need no prefix.
- **Six post-2007 functions work, but only with an `_xlfn.` prefix**, because openpyxl writes your formula into the XML verbatim and Excel stores post-2007 names prefixed (its UI hides the prefix): `_xlfn.TEXTJOIN`, `_xlfn.CONCAT`, `_xlfn.IFS`, `_xlfn.SWITCH`, `_xlfn.MAXIFS`, `_xlfn.MINIFS`. Written bare, each yields `#NAME?`.
- **Never use `XLOOKUP`, `XMATCH`, `SORT`, `FILTER`, `UNIQUE`, or `SEQUENCE`.** LibreOffice cannot reliably evaluate them; newer builds that do are spilling array functions, and an openpyxl-written file has no spill metadata, so only the top-left cell of the range gets a value — and `recalc.py` reports `total_errors: 0` on the truncated result. Use `INDEX`/`MATCH` for lookups, and sort, filter, and de-duplicate in Python before writing the cells.
- A formula LibreOffice could not parse is written back **lowercased** — a quick tell beside a `#NAME?`.

## openpyxl gotchas

- **Reading a model takes two loads.** `data_only=True` yields cached values with the formulas gone; the default yields formula strings with no values. One pass cannot give you both.
- **`data_only=True` is destructive if you save.** That workbook has no formulas left, so saving replaces every one with a literal — permanently.
- **`data_only=True` on a file openpyxl just wrote returns `None` everywhere** — run `recalc.py` first. (A formula whose result is `""` also reads back as `None`.)
- **Merged cells: write the top-left anchor only.** Every other cell in the range is a `MergedCell` whose `.value` is read-only.
- **`.xlsm` loses its macros unless you pass `keep_vba=True`** to `load_workbook`.
- **A sheet name containing a space must be quoted** in a cross-sheet reference: `='Assumptions Inputs'!$B$5`. Unquoted, it evaluates to `#VALUE!`.

## Financial models

Unless the user says otherwise, or the existing file already does something else.

**Color:** blue text (`0,0,255`) for hardcoded inputs and scenario levers · black for formulas · green (`0,128,0`) for links to another sheet · red (`255,0,0`) for links to another file · yellow fill (`255,255,0`) for key assumptions and cells the user should fill in.

**Numbers:** currency `$#,##0`, with the unit named in the header (`Revenue ($mm)`) · zeros render as `-`, including in percentages (`$#,##0;($#,##0);-`) · negatives in parentheses · percentages `0.0%`, **stored as fractions** (`0.15` renders `15.0%`; storing `15` renders `1500.0%`) · valuation multiples `0.0x` · years as text (`"2024"`, never `2,024`).

**Structure:** every assumption in its own labeled cell, referenced by the formulas that use it (`=B5*(1+$B$6)`, never `=B5*1.05`) · formulas consistent across every projection period, since a lone edited cell mid-row is the commonest silent error · guard denominators that can be zero.

For full investment-banking conventions (balance checks, sensitivity tables, named ranges), install the optional skill: `hermes skills install official/finance/excel-author`.

## Verification

1. `python scripts/recalc.py output.xlsx` → `status: success`, `total_errors: 0`.
2. Spot-check 2–3 computed cells against expected values (`load_workbook(data_only=True)` *after* recalc).
3. `markitdown output.xlsx` — scan for missing sheets, misplaced headers, leftover placeholders.

## Related skills

`docx` (Word documents), `pdf` (PDF work), `powerpoint` (decks), optional `excel-author` (finance-grade modeling standards).

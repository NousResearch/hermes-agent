---
name: csv-inspect
description: "Profile CSV/TSV files with zero dependencies: schema inference, nulls, distincts, numeric summaries."
version: 1.0.0
author: Nous Research
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [csv, tsv, data-profiling, schema, inspection, productivity]
    category: productivity
    related_skills: [xlsx, jupyter-notebook]
---

# CSV Inspect

Profile delimited files (CSV/TSV) **without opening a notebook or installing
pandas**: one stdlib-only script answers "what is in this file" before you
commit to a workflow.

Use it when the user asks about a CSV's shape or contents ("what columns does
this have", "how many rows", "why is this column full of blanks"), before
writing conversion code (`xlsx` skill), or as the first step of any data task
so you quote real column names instead of guessed ones.

## What it reports

- File facts: encoding (BOM/cp1252 detection), delimiter (sniffed), row and
  column counts, blank rows.
- Per column:
  - inferred type — `int`, `float`, `bool`, `date`, `text` (majority vote)
  - null/empty count and percentage
  - distinct count (and distinct-percentage for low-cardinality columns)
  - numeric columns: min / max / mean (parsed, so `1,234` and `$5.00` count)
  - `date` columns: min / max observed value
  - up to 5 sample values

## Usage

```bash
# Full profile as JSON
csv_inspect.py data.csv

# First 3 data rows as arrays (after sniffing header/delimiter)
csv_inspect.py data.csv --head 3

# Machine-readable subset: only the schema table
csv_inspect.py data.csv --json | jq '.columns[].name'

# Semicolon-separated European export with latin-1 text
csv_inspect.py export.csv --delimiter ";" --encoding cp1252

# Trust an existing header row exactly; treat "-" as missing
csv_inspect.py weird.tsv --no-header --na-values "-"
```

All output goes to stdout as JSON (`--pretty` for indented). Exit code is
nonzero only when the file cannot be read/parsed, so it chains in pipelines.

## Options

| Flag | Meaning |
|------|---------|
| `--delimiter SEP` | Force field separator (default: auto-sniff `,` `\t` `;` `\|`) |
| `--encoding ENC` | Force encoding (default: UTF-8, BOM-aware, cp1252 fallback) |
| `--no-header` | File has no header row; columns are named `col_1..N` |
| `--head N` | Include first N parsed data rows in output |
| `--na-values LIST` | Space-separated extra strings treated as null (default: empty string; add e.g. `- NA N/A null`) |
| `--max-sample K` | Distinct-value sample cap per column (default 10000 rows scanned for distincts) |
| `--pretty` | Indent JSON output |

## Notes and limits

- Streams row-by-row: handles multi-GB files without loading them into memory;
  only the distinct-value sample caps are held in RAM.
- Type inference uses majority vote over sampled rows; a column of mostly
  numbers with a few stray words reports `text` — check `sample_values`.
- Quoted fields with embedded newlines are handled correctly by the stdlib
  csv parser but inflate row-parse time on pathological files.
- For Excel workbooks use the `xlsx` skill; for PDF text use the `pdf` skill.

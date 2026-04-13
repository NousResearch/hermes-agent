---
name: polars-lazyframes
description: >
  Fast columnar dataframes with Polars — lazy scans, schema inspection, head previews,
  and CSV/Parquet/NDJSON conversion. Use for large single-file analytics, ETL-shaped
  transforms, and Arrow-friendly pipelines without a SQL engine or Jupyter session.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [polars, dataframe, parquet, csv, lazy, etl, data-science]
    category: data-science
---

# Polars LazyFrames & DataFrames

**Polars** is a widely used Rust-backed dataframe library for Python. This skill covers
**lazy scans**, **schema discovery**, **bounded previews**, and **format conversion**
via a small CLI that prints **JSON** (easy for agents to parse).

## When to Use This Skill

| Situation | Prefer |
|-----------|--------|
| Stateful exploration, notebook-style iteration | `jupyter-live-kernel` |
| Embedded SQL over Parquet/CSV with DuckDB | dedicated DuckDB skill (if installed) |
| Lazy columnar scans, head/sample, Parquet/CSV/NDJSON I/O | **this skill** |

## Prerequisites

- **Python 3.11+**
- Install Polars only for the command (keeps the core agent slim):

```bash
uv run --with polars python skills/data-science/polars-lazyframes/scripts/polars_run.py --help
```

Adjust the path if your skill root differs (bundled copy under `~/.hermes/skills/`).

## CLI Reference (`polars_run.py`)

All subcommands write **one JSON object** to stdout. On failure: `"ok": false` plus `error` and `message`.

### 1. Inspect lazy schema

Supported inputs: `.parquet`, `.csv`, `.tsv`, `.json`, `.jsonl`, `.ndjson`.

```bash
uv run --with polars python skills/data-science/polars-lazyframes/scripts/polars_run.py inspect ./data/events.parquet
```

Example success shape: `{"ok": true, "path": "events.parquet", "schema": {"id": "Int64", ...}}`.

### 2. Head preview (collect first N rows)

```bash
uv run --with polars python skills/data-science/polars-lazyframes/scripts/polars_run.py head ./data/events.csv -n 5
```

Returns `rows` as a list of row dicts. Keep `n` small on huge files.

### 3. Convert formats (eager)

```bash
uv run --with polars python skills/data-science/polars-lazyframes/scripts/polars_run.py convert ./in.csv ./out.parquet
```

Outputs: `.parquet`, `.csv`, `.ndjson`, or `.jsonl`.

## Polars Patterns (in-code)

For heavier logic inside `uv run --with polars` scripts or notebooks:

- Prefer `scan_parquet` / `scan_csv` + `filter`, `select`, `with_columns`, then `collect()`.
- Use expressions (`pl.col("x").sum()`) over Python loops.
- For strings, see `str` and `replace` expression namespaces in Polars docs.

## Safety

- Pass **resolved** paths; the CLI resolves arguments to absolute paths internally.
- `head` materializes only **N** rows; avoid huge `n` on remote-mounted data.
- `convert` loads the **full** file into memory — use only when size is acceptable.

---
name: duckdb-analytics
description: >
  Run fast analytical SQL with DuckDB over Parquet, CSV, JSON, and in-process
  Arrow/Pandas data. Covers embedded OLAP, serverless-friendly local analytics,
  joins across files, window functions, and exporting results. Use when the user
  mentions DuckDB, Parquet analytics, SQL on CSV, OLAP, columnar queries,
  read_parquet, or lightweight ETL without a separate database server.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags:
      - duckdb
      - sql
      - parquet
      - csv
      - olap
      - analytics
      - data-science
      - embedded-database
    category: data-science
    related_skills: [jupyter-live-kernel]
---

# DuckDB Analytics

[DuckDB](https://duckdb.org/) is an in-process analytical SQL engine (columnar, vectorized). It is ideal for querying Parquet/CSV on disk, federating multiple files, and fast aggregations without running PostgreSQL or Spark.

## When to Use

| Situation | Use this skill |
|-----------|----------------|
| SQL over `.parquet` / `.csv` / `.json` | Yes |
| Multi-file joins and aggregations | Yes |
| Reproducible one-off analytics in the terminal | Yes |
| Need a long-lived server or multi-user RDBMS | No — use a proper server database |
| Stateful notebook exploration | Prefer `jupyter-live-kernel` |

## Prerequisites

- **uv** available (`uv --version`), or a Python 3.11+ environment where you can `pip install duckdb`.
- Optional **DuckDB CLI** (`duckdb --version`) for interactive sessions — the bundled script works with only the Python package.

## Helper Script (recommended)

From the repository root (or any cwd), run via `uv` so `duckdb` does not need a prior global install:

```bash
uv run --with duckdb python skills/data-science/duckdb-analytics/scripts/duckdb_run.py \
  -q "SELECT version()"
```

Register Parquet/CSV files as **views** (the script resolves each path to an absolute file, rejects odd characters, and builds a safe `read_*` call — do not pass untrusted path strings from remote input):

```bash
uv run --with duckdb python skills/data-science/duckdb-analytics/scripts/duckdb_run.py \
  --parquet orders=data/orders.parquet \
  --parquet customers=data/customers.parquet \
  -q 'SELECT COUNT(*) AS n FROM orders' \
  --format json
```

- `--format json` (default): array of objects, UTF-8, pretty-printed.
- `--format tsv`: header row + tab-separated values.

**Exit codes:** `0` success, `1` SQL/runtime error, `2` missing `duckdb` package.

## Core SQL Patterns

**Read files inline (no view):**

```sql
SELECT * FROM read_parquet('path/to/file.parquet') LIMIT 10;
SELECT * FROM read_csv_auto('path/to/file.csv');
SELECT * FROM read_json_auto('path/to/file.json');
```

**Attach for multiple queries (CLI or interactive):**

```sql
ATTACH 'analytics.duckdb' AS db;
CREATE TABLE db.summary AS SELECT region, SUM(amount) AS rev FROM read_parquet('sales/*.parquet') GROUP BY 1;
```

**Windows paths:** Prefer forward slashes in SQL strings, or double backslashes; avoid pasting raw unescaped backslashes.

## Performance and Scale

- DuckDB parallelizes scans; keep predicates selective when possible.
- For many Parquet files, use globs: `read_parquet('dir/**/*.parquet')` when layout allows.
- Spill-to-disk settings exist for large sorts/joins; if queries fail with OOM, suggest smaller date slices or `SET threads=N`.

## Pitfalls

- **Single-statement helper:** `duckdb_run.py` runs one `-q` string per invocation; use the CLI or Python API for scripts with multiple statements.
- **Mutating files:** Writing Parquet/CSV back is supported in DuckDB, but confirm paths and backups before `COPY ... TO`.
- **Type inference:** `read_csv_auto` guesses types; cast explicitly (`TRY_CAST`, `::DOUBLE`) when results look wrong.

## Verification

1. `uv run --with duckdb python .../duckdb_run.py -q "SELECT 1 AS ok"` → JSON `[{"ok": 1}]`.
2. Run a `COUNT(*)` on a registered `--parquet` view and confirm row count matches expectations.

## Further Reading

- Official docs: [duckdb.org/docs](https://duckdb.org/docs/)
- SQL introduction: [duckdb.org/docs/sql/introduction](https://duckdb.org/docs/sql/introduction)

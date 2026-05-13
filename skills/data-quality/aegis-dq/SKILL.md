---
name: aegis-dq
description: "Agentic data quality validation across warehouses (DuckDB, BigQuery, Athena, Databricks, Postgres) with LLM diagnosis, root cause analysis, and audit trail."
version: 0.7.0
author: Aegis Contributors
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [data-quality, sql, warehouse, analytics, audit, duckdb, bigquery, athena, databricks, postgres]
    category: data-quality
---

# Aegis DQ

Runs structured data quality rules against your warehouses and uses LLMs to diagnose failures, trace root causes, and propose SQL remediations. Every decision is audit-logged.

## When to Use This Skill

Use Aegis DQ when you need to:
- Validate data in a warehouse against business rules (nulls, ranges, referential integrity, custom SQL)
- Understand *why* a data quality check failed, not just *that* it failed
- Search past diagnoses across runs
- Run validation on a schedule and get a conversational summary

## Prerequisites

```bash
pip install aegis-dq
```

Set warehouse environment variables for any system you want to validate:

| Warehouse | Required env vars |
|---|---|
| DuckDB | `DUCKDB_PATH` (default: `:memory:`) |
| BigQuery | `BQ_PROJECT`, `BQ_DATASET` |
| Athena | `ATHENA_S3_STAGING_DIR`, `AWS_REGION` |
| Databricks | `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_TOKEN` |
| Postgres / Redshift | `POSTGRES_DSN` |

Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) for LLM diagnosis. Omit to run offline.

## Available Tools

The Aegis MCP server exposes these tools:

- **`run_validation`** — Run a rules YAML file against a warehouse. Returns pass/fail per rule, LLM diagnosis, root cause, and remediation SQL.
- **`list_runs`** — List recent run IDs from the audit trail.
- **`get_run_report`** — Get the full report for a past run by ID.
- **`get_trajectory`** — Get the node-by-node LLM decision log for a run.
- **`search_decisions`** — Full-text search across all past LLM decisions.

## Example Prompts

- "Run my rules.yaml against BigQuery and tell me what failed."
- "Show me the last 10 validation runs."
- "What was the root cause in yesterday's run?"
- "Search the audit trail for anything about null order IDs."
- "Run rules.yaml against Athena offline — no LLM, just pass/fail."

## Running a Validation

The `run_validation` tool takes:
- `rules_path` — path to your rules YAML file
- `warehouse` — one of: `duckdb`, `bigquery`, `athena`, `databricks`, `postgres`
- `connection_params` — JSON object with connection kwargs (falls back to env vars if omitted)
- `no_llm` — set `true` to skip LLM diagnosis for fast offline checks

Example: validate against BigQuery using env vars

```
run_validation(rules_path="/home/user/rules/orders.yaml", warehouse="bigquery")
```

Example: validate against Postgres with explicit DSN

```
run_validation(
  rules_path="/home/user/rules/orders.yaml",
  warehouse="postgres",
  connection_params="{\"dsn\": \"postgresql://user:pass@host:5432/db\"}"
)
```

## Edge Cases

- If `connection_params` is omitted and required env vars are missing, the tool returns a clear error listing which variables to set.
- `no_llm: true` skips all LLM calls — useful for fast checks or when no API key is configured.
- Rules that reference tables not present in the warehouse return a clear SQL error.

## Links

- Docs: https://aegis-dq.dev/integrations/hermes
- GitHub: https://github.com/aegis-dq/aegis-dq
- PyPI: https://pypi.org/project/aegis-dq/

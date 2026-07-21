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
- Load a pipeline manifest and run it with a single prompt
- Search past diagnoses across runs
- Compare two validation runs to spot regressions
- Run validation on a schedule and get a conversational summary

## Prerequisites

**1. Install Aegis and scaffold a project**

```bash
pip install aegis-dq
aegis init my-project --name my-pipeline
cd my-project
```

This creates `aegis.yaml` (project-wide LLM + warehouse config), a starter `pipelines/my-pipeline/pipeline.yaml`, and `rules.yaml`.

**2. Configure Hermes**

Add Aegis to `~/.hermes/config.yaml`:

```yaml
model:
  default: claude-haiku-4-5-20251001
  provider: anthropic

mcp_servers:
  aegis:
    command: aegis
    args: [mcp]
    env:
      ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
      DUCKDB_PATH: /data/prod.duckdb
```

**3. Set warehouse env vars**

| Warehouse | Required env vars |
|---|---|
| DuckDB | `DUCKDB_PATH` (default: `:memory:`) |
| BigQuery | `BQ_PROJECT`, `BQ_DATASET` |
| Athena | `ATHENA_S3_STAGING_DIR`, `AWS_REGION` |
| Databricks | `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_TOKEN` |
| Postgres / Redshift | `POSTGRES_DSN` |

Set `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`, or `AWS_DEFAULT_REGION` for Bedrock) for LLM diagnosis. Omit to run offline with `no_llm: true`.

## Available Tools

The Aegis MCP server exposes 9 tools:

- **`load_pipeline`** — Load a `pipeline.yaml` manifest and return its config, connection params, and goal as context. Use before `run_validation` so Hermes understands the pipeline without re-explanation.
- **`run_validation`** — Run a rules YAML file against a warehouse. Returns pass/fail per rule, LLM diagnosis, root cause, and remediation SQL.
- **`list_runs`** — List recent run IDs from the audit trail, newest first.
- **`get_run_report`** — Get the full report for a past run by ID.
- **`get_trajectory`** — Get the node-by-node LLM decision log for a run — every prompt, response, cost, and latency.
- **`search_decisions`** — Full-text search across all past LLM decisions.
- **`compare_reports`** — Diff two runs side by side — regressions, fixes, pass-rate delta.
- **`summarize_reports`** — Compact summary of one or more runs — pass rate, top failures, cost.
- **`check_consistency`** — Detect flapping rules and rule-set drift between two runs.

## Pipeline Manifests

The best way to use Aegis with Hermes is a **pipeline manifest** — a single YAML file that captures your rules, warehouse, and goal. Define it once; invoke it with two words.

```yaml
# pipelines/orders-dq/pipeline.yaml
name: orders-dq
description: Daily order data quality checks
rules: ./rules.yaml
goal: |
  For every failure explain the business impact, the likely root
  cause, and a concrete remediation step. Group by severity.
```

Hermes calls `load_pipeline` → reads the manifest → calls `run_validation` with the right params. You never re-explain the context.

## Example Prompts

**Pipeline manifest (recommended)**
- "Load the pipeline at my-project/pipelines/orders-dq/pipeline.yaml and run it."
- "Load the fraud pipeline and run it, then send me a severity summary."

**Direct validation**
- "Run /home/user/rules/orders.yaml against BigQuery and tell me what failed."
- "Run rules.yaml against Athena offline — no LLM, just pass/fail."

**Audit trail**
- "Show me the last 10 validation runs."
- "What was the root cause of the CTR filing failures in yesterday's run?"
- "Search the audit trail for anything about null order IDs."
- "Compare today's run with yesterday's — what newly failed?"
- "Have we ever seen OFAC sanction hits before?"

**Scheduling**
- "Run the fraud-aml pipeline every morning at 8am and alert me in Slack if anything is CRITICAL."

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
- `load_pipeline` resolves all paths relative to the manifest file, so manifests are portable across machines.

## Links

- Docs: https://aegis-dq.dev/integrations/hermes
- GitHub: https://github.com/aegis-dq/aegis-dq
- PyPI: https://pypi.org/project/aegis-dq/

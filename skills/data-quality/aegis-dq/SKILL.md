---
name: aegis-dq
description: "Validate warehouse data with LLM diagnosis and audit log."
version: 0.7.0
author: "Shiva Koreddi (@koreddi) with Hermes Agent"
license: Apache-2.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [data-quality, sql, warehouse, analytics, audit, duckdb, bigquery, athena, databricks, postgres]
    category: data-quality
---

# Aegis DQ Skill

Runs structured data quality rules against warehouses and uses LLMs to diagnose failures, trace root causes, and propose SQL remediations. Every LLM decision is audit-logged. Does not replace your warehouse's native constraints — it validates business rules on top of them.

## When to Use

Use Aegis DQ when you need to:
- Validate data against business rules (nulls, ranges, referential integrity, custom SQL)
- Understand *why* a check failed, not just *that* it failed
- Search past LLM diagnoses across runs
- Compare two validation runs to spot regressions

## Prerequisites

**1. Install Aegis with MCP support**

```bash
pip install aegis-dq[mcp]
```

**2. Add the MCP server to Hermes**

```yaml
# ~/.hermes/config.yaml
mcp_servers:
  aegis:
    command: aegis
    args: [mcp]
    env:
      ANTHROPIC_API_KEY: "${ANTHROPIC_API_KEY}"
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

## How to Run

Ask Hermes to run a rules file against your warehouse:

- "Run `rules/orders.yaml` against BigQuery and tell me what failed."
- "Run rules.yaml against Athena offline — no LLM, just pass/fail."
- "Show me the last 10 validation runs."
- "What was the root cause in yesterday's run?"
- "Search the audit trail for anything about null order IDs."
- "Compare today's run with yesterday's — what newly failed?"

## Quick Reference

All tools are prefixed `mcp__aegis__`:

| Tool | What it does |
|---|---|
| `mcp__aegis__run_validation` | Run a rules YAML against a warehouse; returns pass/fail, diagnosis, remediation SQL |
| `mcp__aegis__list_runs` | List recent run IDs from the audit trail, newest first |
| `mcp__aegis__get_run_report` | Full report for a past run by ID |
| `mcp__aegis__get_trajectory` | Node-by-node LLM decision log for a run |
| `mcp__aegis__search_decisions` | Full-text search across all past LLM decisions |
| `mcp__aegis__compare_reports` | Diff two runs — regressions, fixes, pass-rate delta |
| `mcp__aegis__summarize_reports` | Compact summary of one or more runs |
| `mcp__aegis__check_consistency` | Detect flapping rules and rule-set drift between two runs |
| `mcp__aegis__load_pipeline` | Load a `pipeline.yaml` manifest and return its config as context |

## Procedure

**Run a validation**

`mcp__aegis__run_validation` takes:
- `rules_path` — path to your rules YAML file
- `warehouse` — one of: `duckdb`, `bigquery`, `athena`, `databricks`, `postgres`
- `connection_params` — JSON with connection kwargs (falls back to env vars if omitted)
- `no_llm` — set `true` to skip LLM diagnosis for fast offline checks

**Use a pipeline manifest**

A `pipeline.yaml` captures rules, warehouse, and goal in one file. Call `mcp__aegis__load_pipeline` first — Hermes reads the manifest and calls `mcp__aegis__run_validation` with the right params automatically.

## Pitfalls

- If `connection_params` is omitted and required env vars are missing, the tool returns a clear error listing which variables to set.
- `no_llm: true` skips all LLM calls — useful when no API key is configured.
- Rules referencing tables not present in the warehouse return a SQL error, not a pass.

## Verification

After a run, confirm Hermes used the MCP tools:
- Check that `mcp__aegis__run_validation` appears in the tool call log.
- `mcp__aegis__get_run_report` on the returned run ID should show per-rule pass/fail with diagnosis.

## Links

- Docs: https://aegis-dq.dev/integrations/hermes
- GitHub: https://github.com/aegis-dq/aegis-dq
- PyPI: https://pypi.org/project/aegis-dq/

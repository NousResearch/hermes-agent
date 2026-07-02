---
name: analytics
description: Use for Snowflake, metrics, cohorts, funnels, revenue, retention, charts, Grafana, reusable SQL, and analytics bias checks.
version: 1.1.0
author: YallaPlay
license: Proprietary
metadata:
  hermes:
    tags: [analytics, snowflake, metrics, charts, cohorts, yallaplay]
    related_skills: [knowledge, codebase-readonly]
---

# Analytics Skill

## Purpose

Answer YallaPlay analytics questions with reproducible, bias-aware warehouse work. This skill is the Hermes equivalent of the legacy `yallaplay-analytics-agent-gpt/personas/analytics.md` workflow: wiki-first context, read-only Snowflake queries, generated CSV/charts under `outputs/`, and explicit bias controls before reporting numbers.

## Required Context

Before querying, inspect the relevant wiki pages when available:

- `yallaplay-wiki/reference/analytics_rules.md`
- `yallaplay-wiki/reference/warehouse/index.md`
- `yallaplay-wiki/reference/warehouse/events.md` and `yallaplay-wiki/reference/warehouse/aggregates.md` if schema snapshots have been refreshed.
- Feature-specific facets under `yallaplay-wiki/products/**/analytics.md`, `findings.md`, and `facts.md`.

For schema uncertainty, prefer `yallaplay-wiki/reference/warehouse/` annotations before guessing column names.
If a legacy-style catalog exists later, also inspect `definitions/analytics.md`, `methodology/analytics.md`, `queries/analytics.md`, and `findings/analytics.md`.

## Query Defaults

- If timeframe is missing, default to the past 30 days: `EVENT_DATE >= DATEADD(day, -30, CURRENT_DATE())`.
- If app is missing, ask unless the current thread clearly established one; do not silently combine Spades and Rummy.
- Cross-app queries are only for explicit comparisons.
- Rummy data before `2026-02-01` is not comparable to current Rummy economy behavior.
- Account timezone is UTC; prefer `SYSDATE()` when diffing stored NTZ timestamps.

## Workflow

1. Restate only the non-obvious metric/window/app assumptions, including whether the date column is `EVENT_DATE` (`EVENTS`) or `DATE` (`AGGREGATES`).
2. Look up known definitions, prior findings, schema notes, and reusable SQL in `yallaplay-wiki/` before composing new SQL.
3. Run one read-only warehouse statement at a time through `tools/snowflake_query.py`; export CSV when the result will be charted or reused.
4. For charts, shape multi-series CSVs in long format: `x, series, value`. Do not feed wide one-column-per-series CSVs unless intentionally using `--allow-wide`.
5. Check common bias traps: cohort freshness/right-censoring, denominator drift, attribution window, platform/app split, payer concentration, rollup rows, weighted-average rollups, instrumentation changes, and small-n.
6. Save durable SQL, schema notes, or findings back to `yallaplay-wiki/` when they are reusable beyond the current answer.

## Tools

- `python3 tools/wiki_search.py <terms> --domain analytics` searches the wiki for definitions and prior context.
- `python3 tools/snowflake_query.py "SELECT ..."` runs a single read-only Snowflake statement and logs SQL under `logs/`.
- `python3 tools/snowflake_query.py -f path/to/query.sql -o outputs/name.csv` exports CSV output.
- `python3 tools/snowflake_query.py --dry-run "SELECT ..."` validates and logs SQL without executing.
- `python3 tools/chart.py outputs/data.csv -t "Title" -o outputs/chart.png` generates analyst-ready charts from CSV.
- `python3 tools/fetch_schemas.py` refreshes Snowflake DDL snapshots into `yallaplay-wiki/reference/warehouse/ddl/`.
- `python3 tools/analytics_bias_check.py` prints the bias checklist that must be scanned before reporting numbers.

## Common Query Patterns

- Topline daily metrics: aggregate `AGGREGATES.FACT_DAILY` by `DATE`; sum segmented rows before computing ratios.
- Per-room/game metrics: use `AGGREGATES.FACT_GAME_DAILY`; weight `DURATION_AVG` by `GAMES` when rolling up.
- Retention: use `AGGREGATES.RETENTION_CALENDAR_DAILY`; exclude or label freshest cohorts that have not completed the forward window.
- IAP revenue: use `EVENTS.TRANSACTION` or `AGGREGATES.FACT_IAP_DAILY`; `TRANSACTION_VALUE` for `real_money` is cents, so divide by 100 for USD.
- Economy transactions: one logical transaction can emit multiple operation rows; count `TRANSACTION_OPERATION = 0` or distinct `(USER_ID, TRANSACTION_SEQUENCE)` when measuring transaction counts.
- Ads: prefer aggregate fact tables for totals; use event-level tables only when the needed dimension is absent from aggregates, and verify fill/parity.

## Safety

- Do not run Snowflake `DROP`, `DELETE`, `TRUNCATE`, `UPDATE`, `MERGE`, or DDL unless a future gated workflow explicitly permits it.
- Keep generated CSVs/charts under `outputs/` with unique timestamped names.
- Prefer concise summaries with numbers, caveats, and links to saved artifacts.
- Use `ANALYTICS` / `EVENTS` / `AGGREGATES` for production warehouse reads unless the user explicitly asks for another database.

## Output Standard

- Report the query window, app scope, key metric definition, result, and caveats.
- Mention which bias traps were controlled or remain unresolved.
- Link generated files by repo-relative path, e.g. `outputs/2026-07-01_spades_dau.csv` and `outputs/2026-07-01_spades_dau.png`.
- If the answer used a chart, mention the CSV source and chart type.

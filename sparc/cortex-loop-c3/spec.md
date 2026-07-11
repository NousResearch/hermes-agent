---
type: concept
title: Cortex Loop Engine C3 Specification
phase: 1
status: complete
created_at: '2026-06-23T10:41:00.000Z'
gate_passed: true
ingested_via: 'mcp:put_page'
ingested_at: '2026-06-23T14:52:43.972Z'
source_kind: 'mcp:put_page'
---

# Cortex Loop Engine C3 — Production-Grade Cadence + Spend + Migration

## Ground State Summary

Loop Engine v1/C2 shipped with autonomous verifier + tick-poller infrastructure. Schema has `spend_usd` field but no writers. Tick-poller ignores declared cadence (would burn daily loops every 2 min). One paused scheduled loop exists as test target.

## Functional Requirements

**FR1 — Cadence Enforcement**
- Loops with `schedule` field respect their declared frequency
- Daily loops run once per day max, hourly once per hour, etc.
- Poller skips loops not yet due for next run
- Next run time persists across restarts

**FR2 — Spend Telemetry**
- Workers report realized USD cost after each task completion
- Loop accumulates spend at iteration and total level
- Budget enforcement: loops pause when spend approaches budget_usd

**FR3 — Cron Migration Proof**
- Port one existing Hermes cron to Loop primitive
- Demonstrate equivalent functionality with better observability
- Document migration pattern for remaining crons

## Non-Functional Requirements

**NFR1 — Performance**
- Tick-poller completes one pass in <500ms for 100 loops
- Spend reporting adds <50ms latency to task completion

**NFR2 — Backward Compatibility**
- Existing loops without `next_run_at` migrate gracefully
- Manual loops (no schedule) remain unaffected

**NFR3 — Observability**
- Events emitted: `loop_spend_reported`, `loop_budget_exceeded`, `loop_scheduled_tick`
- Dashboard shows spend burn rate per loop

## Acceptance Criteria

**AC1** — Given a loop with schedule="daily", When tick-poller runs every 2 min, Then loop advances at most once per 24h
**AC2** — Given a loop with schedule="*/30 * * * *", When tick-poller runs, Then loop respects 30-min cadence
**AC3** — Given next_run_at in future, When tick-poller runs, Then loop_tick is skipped with no state change
**AC4** — Given a worker completes a task costing $0.05, When it calls report_spend, Then loop.spend_usd increments by 0.05
**AC5** — Given loop spend reaches 90% of budget, When tick runs, Then loop pauses with reason "approaching_budget_limit"
**AC6** — Given an existing researcher cron, When migrated to Loop, Then it produces equivalent output on same schedule
**AC7** — Given loops table pre-migration, When migration runs, Then next_run_at column exists and loops boot normally
**AC8** — Given spend report with loop_id+iter+task_id, When same tuple reported again, Then spend is idempotent (no double-count)
**AC9** — Given tick-poller crash, When restarted, Then no loops get double-ticked or skipped
**AC10** — Given dashboard /loops page, When loop has spend > 0, Then spend% bar shows correctly

## Edge Cases

**E1** — Clock drift: server time changes backward → next_run_at in future prevents infinite loop
**E2** — Zero-budget loops → spend reporting allowed but first cent triggers pause
**E3** — Malformed schedule string → defaults to manual (no auto-tick) with logged warning
**E4** — Worker dies before reporting spend → task marked failed, spend=0 (conservative)
**E5** — Concurrent spend reports → last-write-wins at iteration level, sum at loop level

## Constraints

- Must ship in single session (~5hr)
- Schema migration must be backward-compatible
- No breaking changes to existing Loop API
- Spend reports must handle OpenRouter/Anthropic/custom provider variance

## Scope Decisions

**In Scope (C3):**
- next_run_at column + tick enforcement
- loop_report_spend() lib function + worker integration
- researcher → Loop migration as proof
- Dashboard spend visualization

**Deferred to C4:**
- Drift detector  
- Loop-of-loops orchestration
- All remaining cron migrations
- Per-task spend breakdown UI

## Resolved Defaults

**Q1** — Cadence parser library? **cron-descriptor** (proven in Hermes cron, handles both cron + natural language)
**Q2** — Spend storage granularity? **Iteration-level** with task_id for attribution but no task table
**Q3** — Which cron to migrate first? **Researcher** — simplest (one fleet role, daily cadence)
**Q4** — Budget enforcement threshold? **90%** — gives headroom for one more iteration to wrap up
**Q5** — Idempotency key for spend? **(loop_id, iter_n, task_id)** tuple, last-write-wins

## User Sign-off

Proceeding with A→B→D plan: cadence enforcement, spend telemetry, researcher migration proof. Gate 1 criteria met.

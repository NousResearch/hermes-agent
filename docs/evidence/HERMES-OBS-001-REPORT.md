# HERMES-OBS-001 — Final Evidence Report

## Goal

Add a durable per-API-call usage observability ledger to the Kanban system so that every primary (conversation loop, Codex app-server) and auxiliary model call made by a Kanban worker is independently and persistently recorded with its full runtime context: board, task, run, call kind, provider, model, token breakdown, cost, parent associations, and accepted-result semantics. The ledger must be additive (non-destructive migration), idempotent, fail-safe, privacy-preserving, and correctly aggregated without double-counting.

## Commits

| # | SHA | Task | Title | Files Changed |
|---|-----|------|-------|---------------|
| 1 | `cfe24230b732d58495edd9fe15e1fffebd414f53` | `t_3ec6b7ad` | Migrate old run_usage schema for HERMES-OBS-001 | `hermes_cli/kanban_db.py`, `tests/hermes_cli/test_kanban_usage_ledger.py` |
| 2 | `5e6c7bb28fd8ac6e3a6ebbf6db5816d3f7502ea6` | `t_ab837761` | Preserve all parent associations on usage events | `hermes_cli/kanban_usage_ledger.py`, `tests/hermes_cli/test_kanban_usage_ledger.py` |
| 3 | `f14ff91d58f27c884a7b99e352afaff189af9f40` | `t_f83934a6` | Normal conversation runtime usage boundary | `agent/conversation_loop.py`, `hermes_cli/kanban_usage_ledger.py`, `tests/hermes_cli/test_kanban_usage_ledger.py` |
| 4 | `2529f0beed5eeb63e7520f9560c2bcaaa5603626` | `t_b30e5e5b` | Codex runtime usage boundary | `agent/codex_runtime.py`, `hermes_cli/kanban_usage_ledger.py`, `tests/hermes_cli/test_kanban_usage_ledger.py` |
| 5 | `31222e52a659a6c4b3b0caaf97ef5242a687ac64` | `t_d03f5df8` | Auxiliary runtime usage boundary | `agent/auxiliary_client.py`, `hermes_cli/kanban_usage_ledger.py`, `tests/hermes_cli/test_kanban_usage_ledger.py` |
| 6 | `b8432084b0535f0872a4316581593e9feb935561` | `t_33f75820` | Usage aggregation across event boundaries | `hermes_cli/kanban_usage_ledger.py`, `tests/hermes_cli/test_kanban_usage_ledger.py` |
| 7 | `d75b5d9ea42e7399cb3edf2e567368ed83b526bf` | `t_09d5a529` | Repair parent runtime-context assertion (R4H) | `tests/hermes_cli/test_kanban_usage_ledger.py` |

**Original base:** `7b48b584ce10621990e9ed27b59ff8bf90986bed`
**Final HEAD:** `d75b5d9ea42e7399cb3edf2e567368ed83b526bf`
**Branch:** `feature/hermes-obs-001`

## Board and Tasks

- **Board:** `hermes-obs-001-20260710-1544z`
- **Checker task:** `t_4f417a0e` — status: `done`, outcome: PASS (run 26)

| Task ID | Title | Assignee | Outcome |
|---------|-------|----------|---------|
| `t_3ec6b7ad` | R4A — Old run_usage schema migration | builder-grok | completed |
| `t_ab837761` | R4B — Preserve all parent task associations | builder-grok | completed |
| `t_f83934a6` | R4C — Normal conversation runtime usage boundary | builder-grok | completed |
| `t_b30e5e5b` | R4D — Codex runtime usage boundary | builder-grok | completed |
| `t_d03f5df8` | R4E — Auxiliary runtime usage boundary | builder-grok | completed |
| `t_33f75820` | R4F — Usage aggregation across event boundaries | builder-grok | completed |
| `t_09d5a529` | R4H — Repair parent runtime-context assertion | builder-qwen | completed |
| `t_4f417a0e` | R4G — Independent complete-chain integration checker | checker | PASS |

## Scope — Files Touched (6 total)

```
agent/auxiliary_client.py
agent/codex_runtime.py
agent/conversation_loop.py
hermes_cli/kanban_db.py
hermes_cli/kanban_usage_ledger.py
tests/hermes_cli/test_kanban_usage_ledger.py
```

## Test Commands and Results

All tests run via:
```
.venv/Scripts/python.exe -m pytest tests/hermes_cli/test_kanban_usage_ledger.py -v --tb=short
```

### Checker run 26 (final PASS) — full suite

```
101 passed in ~195s
```

### Focused groups (independently re-verified)

| Group | Result |
|-------|--------|
| `TestRunUsageSchema` | 7/7 passed |
| `TestMultiParentJoinTable` | 7/7 passed |
| `TestRuntimeContextFields` | 5/5 passed |
| `TestConcurrency` + `TestMultiProcessConcurrency` | 3/3 passed |
| `TestAggregationEventBoundaries` | 8/8 passed |

### R4H verification (post-repair, independently run)

```
.venv/Scripts/python.exe -m pytest \
  tests/hermes_cli/test_kanban_usage_ledger.py::TestRuntimeContextFields \
  tests/hermes_cli/test_kanban_usage_ledger.py::TestMultiParentJoinTable \
  -v --tb=short

12 passed in 1.56s
```

## Runtime Boundaries Verified

### Normal conversation loop (R4C, commit `f14ff91d5`)

- `conversation_loop.py` persists one distinct primary usage event per observable API call via `_record_kanban_usage_at_boundary`.
- Captures: board, task, run, call_kind=primary, provider, model, input/output/cache_read/cache_write/reasoning tokens, elapsed_ms, cost_usd, cost_status.
- api_call_index is stable per run.
- Fail-safe: ledger write failures are caught and logged at debug level, never breaking model execution.

### Codex app-server runtime (R4D, commit `2529f0bee`)

- `codex_runtime.py` `_record_codex_app_server_usage` persists one distinct primary event per observable Codex app-server API call.
- Same boundary function and fail-safe contract as conversation loop.

### Auxiliary runtime (R4E, commit `31222e52a`)

- `auxiliary_client.py` `_validate_llm_response` persists a distinct stable auxiliary event for every observable auxiliary API call.
- Auto-incrementing `api_call_index` via `_aux_call_indices` dict keyed by (board, task, run) — DB-seeded so retries never overwrite index 0.
- `token_source="incomplete"` because auxiliary token observation is often partial.
- Same fail-safe contract.

## Migration Behavior (R4A, commit `cfe24230b`)

- Idempotently adds `accepted_result_tokens` and `api_calls` columns to legacy 25-column `run_usage` tables via `_migrate_add_optional_columns`.
- Creates `run_usage_parents` join table if missing.
- Additive only — no destructive rebuild, no data loss.
- Legacy rows are preserved through the migration.
- Re-running `init_db()` is idempotent.

## Parent Association Behavior (R4B, commit `5e6c7bb28`)

- `record_run_usage` accumulates every parent into `run_usage_parents` via `INSERT OR IGNORE`.
- First denormalized `parent_task_id` is preserved via `COALESCE` on upsert — adding parent B never overwrites parent A.
- `list_parents()` returns all parents for a given event key.
- Repeated recording of the same event-parent association is idempotent.
- Aggregation counts the event once regardless of parent count.

## Auxiliary Event Identity Behavior (R4E, commit `31222e52a`)

- Two auxiliary calls through the boundary function get distinct `api_call_index` values (0, 1, 2, ...).
- Primary and auxiliary events at the same local index are distinct (different `call_kind`).
- Counter is keyed by (board, task, run) so different runs do not collide.
- Previously: all auxiliary calls used index 0, causing silent overwrite — fixed by R4E.

## Aggregation Behavior (R4F, commit `b8432084b`)

- API call count uses the full stable event key: `COUNT(DISTINCT board || '|' || task_id || '|' || run_id || '|' || call_kind || '|' || api_call_index)`.
- Multi-parent events are counted once each via `EXISTS` subquery, not `JOIN` (which would multiply).
- `aggregate_usage()` supports `call_kind` filter to separate primary from auxiliary.
- Events across different runs are correctly aggregated without undercounting or double-counting.
- Accepted-result tokens use `COALESCE(SUM(...), 0)` so NULL unobserved values do not corrupt totals.

## Privacy and Fail-Safe Verification

### Privacy

- No prompt, message content, response body, or credential is ever stored in the ledger.
- Schema columns are exclusively numeric/textual metadata: token counts, identifiers, provider/model names, cost, timestamps.
- Secret rejection works: secrets in token-like fields are rejected.
- Checker independently inspected all three runtime hook sites and confirmed no forbidden data flows to the ledger.

### Fail-safe

- `safe_record_from_canonical_usage` returns `None` on failure (e.g., closed connection), never raises.
- `_record_kanban_usage_at_boundary` skips silently when `HERMES_KANBAN_TASK` is not set (non-Kanban context).
- All three runtime boundaries wrap their ledger writes in `try/except` with `logger.debug` diagnosable trail.
- Codex boundary previously swallowed errors silently without logging — R4D added debug-level logging for diagnosability.

## Checker Provider, Model, and Final PASS

- **Task:** `t_4f417a0e`
- **Checker profile:** `checker`
- **Provider:** `zai` (Z.AI)
- **Model:** `glm-5.2`
- **Fallback:** none (`fallback_providers: []`)
- **Final run:** Run 26, PID 11980, session `20260710_182024_da9da3`
- **Outcome:** PASS — all 101 tests pass, all live integration checks pass, scope clean

### Checker findings summary

| Run | Outcome | Finding |
|-----|---------|---------|
| Run 23 | crashed | OpenAI Codex HTTP 429 — usage limit exhausted (never began substantive work) |
| Run 24 | FAIL — repairable | 1 stale test assertion at line 1552 (implementation correct) |
| Run 26 | PASS | All 101 tests pass; zero findings |

## OpenAI Quota Failure and GLM Recovery

1. **Run 23 crash:** The checker profile was configured with `openai-codex` / `gpt-5.6-sol` with fallback to the same exhausted quota pool. The worker's first API call immediately received HTTP 429 `usage_limit_reached` (plan_type: plus, resets_at: 1783726096). After 3 retries, the worker process died. The dispatcher detected PID 24924 gone and marked it crashed.

2. **Diagnosis:** READ-ONLY investigation confirmed the crash was purely a quota exhaustion — no code, configuration, or logic defect. The checker never executed any tool calls, file inspection, or test commands.

3. **Recovery:** The checker profile was switched to `zai` / `glm-5.2` with `fallback_providers: []` (no route to the exhausted Codex pool). Run 24 executed successfully on GLM 5.2 and produced the first substantive checker verdict.

## Known Orchestrator Defects Discovered

1. **Stale assertion shipped green (R4C → R4G):** The R4C builder ran only R4C-specific focused tests. `TestRuntimeContextFields` was not in scope for R4C, so the stale assertion at line 1552 — which contradicted R4B's auto-population behavior — was not caught until the independent checker ran the full suite. **Lesson:** serial implementation tasks that add tests for shared behavior should include all test groups touching that file in their focused test run, or the checker must run before declaring the chain complete.

2. **Successful workers blocked instead of completing:** builder-grok workers consistently called `kanban_block` with `review-required` instead of `kanban_complete`, even when their work met all acceptance criteria. This forced manual orchestrator intervention for every task. **Root cause:** the task briefs included a "leave all unrelated dirty changes untouched" instruction that the workers interpreted as requiring human review. **Fix applied:** subsequent task briefs explicitly instructed "call kanban_complete on success; call kanban_block only for an actual unmet requirement."

3. **Max-retries=1 caused immediate give_up on checker crash:** The checker task had `max_retries=1`, so the quota-induced crash immediately exhausted the failure limit with zero retry attempts. If the crash had been transient rather than a terminal quota exhaustion, the task would have been permanently blocked.

## Current Git Status

```
## feature/hermes-obs-001
```

Working tree is **clean** — no uncommitted changes. All implementation and test work has been committed across the seven serial commits.

## Remaining Limitations

1. **No CI integration:** The ledger tests run locally only. No GitHub Actions or CI pipeline runs them on push/PR.
2. **No dashboard/query UI:** Usage data is persisted to SQLite but there is no CLI query command or web UI to view aggregated usage. `aggregate_usage()` and `query_usage()` are library functions only.
3. **Auxiliary token observation is incomplete:** Auxiliary calls use `token_source="incomplete"` because caching, reasoning, and provider-side abstraction often hide full token counts. This is documented behavior, not a defect.
4. **Cost tracking is best-effort:** `cost_usd` is only populated when the provider returns authoritative usage data with a computable cost. Auxiliary and some primary calls have `cost_usd=NULL`.
5. **Not merged to main:** The feature branch `feature/hermes-obs-001` is ready for review and merge but has not been pushed or merged.

## Next Recommended Project

**HERMES-OBS-002 — Usage query CLI and aggregation dashboard.**

Expose the ledger data through a user-facing `hermes kanban usage` CLI command with filters (board, task, profile, provider, model, date range, call_kind) and summary output. Optionally add a dashboard panel to the web admin for visualizing usage trends. This makes the observability data actionable rather than just persisted.

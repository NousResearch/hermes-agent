# Hermes PR #3 smoke evidence (repair ae4420)

**Branch:** `feat/hermes-forge-kynver-first-tools` (rebased onto `fork/main`)  
**Repair date:** 2026-05-30  
**AgentOS task:** `ae4420b4-5284-42f8-a33e-c891edddcf1e`

## Git / mergeability

| Check | Result |
|-------|--------|
| Rebased onto current `main` (`4d66efa38`) | Yes — cherry-pick + conflict resolution |
| Prior `mergeStateStatus: DIRTY` | Cleared after rebase (verify on GitHub after push) |

## Targeted tests (local)

```bash
cd hermes-agent
python -m pytest tests/plugins/memory/test_kynver_*.py -q
# 36 passed in ~1.7s (2026-05-30)
```

Coverage includes:

- `in_progress` focus uses `POST …/progress-focus`, never row `running`
- Harness `running` lease read-back → Hermes `pending` unless `inProgressRowKey` matches
- `KynverTodoStore` projection + degraded local fallback
- `inspect_todo_write` / `assert_focus_allowed` pre-transition blocks
- `probe_agentos_health` substrate gating

## Smoke checklist (Lorentz)

| # | Scenario | Status | Evidence |
|---|----------|--------|----------|
| 1 | Forge profile with `KYNVER_PLAN_ID`, creds, health OK | **Not run live** | Requires operator Forge session + prod/staging Kynver ≥ #353 |
| 2 | One `in_progress` todo → Command Center `hermes-todo:{id}` focus | **Unit** | `test_kynver_todo_store.py`, `test_kynver_plan_progress.py` |
| 3 | Harness row `running` → Hermes todo not `in_progress` | **Unit** | `test_kynver_plan_progress.py` (`running` not in focus calls) |
| 4 | AgentOS failure → degraded operating prompt | **Unit** | `test_kynver_substrate.py`, `KynverTodoStore` degraded injection |

## Deploy gate (unchanged)

Kynver production must expose `progress-focus`, `inProgressRowKey`, and `in_progress` row status (Kynver [#353](https://github.com/Totalsolutionsync/Kynver/pull/353)) before enabling Forge substrate in production.

## Handoff

Do **not** merge from this repair worker. Re-request Lorentz deep review after CI green + operator live smoke (items 1–4).

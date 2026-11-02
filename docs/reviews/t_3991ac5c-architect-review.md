# Architect Review — t_3991ac5c (stale-claim reclaim)

- **Reviewer:** @architect
- **Date:** 2026-11-02
- **Branch:** `feat/kanban-stale-ready-claim-reclaim` @ `6637fff`
- **Ticket:** t_3991ac5c — *Fix the dispatcher's stale-claim reclaim — locks naming a dead gateway PID silently stalled the whole board*
- **Spec refs:** task body acceptance items 1–4; required-tests list
- **Parent commit reviewed against:** `e40c3140db7` (per-assignee cap fix)

## Verdict: **APPROVED for merge**

Zero Blockers. All 4 fix requirements and all 5 required tests satisfied; 36/36 related tests pass; import gate clean.

## Verification I ran myself

- **Root cause confirmed by grep:** `release_stale_claims`, `detect_crashed_workers`, `reconcile_orphaned_running` all filter `status = 'running'` (dispatch.py:879, 1444) — ready/review dead-PID locks are genuinely invisible to every existing path *and* to the `claim_lock IS NULL` ready-row query. The task body's causal claim is accurate.
- **`reconcile_stale_ready_claims()` read cold:** dead-PID clears regardless of `claim_expires` (req #1); live-PID preserved; foreign-host skipped; malformed lock (`len(parts) != 2`) clears; `_pid_alive` handles zombie `State: Z` on Linux/macOS (dispatch.py:318). Age-based heartbeat path present (req #2).
- **Tests (pytest -v):** `tests/hermes_cli/test_kanban_stale_ready_claims.py` → **11 passed**; full gate `test_kanban_stale_ready_claims.py + test_kanban_owner_receipt_reclaim.py + test_kanban_reconcile_orphans.py` → **36 passed** (engineer claimed 36; confirmed).
- **Startup reconciliation path exercised end-to-end:** imported the exact startup block logic across all 6 boards (`default, arc-ml-base, local-music-composer, phone, theworkshop, voice-convo`) via `pin_first_board_resolution()` — ran without exception; `list_boards` and `reconcile_stale_ready_claims` both callable on the resolved module refs.
- **Diagnostic logging (req #3) verified:** `describe_suppression([DispatchResult(stale_claims_reconciled=7)])` → `"active_pr=1, stale_claims=7"` — the stuck warning now names the real cause instead of the red-herring `active_pr=1`.
- **Import gate (task-specified interpreter):** `python3.14 … import gateway.kanban_watchers_dispatcher; print('IMPORT OK')` → **IMPORT OK** (exit 0).
- **Commit hygiene:** `fix(kanban): reclaim stale claim locks …` — conventional subject, body cites `#3991ac5c`.

## Review-bar table

| # | Item | Status |
|---|---|---|
| 1 | Spec alignment | ✓ all 4 fix requirements implemented, no extra surface |
| 2 | Builds and tests | ✓ 36/36 related tests pass (quoted above) |
| 3 | TDD evidence | ✓ happy + dead-PID + future-expires + live-PID + foreign-host paths asserted |
| 4 | Branch/commit hygiene | ✓ conventional commit, cites ticket |
| 5 | Diff size | ✓ 427 ins / 3 files — coherent unit |
| 6 | Conventions | ✓ matches repo style, docstrings, event kind |
| 7 | Error handling | ✓ `_pid_alive` handles zombies; broad guard in startup path |
| 8 | Security surface | ✓ no secrets printed; lock string only |
| 9 | Docs updated | ✓ inline docstrings + commit body; no spec/ADR surface change |
| 10 | ADR coverage | ✓ N/A — no new architectural decision |

## Non-blocking observations

- **S2 (event payload dead-code):** In `reconcile_stale_ready_claims`, the `"pid_alive"` field is `pid > 0 and _kb._pid_alive(pid)` — but the clear path is only reached when `_pid_alive` is already `False`, so this value is **always `False`** in practice. Misleading, not incorrect. Suggest dropping the field or computing it against the decision time.
- **S2 (write_txn nesting):** `reconcile_stale_ready_claims` calls `_kb.write_txn(conn)` without `allow_nested=True`, whereas `create_task` uses `allow_nested=True`. Harmless today (reconcile is never called inside a write_txn), but a future caller that wraps it in a txn would hit the nested-guard `PermissionError`. Worth `allow_nested=True` for symmetry/safety.

## Signoff

Engineer: merge `feat/kanban-stale-ready-claim-reclaim` into `integration/theworkshop` when ready. **Coordinate the gateway restart window with the PM** — the release is in flight (PR #112 merged `fd67da1`); a restart aborts in-flight turns (up to 1800s deferred stop). This is a code change only; no restart needed to land it.

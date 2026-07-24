# Verification report — `t_911989ea`

**Gate:** Verify deployed facts and rebase ground truth against fetched remote.
**Head checked:** `a61183b56fdb45b9d2a0f2f6b8482e665ccf702f` (local `HEAD` and `origin/main` identical; `git rev-list --left-right --count HEAD...origin/main` = `0 0`).
**Design doc under verification:** `docs/kanban/worker-session-provenance-schema.md` (untracked, authored by `t_89462c86`).

## Verdict

**No drift.** Every deployed fact called out in the design doc holds against
`origin/main` at the pinned head, and the proposed `board_instance_id` /
`state_db_instance_id` / reciprocal `worker_session_links` schema does NOT
collide with any existing code, table, or migration. The implementation task
(`t_1090554f`) may proceed against this ground truth.

## Deployed facts the design doc relies on (all current)

| Fact | Where it lives on `origin/main` | Status |
|---|---|---|
| Kanban pins `HERMES_KANBAN_TASK` | `hermes_cli/kanban_db.py:8754` (`env["HERMES_KANBAN_TASK"] = task.id`) | current |
| Kanban pins `HERMES_KANBAN_RUN_ID` | `hermes_cli/kanban_db.py:8772-8773` | current |
| Kanban pins `HERMES_KANBAN_CLAIM_LOCK` | `hermes_cli/kanban_db.py:8774-8775` | current |
| Kanban pins `HERMES_KANBAN_BOARD` | `hermes_cli/kanban_db.py:8806-8807` | current |
| Kanban pins `HERMES_KANBAN_DB` | `hermes_cli/kanban_db.py:8801` | current |
| Kanban pins `HERMES_KANBAN_WORKSPACE` / `WORKSPACES_ROOT` | `hermes_cli/kanban_db.py:8755`, `8802` | current |
| Kanban pins `HERMES_PROFILE` (worker profile) | `hermes_cli/kanban_db.py:8812` | current |
| `tasks.session_id` is the originating chat session | `hermes_cli/kanban_db.py:1205-1210` (column doc: "Originating chat/agent session id when the task was created from inside an agent loop that propagated `HERMES_SESSION_ID`.") | current; meaning preserved, not repurposed |
| `task_runs.profile` exists | `hermes_cli/kanban_db.py:1256-1260` (`task_runs` DDL: `profile TEXT`) | current |
| `sessions.profile_name` exists, no task/run provenance on `sessions` | `hermes_state.py:1101` (`profile_name TEXT`); no `task_id` / `run_id` / `board_*` / `state_db_*` columns in the `sessions` table (lines 1056-1105) | current |
| `worker_session_id` completion metadata stamped from `HERMES_SESSION_ID` | `tools/kanban_tools.py:160-165` (`stamped["worker_session_id"] = session_id`); tests in `tests/tools/test_kanban_tools.py:343-364` prove the env value overrides caller-supplied metadata | current; remains diagnostic only — no schema or authority tied to it |
| Kanban `connect()` enables `PRAGMA foreign_keys=ON` (fast + init paths) | `hermes_cli/kanban_db.py:2109`, `2143` | current |
| Writable `SessionDB` enables `PRAGMA foreign_keys=ON` before `_init_schema()` | `hermes_state.py:1669-1671` | current |
| `HERMES_KANBAN_*` keys also documented centrally in `KANBAN_ENV_KEYS` | `agent/delegation_context.py:22-30` | current |
| `HERMES_PROFILE` documented as the comment-authoring profile pin | `hermes_cli/kanban_db.py:8808-8812` | current |

Nothing on the list has drifted since the design was written. The design
doc's "ground truth: `origin/main` at `a61183b56...`" header is faithful to
the live checkout.

## Proposed-schema collision check

Repo-wide ripgrep for `board_instance_id`, `state_db_instance_id`, and
`worker_session_links` returns exactly one hit — the untracked design doc
itself (`docs/kanban/worker-session-provenance-schema.md`). No table,
column, index, or test on `origin/main` references any of the three
identifiers. Implementation has not started.

| Proposed object | Collision? | Notes |
|---|---|---|
| `board_meta` singleton table (board) | none | name unused on `origin/main` |
| `state_db_meta` singleton table (state) | none | name unused on `origin/main`; sits next to existing `state_meta` (FTS storage version marker) without overlap |
| Board `worker_session_links` (7 cols) | none | relation name unused; the new `uq_task_runs_id_task_id` UNIQUE INDEX is also unused |
| State `worker_session_links` (7 cols, FK to `sessions(id)` ON DELETE CASCADE) | none | relation name unused |
| `state_db_instance_id` column on state reciprocal | none | column name unused |
| Bump `SCHEMA_VERSION` 23 → 24 in `hermes_state.py` | safe | `SCHEMA_VERSION = 23` at `hermes_state.py:213`; migration gating at `hermes_state.py:2999-3256` advances schema_version after data migrations; v24 is the next free slot |
| `BEGIN IMMEDIATE` + retry policy mirror | safe | pattern already in use at `hermes_state.py:1665` (explicit `isolation_level=None`) and in kanban `write_txn`; the design doc explicitly forbids routing schema init through `write_txn`'s delegated-child mutation guard |

The proposed DDL is additive and idempotent (`IF NOT EXISTS` everywhere, no
column renames, no existing table touched). The two existing
`PRAGMA foreign_keys=ON` activation points are exactly the ones the design
doc claims they are.

## `claim_task` / `_default_spawn` / `_set_worker_pid` / run-closing impact

- `claim_task` (`hermes_cli/kanban_db.py:3989`): writes `task_runs` with
  `status='running'`, `claim_lock`, `claim_expires`. Adding the
  board-identity CAS in the design doc happens before `task_runs` insert
  and the existing invariant recovery at lines 4034-4049 is preserved.
- `_default_spawn` (`hermes_cli/kanban_db.py:8705`): builds the child env
  from `task.assignee` + claim state. The pinned-env list (lines 8754-8812)
  is exactly what the design doc needs; no new env var is required at
  spawn time — `board_instance_id` and `state_db_instance_id` are read
  from the live singletons inside each phase's transaction, not carried
  in env.
- `_set_worker_pid` (`hermes_cli/kanban_db.py:7736`): pure write to
  `tasks.worker_pid` + `task_runs.worker_pid` + `spawned` event. The
  design doc's worker attachment phase 3 happens strictly before
  `_set_worker_pid` (phase 3 is what makes the session `attached`; the
  pid record follows). No conflict.
- Run-closing: `complete_task` (`hermes_cli/kanban_db.py:4599`) and
  `claim_task`'s reclaim path (lines 4034-4049) write to `task_runs`
  outcome / `ended_at` only. The design doc explicitly says run closure
  marks still-`allocated` board rows `orphaned` and leaves `attached`
  rows `attached`. That is an additive write against
  `worker_session_links.state` and does not touch the run row.

The two-phase attach (board `allocated` → state session+reciprocal
`attached` → board CAS `attached`) slots in cleanly between the claim
CAS (which already does idempotent `ON CONFLICT` semantics) and
`_set_worker_pid`. No existing call site has to change shape; the
contract is purely additive.

## Migration conflict check on `origin/main`

`SCHEMA_VERSION = 23` is the current value. There is no `v24` migration
landed on `origin/main` and no work-in-progress file under
`hermes_state.py` declares `state_db_meta` or `worker_session_links`.
The 900-commit gap called out in the design doc was a 2026-07-24
observation; the current checkout is in sync with the fetched remote,
so any drift that may have existed at design time is now resolved and
the design doc's "fetch and pin current `origin/main`; do not implement
against an installed stale checkout" instruction in its rollout
checklist is satisfied.

## `tasks.session_id` non-repurposing

The column at `hermes_cli/kanban_db.py:1210` is documented as
"Originating chat/agent session id when the task was created from
inside an agent loop that propagated `HERMES_SESSION_ID`." No write
path on `origin/main` ever populates it with a worker transcript id,
and no read path uses it to claim worker-session ownership. The design
doc's "do not repurpose" invariant is consistent with the current code.

## Recommendation

Implementation (`t_1090554f`) may proceed against this ground truth.
There is nothing to route back to the schema design task. Before merge,
the implementer should still:

1. Re-pin `origin/main` and re-run the verification once more
   immediately before opening the PR (this report's freshness is bound
   to head `a61183b56...`).
2. Honor the design doc's mandatory failure-mode test list
   (concurrent first-open convergence, two-DB crash matrix,
   same-cwd-human-CLI non-linkage, profile rename/copy rekey, exact
   reciprocal mismatch, etc.).
3. Treat the untracked design doc as part of the contract — if the
   PR disagrees with it, route the disagreement back to the schema
   design task rather than silently diverging.

## Files cited

- `docs/kanban/worker-session-provenance-schema.md` (untracked; design contract)
- `hermes_cli/kanban_db.py` (`claim_task` 3989, `complete_task` 4599, `_set_worker_pid` 7736, `_default_spawn` 8705, env pin 8754-8812, `connect()` 2080-2163)
- `hermes_state.py` (`SCHEMA_VERSION` 213, `sessions` DDL 1056-1105, writable-init FK pragma 1669-1671, migration gate 2999-3256)
- `tools/kanban_tools.py` (worker-session-id stamp 160-165)
- `agent/delegation_context.py` (`KANBAN_ENV_KEYS` 22-30)
- `tests/tools/test_kanban_tools.py` (worker-session-id behaviour 343-391)

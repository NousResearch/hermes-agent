# Kanban Task Watchers V1 Implementation Plan

> **For Hermes:** Implement this slice in one shared lane with focused tests after each phase. Preserve unrelated local workspace changes.

**Goal:** Add single-session Kanban task watchers that auto-bind on create, survive through the task DB, and hand control back to the bound gateway lane when the task reaches `blocked` or `done`.

**Architecture:** V1 keeps watcher state in a dedicated DB table instead of the `tasks` row, resolves watcher binding from the creating execution context (`HERMES_KANBAN_TASK` watcher row first, then current `HERMES_SESSION_KEY`), and uses a gateway-owned bridge helper to enqueue synthetic internal session events. Because worker completions/blocks happen in subprocesses, the gateway discovers terminal watched-task events from shared Kanban DB state instead of a new IPC channel.

**Tech Stack:** Python, SQLite, Hermes gateway runner, Hermes Kanban CLI/tool surfaces, pytest.

**Source spec:**
- `docs/specs/kanban-task-watchers/00-master-spec.md`

---

## 1) File-by-file change list

### `hermes_cli/kanban_db.py`
Modify.

Responsibilities:
- add `task_watchers` schema + indexes
- add additive migration for legacy boards
- add helper API:
  - `set_task_watcher(...)`
  - `get_task_watcher(...)`
  - `clear_task_watcher(...)`
  - `list_pending_task_watcher_events(...)` (or equivalent)
- extend `Task` to carry `watcher_session_key`
- ensure `get_task(...)` hydrates watcher state
- thread optional `watcher_session_key` through `create_task(...)`
- include watcher cleanup in hard-delete paths

### `hermes_cli/kanban.py`
Modify.

Responsibilities:
- add `--watch` to `kanban create`
- resolve watcher binding from current task watcher first, then current `HERMES_SESSION_KEY`
- log warning and continue when `--watch` cannot resolve a key
- pass resolved watcher binding into `kb.create_task(...)`
- include `watcher_session_key` in JSON task payloads

### `tools/kanban_tools.py`
Modify.

Responsibilities:
- add `watch` boolean to `KANBAN_CREATE_SCHEMA`
- resolve watcher binding with the same precedence as the CLI
- log warning and still create the task when `watch=true` but no key is available
- return `watcher_session_key` in the tool result when present

### `gateway/session.py`
Modify.

Responsibilities:
- add a small `SessionStore.get_entry(session_key)` helper so gateway watcher delivery can resolve the stored origin without poking private maps directly from call sites

### `gateway/run.py`
Modify.

Responsibilities:
- add gateway helper for watcher event enqueueing
- add a background bridge loop that discovers terminal watched-task events from the Kanban DB
- map `completed` -> `Done` and `blocked` -> `Blocked`
- enqueue synthetic internal event text exactly shaped like:
  - `[KANBAN_WATCHER_EVENT] board={board} task_id={task_id} status={Done|Blocked}`
- keep the watcher row alive across successful `Blocked` handbacks by advancing a per-watcher cursor
- clear the watcher row only after successful `Done` handback delivery
- start the new watcher bridge during gateway startup

### `tests/hermes_cli/test_kanban_db.py`
Modify.

Responsibilities:
- cover schema/migration for `task_watchers`
- cover create/get/clear watcher helpers
- cover `create_task(... watcher_session_key=...)`
- cover task hard-delete cleanup for watcher rows

### `tests/hermes_cli/test_kanban_cli.py`
Modify.

Responsibilities:
- cover `create --watch --json` when `HERMES_SESSION_KEY` is present
- cover `create --watch` when no session key is available (task still created, watcher is null)
- cover JSON payload includes `watcher_session_key`

### `tests/tools/test_kanban_tools.py`
Modify.

Responsibilities:
- cover `kanban_create(watch=true)` inheriting watcher from current `HERMES_KANBAN_TASK`
- cover `kanban_create(watch=true)` falling back to `HERMES_SESSION_KEY`
- cover `kanban_create(watch=true)` with no available key still succeeding without a watcher

### `tests/hermes_cli/test_kanban_notify.py`
Modify or extend.

Responsibilities:
- cover gateway watcher bridge delivering a watched `completed` event
- cover gateway watcher bridge delivering a watched `blocked` event
- cover successful delivery clearing the watcher row

---

## 2) Function/method-level responsibilities

### DB layer

Add helpers in `hermes_cli/kanban_db.py`:
- `set_task_watcher(conn, task_id, session_key)`
  - upsert one watcher row per task
- `get_task_watcher(conn, task_id)`
  - return session key or `None`
- `clear_task_watcher(conn, task_id)`
  - delete watcher row
- `list_pending_task_watcher_events(conn)`
  - return one-shot terminal handback candidates by joining `task_watchers` to `task_events`
  - only surface `blocked` / `completed` events at or after watcher creation time
  - select the first terminal event per watched task

Update:
- `Task` dataclass
- `Task.from_row(...)`
- `get_task(...)`
- `create_task(...)`
- `delete_archived_task(...)`
- `delete_task(...)`
- migration helper `_migrate_add_optional_columns(...)`

### CLI layer

Add helper(s) in `hermes_cli/kanban.py`:
- `_current_session_key()`
- `_resolve_create_task_watcher(conn)`

Update `_cmd_create(...)` to:
- parse `args.watch`
- resolve watcher key
- log a warning when unresolved
- pass `watcher_session_key=` into `kb.create_task(...)`

### Tool layer

Add helper(s) in `tools/kanban_tools.py`:
- `_current_session_key()`
- `_resolve_create_task_watcher(kb, conn)`

Update `_handle_create(...)` to:
- parse `watch`
- resolve watcher key
- log warning and continue on miss
- pass `watcher_session_key=` into `kb.create_task(...)`

### Gateway layer

In `gateway/session.py`:
- add `SessionStore.get_entry(session_key)`

In `gateway/run.py` add:
- `enqueue_session_event(...)` or equivalently named helper
  - start-now if idle
  - FIFO enqueue if busy
- `enqueue_task_watcher_event(...)`
  - format watcher event text
  - resolve session entry/origin
  - delegate to session-event helper
- `_kanban_task_watcher_watcher(...)`
  - enumerate boards
  - read pending watcher terminal events
  - enqueue into gateway
  - clear watcher row on successful handback

---

## 3) Exact test cases

### `tests/hermes_cli/test_kanban_db.py`
Add:
1. new board schema contains `task_watchers`
2. migration creates `task_watchers` for a legacy DB
3. `set_task_watcher` + `get_task_watcher` round-trip
4. `create_task(... watcher_session_key=...)` persists watcher state
5. `get_task(...)` exposes `watcher_session_key`
6. `list_pending_task_watcher_events(...)` returns `blocked` and `completed` candidates correctly
7. `delete_archived_task(...)` removes watcher rows
8. `delete_task(...)` removes watcher rows

### `tests/hermes_cli/test_kanban_cli.py`
Add:
1. `create --watch --json` with `HERMES_SESSION_KEY` stores watcher key
2. `create --watch --json` with no session key leaves `watcher_session_key` null and still succeeds
3. `show --json` returns `watcher_session_key`

### `tests/tools/test_kanban_tools.py`
Add:
1. `kanban_create(watch=true)` inherits watcher from current worker task row
2. `kanban_create(watch=true)` falls back to `HERMES_SESSION_KEY`
3. `kanban_create(watch=true)` without either context succeeds and returns no watcher binding

### `tests/hermes_cli/test_kanban_notify.py`
Add:
1. watched task with `completed` event causes gateway watcher bridge to enqueue `[KANBAN_WATCHER_EVENT] ... status=Done`
2. watched task with `blocked` event causes gateway watcher bridge to enqueue `[KANBAN_WATCHER_EVENT] ... status=Blocked`
3. successful enqueue clears the watcher row

---

## 4) Ordered implementation sequence

1. Add the spec + this plan.
2. Add `task_watchers` schema, migration, dataclass field, and DB helpers.
3. Extend `create_task(...)` and `get_task(...)` watcher plumbing.
4. Add `--watch` to CLI create and wire watcher resolution.
5. Add `watch=true` to `kanban_create` tool and wire the same resolution order.
6. Add `SessionStore.get_entry(...)` and the gateway session-event helper.
7. Add the gateway watcher bridge loop and startup hook.
8. Add focused DB / CLI / tool / gateway tests.
9. Run targeted test selection and record remaining gaps.

---

## 5) Scope guardrails

- Do not add dashboard code.
- Do not add explicit watcher clear/reassign/list commands.
- Do not widen to multi-watcher semantics.
- Do not fail create surfaces when watch context is unavailable.
- Do not introduce a new dedicated worker→gateway IPC channel in v1.

---

## 6) Phase exit criteria for the current v1 start

The first usable v1 checkpoint is complete when:
- tasks can be created with watcher binding from CLI and tool surfaces
- watcher binding persists in DB and appears in task detail JSON
- gateway can hand back watched `blocked` / `done` tasks into the bound lane
- targeted tests for the above pass

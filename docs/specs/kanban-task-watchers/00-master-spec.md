# Kanban Task Watchers — Master Spec

Status: Draft (v1 implementation in progress)
Owner: Hermes Kanban / Gateway
Scope: Kanban DB, CLI/tool create surfaces, gateway event bridge, and task-scoped subscription rows. Dashboard and explicit watcher-management UX are out of scope.

---

## 1) Goal

Add a lightweight task watcher binding so a Kanban task can remember one controlling gateway lane and hand control back there when the task reaches `blocked` or `done`.

V1 models this through task-scoped rows in `kanban_notify_subs`, using `delivery_mode='session_event'`. This is a single-lane controller handback feature, not a general multi-subscriber notification system.

---

## 2) In-scope behavior

This slice implements all of the following:

1. Task-scoped watcher bindings stored in `kanban_notify_subs` rather than a dedicated `task_watchers` table.
2. Explicit watcher assignment from `watcher_session_key` / `--watcher-session-key` at task creation time.
3. Explicit validation/parsing of the provided watcher session key into a task-scoped session-event subscription row.
4. Default unwatched task creation when no watcher session key is provided.
5. Hard-failing behavior when `watcher_session_key` / `--watcher-session-key` is provided but invalid.
6. Gateway-side event enqueueing for watched tasks that reach `blocked` or `done`.
7. `watcher_session_key` on the `kanban_create` worker/orchestrator tool.
8. `--watcher-session-key` on `hermes kanban create`.
9. Task detail surfaces remain task-centric and do not invent a synthetic single `watcher_session_key` field; the subscription rows are authoritative.

Out of scope for this slice:

- Dashboard UI.
- User-facing watcher mutation commands (clear, reassign, list-by-session, etc.).
- Broadcast / multi-subscriber semantics.
- Archive-time watcher events beyond the normal terminal cleanup rules.
- A dedicated worker→gateway push IPC channel.

---

## 3) Core model

### 3.1 One controlling lane per task in v1

V1 watcher creation binds one controlling session lane to a task by inserting a task-scoped `kanban_notify_subs` row with:

- `delivery_mode='session_event'`
- a non-empty `session_key`
- the originating `platform`, `chat_id`, and optional `thread_id` / `user_id`

The authoritative watcher state lives in those subscription rows, not in a separate watcher table and not in a derived task JSON shortcut.

### 3.2 Why `kanban_notify_subs`

Using the unified subscription table:

- keeps watcher/session-event delivery aligned with the broader notification model
- avoids parallel watcher-specific schema and cleanup machinery
- preserves enough origin metadata for gateway delivery ownership and routing
- lets worker-created child tasks inherit the same session-event binding without inventing a second representation

### 3.3 Cleanup contract

A watcher session-event subscription must survive intermediate `blocked` handbacks so the same lane can hear about later retries and eventual completion.

V1 cleanup rules:

- `blocked` handback does **not** clear the watcher
- `done` and `archived` are terminal cleanup boundaries
- deleting the task deletes the associated subscription rows through normal task cleanup

---

## 4) Assignment semantics

### 4.1 Create surfaces

Two create surfaces gain explicit watcher assignment:

- CLI: `hermes kanban create ... --watcher-session-key <session_key>`
- Worker/orchestrator tool: `kanban_create(..., watcher_session_key="...")`

### 4.2 Resolution order

When an explicit watcher session key is provided, the create path validates it and derives the task-scoped session-event subscription row from that key.

If no explicit watcher session key is provided, the task is created without a watcher/session-event binding.

### 4.3 Failure contract for explicit watcher session keys

`--watcher-session-key` / `watcher_session_key` is an explicit delivery contract, not a best-effort hint.

If the provided session key is invalid, the create surface must fail with a user-visible error explaining that `watcher_session_key` must be a valid Hermes gateway session key.

This must not silently downgrade into unwatched task creation.

### 4.4 No parent-row inference from the dependency graph

V1 does **not** infer watcher binding from parent task rows or ambient gateway env/session context. The binding is explicit at task creation time.

---

## 5) Delivery semantics

### 5.1 Event kinds that trigger handback

A watched task triggers a handback event when it reaches either:

- `blocked`
- `done` (via the `completed` task event)

V1 event text shape:

```text
[KANBAN_WATCHER_EVENT] board={board} task_id={task_id} status={Done | Blocked}
```

### 5.2 Gateway-owned enqueue helper

The gateway owns the session-event enqueue helper. It takes the bound `session_key` plus resolved origin metadata and enqueues the watcher event onto that lane.

If the lane is idle, processing may begin immediately. If it is busy, the event is appended through the existing session follow-up queue path.

### 5.3 Why the gateway bridges from DB state in v1

Task lifecycle mutations often happen in dispatcher-spawned worker processes, not inside the long-running gateway process. Because there is no dedicated worker→gateway push IPC in this slice, the gateway watcher bridge discovers watched terminal events by reading shared Kanban DB state.

V1 uses task-scoped `kanban_notify_subs` rows plus the task event log/cursor state to find watcher handback events without inventing a parallel push channel.

---

## 6) Surface/representation rules

- Task create/show JSON must remain task-centric.
- The authoritative watcher representation is the subscription row in `kanban_notify_subs`.
- V1 must **not** surface a synthetic single `watcher_session_key` field in task JSON responses.
- Normal human-facing task output should not pretend there is one canonical watcher key when the true binding lives in subscription rows.

---

## 7) Non-goals / guardrails

- Do not add dashboard affordances.
- Do not add watcher reassignment or clearing commands.
- Do not widen this into a general notification system.
- Do not invent multi-watcher semantics.
- Do not silently downgrade explicit `watcher_session_key` requests into unwatched task creation.

---

## 8) Acceptance criteria

1. A task created with `--watch` or `watch=true` binds to a valid session-event subscription when one can be resolved.
2. A watched worker creating child tasks with `watch=true` inherits the current task's session-event binding when one exists.
3. If no valid session-event binding can be resolved, explicit `--watch` / `watch=true` create requests fail and do not create the task.
4. `blocked` handback does not clear the watcher; later retries and eventual completion still target the same lane.
5. The gateway can enqueue `[KANBAN_WATCHER_EVENT] ... status=Blocked|Done` into the bound lane.
6. Watched tasks hand back on `blocked` and `done` without dashboard involvement.
7. Task detail JSON does not surface a synthetic `watcher_session_key`; subscription rows remain authoritative.

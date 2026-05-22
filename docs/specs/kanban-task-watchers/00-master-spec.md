# Kanban Task Watchers — Master Spec

Status: Draft (v1 implementation in progress)
Owner: Hermes Kanban / Gateway
Scope: Kanban DB, CLI/tool create surfaces, gateway event bridge, and task detail JSON. Dashboard and explicit watcher-management UX are out of scope.

---

## 1) Goal

Add a lightweight task watcher binding so a Kanban task can remember one controlling gateway lane and hand control back there when the task reaches either `blocked` or `done`.

This is a single-task, single-session relation intended for orchestrator / controller flows, not a general notification subscription system.

---

## 2) In-scope behavior

This slice implements all of the following:

1. A new task-watcher table with one watcher row per task.
2. Automatic watcher assignment from `watch=true` / `--watch` at task creation time.
3. Watcher inheritance from the creating task environment when a watched worker creates child tasks.
4. Fallback assignment from the current gateway `HERMES_SESSION_KEY` when present.
5. Non-failing behavior when `watch` is requested but no session key can be resolved.
6. Gateway-side event enqueueing for watched tasks that reach `blocked` or `done`.
7. `watch=true` on the `kanban_create` worker/orchestrator tool.
8. `--watch` on `hermes kanban create`.
9. `watcher_session_key` surfaced in task detail JSON responses.

Out of scope for this slice:

- Dashboard UI.
- User-facing watcher mutation commands (clear, reassign, list-by-session, etc.).
- Broadcast / multi-subscriber semantics.
- Archive-time watcher events.
- A dedicated worker→gateway push IPC channel.

---

## 3) Core model

### 3.1 One watcher per task

The watcher relation is stored outside `tasks` in a dedicated table.

Rationale:
- keeps core task lifecycle columns clean
- models watcher binding as a relationship instead of a nullable task attribute
- leaves room for future extension without reshaping the main task row

### 3.2 Table shape

V1 table:
- `task_id TEXT PRIMARY KEY`
- `session_key TEXT NOT NULL`
- `created_at INTEGER NOT NULL`
- `updated_at INTEGER NOT NULL`

The row is task-scoped and unique by `task_id`.

### 3.3 Cleanup contract

Watcher rows are deleted together with the task.

V1 does not expose explicit clear/reassign operations to users. Internally, a watcher row must survive intermediate `Blocked` handbacks so the same lane can hear about later retries and eventual completion. The row is cleared only after a successful `Done` handback delivery or when the task itself is deleted/archived.

---

## 4) Assignment semantics

### 4.1 Create surfaces

Two create surfaces gain watcher assignment:

- CLI: `hermes kanban create ... --watch`
- Worker/orchestrator tool: `kanban_create(..., watch=true)`

### 4.2 Resolution order

When watch is requested, the kernel resolves a watcher session key in this order:

1. **Creating task environment**
   - If `HERMES_KANBAN_TASK` is set, look up that task's watcher row.
   - If found, inherit its `session_key`.
2. **Current gateway session**
   - Else, if `HERMES_SESSION_KEY` is present in the active session context / env, use it.
3. **No binding**
   - Else, create the task normally without a watcher row.
   - Log a warning; do not fail task creation.

### 4.3 No parent-row inheritance in v1

V1 does **not** infer watcher binding from parent task rows. Inheritance is from the *creating execution context*, not from the dependency graph.

---

## 5) Delivery semantics

### 5.1 Event kinds that trigger handback

A watched task triggers a handback event when it first reaches either:
- `blocked`
- `done` (via the `completed` task event)

V1 event text shape:

```text
[KANBAN_WATCHER_EVENT] board={board} task_id={task_id} status={Done | Blocked}
```

### 5.2 Gateway-owned enqueue helper

The gateway owns a helper that takes:
- `session_key`
- resolved session origin/source
- event text

and performs the right session-lane behavior:
- if idle, start processing immediately
- if busy, append to the session FIFO follow-up chain

### 5.3 Why the gateway bridges from DB state in v1

Task lifecycle mutations often happen in dispatcher-spawned worker processes, not inside the long-running gateway process. Because there is no dedicated worker→gateway push IPC in this slice, the gateway watcher bridge must discover terminal watched-task events by reading shared Kanban DB state.

V1 uses the persisted watcher row plus the task event log to find the first terminal watcher handback event after the watcher row was created.

### 5.4 One-shot watcher handback in v1

V1 watcher delivery is one-shot:
- once the gateway successfully enqueues the watcher event for a task
- the watcher row may be removed internally

This keeps the model simple and avoids introducing a separate watcher cursor in v1.

---

## 6) JSON surface

Task detail JSON responses include:
- `watcher_session_key: <string|null>`

This applies to JSON responses that already return full task detail objects (for example create/show surfaces that materialize a full task payload).

---

## 7) Non-goals / guardrails

- Do not add dashboard affordances.
- Do not add watcher reassignment or clearing commands.
- Do not widen this into a general notification system.
- Do not invent multi-watcher semantics.
- Do not fail create paths just because watcher context is unavailable.

---

## 8) Acceptance criteria

1. A task created with `--watch` or `watch=true` binds to exactly one session key when one is available.
2. A watched worker creating child tasks with `watch=true` inherits the current task's watcher binding.
3. Missing session-key context logs a warning and still creates the task.
4. Task detail JSON exposes `watcher_session_key`.
5. The gateway can enqueue `[KANBAN_WATCHER_EVENT] ... status=Blocked|Done` into the bound lane.
6. Watched tasks hand back on `blocked` and `done` without dashboard involvement.

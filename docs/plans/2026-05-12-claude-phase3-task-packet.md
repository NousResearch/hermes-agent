# Claude Code Task Packet — Phase 3 Focused Task Registry Substrate

IMPORTANT: This is a narrow Phase 3 task packet. Implement a coherent, testable task registry substrate only. Do not build Ralph/focused-agent runtime, follow-up classifier, worker lanes, background delegation, or synthesis delivery in this phase.

## Worktree

```text
/tmp/hermes-orchestrator-phase-3
```

## Baseline

```text
63554cbd6 fix(cli): lock pending input drain and enqueue
```

## Source plans/context

- `docs/plans/2026-05-12-hermes-orchestrator-first-update-plan.md`
- `docs/plans/2026-05-12-integrated-busy-queue.md`
- `docs/plans/2026-05-12-orchestrator-implementation-handoff.md`
- `docs/plans/2026-05-12-claude-phase2-task-packet.md`
- `docs/plans/2026-05-12-phase2-structured-pending-queue-notes.md`
- `docs/plans/2026-05-12-claude-phase3-task-packet.md`

## Product intent

Hermes is evolving toward a concierge/front-desk/butler orchestrator:

- The main Hermes process remains accountable for user intent, prioritization, synthesis, and final output.
- Heavy work may be delegated to workers later, but Hermes must remember the active task identity and late user follow-ups.
- When Woo sends many fragmented messages, Hermes eventually needs to decide whether each message is:
  - a follow-up to the active task,
  - a correction/refinement to a running delegated task,
  - a status/cancel request,
  - or a new task.

Phase 3 does NOT make that routing decision. It only provides the task identity/state substrate that Phase 4/5 will use.

## Current phase

```text
Phase 3 — Focused Task Registry Substrate
```

## Goals

Introduce a small `agent/task_registry.py` module that can represent and manage focused user tasks across CLI/gateway/TUI surfaces without changing existing user-facing behavior.

The registry should support:

1. Creating a task from user goal + origin/session metadata.
2. Task statuses:
   - `proposed`
   - `queued`
   - `running`
   - `steerable`
   - `blocked`
   - `done`
   - `error`
   - `cancelled`
3. Attaching pending follow-ups as structured `PendingTurnItem` objects from Phase 2.
4. Tracking worker linkage fields without starting workers yet:
   - `active_worker_id`
   - `worker_kind`
5. Tracking artifacts and notes.
6. Listing active tasks by session key and/or all tasks.
7. Safe serialization/deserialization that excludes or safely serializes any local-only raw payloads.
8. Optional lightweight JSON file persistence if it can be done cleanly with atomic writes. If persistence becomes broad or risky, keep an in-memory registry with explicit serialization helpers and document that durable storage is Phase 4/5.

## Suggested files

Likely new:

```text
agent/task_registry.py
tests/agent/test_task_registry.py
docs/plans/2026-05-12-phase3-task-registry-notes.md
```

Optional only if small and clean:

```text
hermes_cli/commands.py      # add /tasks alias only if cheap and non-invasive
cli.py                     # minimal display hook only if cheap and non-invasive
```

Avoid touching gateway/runtime paths unless strictly necessary for tests. This phase should be a substrate, not behavior rollout.

## Suggested data shape

A reasonable design is:

```python
@dataclass
class TaskOrigin:
    platform: str | None
    chat_id: str | None
    thread_id: str | None
    user_id: str | None
    session_key: str | None

@dataclass
class FocusedTask:
    task_id: str
    session_key: str | None
    user_goal: str
    status: str
    origin: TaskOrigin
    active_worker_id: str | None = None
    worker_kind: str | None = None
    pending_followups: list[PendingTurnItem] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: float = ...
    updated_at: float = ...
```

Registry operations:

```python
create_task(...)
get_task(task_id)
list_tasks(session_key=None, active_only=False)
update_status(task_id, status, ...)
attach_followup(task_id, PendingTurnItem | legacy payload)
attach_artifact(task_id, ...)
add_note(task_id, ...)
cancel_task(task_id, reason=None)
to_dict()/from_dict()
save()/load() if persistence is included
```

## Non-goals / do not build

Do NOT implement:

- Ralph/focused-agent runtime.
- Follow-up classifier or intent model.
- Automatic routing from Telegram/CLI messages into tasks.
- Background worker lanes.
- `delegate_task(background=True)`.
- Gateway delivery/notification changes.
- `/stop <task>` behavior beyond data model helpers.
- Broad SQLite migration unless it is clearly smaller and safer than JSON/in-memory helpers.
- Any credentials, tokens, or secret persistence.

## Acceptance criteria

- Task registry can create, list, update, cancel, and serialize tasks.
- Follow-ups are represented using Phase 2 `PendingTurnItem` and preserve order.
- `raw` payloads from pending items are not serialized/deep-copied.
- Active task filtering excludes `done`, `error`, and `cancelled` by default.
- Unknown/invalid statuses are rejected with clear errors.
- Tests cover:
  - create/list/status transitions
  - follow-up attachment order
  - serialization roundtrip
  - raw passthrough exclusion
  - artifact/note storage
  - active filtering
  - optional JSON persistence if implemented
- Existing Phase 1/2 tests continue passing.

## Required tests

Run at least:

```bash
venv/bin/python -m pytest tests/agent/test_task_registry.py tests/agent/test_pending_turn_queue.py -q
venv/bin/python -m pytest tests/cli/test_busy_queue_coalescing.py tests/cli/test_busy_input_mode_command.py -q
venv/bin/python -m compileall -q agent/task_registry.py agent/pending_turn_queue.py cli.py
```

Also run `git diff --check`.

If you modify CLI command registry or runtime files, add and run targeted tests for those files.

## Return format

Return a concise implementation report with these required sections:

```text
SUMMARY
PURPOSE-FIT DESIGN RATIONALE
WHAT YOU INTENTIONALLY DID NOT BUILD
RALPH/FUTURE FOCUSED-AGENT NOTES
CHANGED FILES
TESTS RUN
RISKS / OPEN QUESTIONS
```

## Worker constraints

- Use Claude Opus-class / `--model opus` if available, `--effort max`.
- Do not silently downgrade to a cheaper model.
- Do not commit.
- Do not push.
- Keep changes within the Phase 3 scope.
- Stop once the substrate is coherent, testable, and reviewable.

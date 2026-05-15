# Task Packet: Frontdesk Phase 1/2 Durable Controller Foundation

## Context

Worktree: `/private/tmp/hermes-frontdesk-live-wiring-20260514-141525`

Primary PRD: `agent-work/frontdesk-always-available-prd.md`

Current state:

- Live frontdesk MVP is mostly implemented and running in the live-wiring worktree.
- Phase A exact `/status` policy and live worker predispatch behavior are mostly implemented.
- Phase B/C scaffold exists: task metadata, worker/reviewer stage fields, review artifact schema, worker-completion-as-review-pending behavior.
- Current worker lane is Hermes oneshot subprocess-backed and in-memory/thread-based; it is not a durable queue.
- Reviewer schema exists, but actual durable reviewer execution/lane is not complete.
- Dirty branch cleanup/commit boundaries should happen before large implementation import.

## Goal

Implement the durable controller foundation for always-available frontdesk without changing live gateway behavior by default.

The implementation should make task/job state restart-safe and prepare the runtime for durable worker/reviewer lanes.

## Scope

Implement Phase 1 plus a small Phase 2 foundation only:

1. Add a durable frontdesk controller store or equivalent SQLite-backed persistence layer.
2. Model task/job/event/artifact records explicitly.
3. Add atomic enqueue/claim/heartbeat/completion helpers.
4. Preserve invariant: worker success queues review/pending-review; it must not mark user-facing completion.
5. Add restart-simulation tests for persisted state.
6. Keep live gateway restart out of scope.
7. Keep broad command UX (`/task`, `/review`, `/import`) out of scope unless needed as internal model stubs.

## Non-goals

- Do not expose a new `/frontdesk` persona/mode.
- Do not change natural-language status routing.
- Do not restart live gateway.
- Do not implement full import/apply UX.
- Do not auto-apply worker artifacts.
- Do not replace all existing in-memory runtime code in one pass; add a durable foundation with compatibility where needed.

## Required invariants

- Exact `/status` only; natural language like `지금 뭐 하고 있어?` remains normal prompt when idle.
- Stop/cancel never queues/replays as ordinary text.
- Session-scoped status/stop must not leak/cancel other sessions’ workers.
- Worker completion cannot become `done_presented` before review pass.
- Duplicate worker completion must be idempotent.
- Durable state must survive controller/store reload.

## Suggested implementation shape

Potential files to modify/create:

- `agent/task_registry.py`
- `agent/orchestration_runtime.py`
- `agent/orchestration_status.py`
- `agent/frontdesk_live.py` only if needed for metadata handoff
- new module if useful: `agent/frontdesk_store.py` or `agent/frontdesk_controller.py`
- tests:
  - `tests/agent/test_task_registry.py`
  - `tests/agent/test_orchestration_runtime.py`
  - new `tests/agent/test_frontdesk_durable_controller.py`
  - existing frontdesk policy/status/runtime suites

Data model minimum:

- Task: id, session key/origin, user goal, state, created/updated timestamps.
- Job: id, task id, kind (`worker`/`reviewer`), state, attempt, lease owner/expiry, pid/session id, heartbeat, exit status.
- Event: task/job id, event type, payload, created timestamp.
- Artifact: id/path/type/producer job/checksum or size if available/import status.

State transition helpers:

- create task + enqueue first worker job atomically.
- claim queued job with lease.
- heartbeat job.
- complete worker job and transition task to `worker_done_pending_review` + enqueue reviewer job exactly once.
- complete reviewer job and transition to review-passed / needs-iteration / blocked / rejected.
- request cancel idempotently.

## Required tests

Add or update tests for:

1. `test_task_job_event_persist_across_restart`
   - create task/job, reload store/controller, assert state preserved.

2. `test_enqueue_task_and_worker_job_atomic`
   - no half-state with task but no first worker job.

3. `test_worker_success_queues_reviewer_not_done`
   - worker completion creates review job and leaves task pending review, not done.

4. `test_duplicate_completion_is_idempotent`
   - same completion submitted twice does not duplicate reviewer jobs/artifacts.

5. `test_expired_lease_transitions_to_recovering_or_requeue`
   - running job with stale heartbeat follows explicit recovery policy.

6. Existing regressions still pass:
   - exact `/status` only.
   - natural-language status-like prompts normal when idle.
   - stop/cancel precedence.
   - session-scoped status/stop isolation.

## Verification commands

Run at minimum:

```bash
python -m py_compile \
  agent/task_registry.py \
  agent/orchestration_runtime.py \
  agent/orchestration_status.py \
  agent/frontdesk_live.py

uv run pytest -o addopts='' \
  tests/agent/test_task_registry.py \
  tests/agent/test_orchestration_runtime.py \
  tests/agent/test_orchestration_status.py \
  tests/agent/test_frontdesk_runtime_loop.py \
  tests/agent/test_control_plane.py \
  tests/agent/test_frontdesk_policy.py \
  tests/gateway/test_frontdesk_live_predispatch.py \
  tests/cli/test_frontdesk_live_predispatch.py \
  tests/tui_gateway/test_frontdesk_live_predispatch.py

git diff --check
```

If adding a new test file, include it in the pytest command.

## Deliverables

- Focused implementation diff.
- Tests passing.
- Short worker summary covering:
  - files changed,
  - state model added,
  - tests run,
  - remaining risks,
  - whether live gateway restart was avoided.

## Commit guidance

Do not commit unless explicitly instructed by the controller. If committing is authorized, use a focused message such as:

```text
feat(frontdesk): add durable controller store foundation
```

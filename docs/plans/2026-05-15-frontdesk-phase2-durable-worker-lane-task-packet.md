# Task Packet: Frontdesk Phase 2 Durable Worker Lane Wiring

## Context

Base worktree: `/private/tmp/hermes-frontdesk-live-wiring-20260514-141525`

Current base commit should include:

- `a84a7d794 feat(frontdesk): add durable controller store foundation`
- `agent/frontdesk_store.py` with SQLite task/job/event/artifact state and claim-token safety.
- `agent/frontdesk_live.py` with the current live default worker lane, Hermes oneshot subprocess runner, artifact paths under `~/.hermes/workers/<task_id>/`, result attachment, and completion notification.

Current state:

- Live frontdesk worker dispatch works using in-memory `OrchestrationRuntime` + `ThreadWorkerLane`.
- Worker completion attaches review-pending metadata to the in-memory task registry.
- A durable SQLite store now exists but is not yet wired into the live worker lane.
- Live gateway restart must not be performed by the worker.

## Goal

Implement a **default-off durable worker lane bridge** so the existing live default worker lane can persist worker task/job lifecycle into `FrontdeskStore` and recover stale/running jobs after a process restart in a controlled way.

The goal is not to replace the whole in-memory runtime in one pass. The goal is to add a compatibility bridge that records the runtime worker lane into SQLite and exposes deterministic recovery helpers/tests.

## Scope

Implement Phase 2 only:

1. Add a small durable bridge in or near `agent/frontdesk_live.py` / a new module if cleaner.
2. Provide a deterministic store path helper, preferably under Hermes home, e.g. `get_hermes_home() / "frontdesk" / "frontdesk.sqlite3"`.
3. Add an explicit feature gate, default off, such as:
   - owner/session config `orchestration.frontdesk_durable_store_enabled`, or
   - an internal helper parameter in tests.
4. When durable bridge is enabled and a worker-shaped request starts:
   - create a durable task + worker job in `FrontdeskStore`;
   - map/record the in-memory task id, session key, source surface, and origin metadata if available;
   - claim the durable worker job with `lease_owner` and `attempt`;
   - record PID/session/artifact metadata when the subprocess starts;
   - heartbeat at least at safe lifecycle checkpoints (start and before completion is acceptable for this pass; no background heartbeat thread required unless simple).
5. On worker success/failure/cancel:
   - complete the durable worker job with current `lease_owner + attempt`;
   - success must transition to `worker_done_pending_review` and enqueue exactly one reviewer job;
   - failure/cancel should transition to an explicit non-presentable state and record error/result metadata.
6. Add a recovery helper that can be called at startup without launching workers:
   - open the durable store;
   - call `recover_expired_leases(now=...)`;
   - return/report recovered jobs/tasks in JSON-safe data.
   - Do not auto-run recovered jobs in this pass.
7. Keep existing in-memory behavior and status text working when the durable gate is off.
8. Do not change exact `/status` status-routing policy.
9. Do not restart the live gateway.

## Non-goals

- No full durable reviewer lane execution.
- No `/task`, `/review`, `/import`, `/discard` UX yet.
- No auto-apply/import of worker artifacts.
- No broad rewrite of `OrchestrationRuntime` or `WorkerLaneRegistry`.
- No live gateway restart.
- No natural-language status heuristic changes.
- No persisting secrets/tokens/API keys.

## Required invariants

- Default-off compatibility: existing frontdesk live tests must pass unchanged when durable gate is off.
- Durable-on worker start records task/job/event/artifact metadata in SQLite.
- Durable completion requires current `lease_owner + attempt` from the claim record.
- Stale/recovered durable jobs cannot be completed by stale workers.
- Worker success never marks `done_presented`; it queues reviewer/pending-review only.
- Cancel/stop must not be queued/replayed as normal text.
- Session-scoped status/stop isolation must not regress.
- SQLite DB/WAL/SHM permissions remain `0600`.

## Suggested files

Likely files:

- `agent/frontdesk_live.py`
- `agent/frontdesk_store.py` only if tiny helper additions are required
- optionally new `agent/frontdesk_durable.py` if separation is cleaner
- tests:
  - `tests/agent/test_frontdesk_durable_worker_lane.py` or similar
  - `tests/gateway/test_frontdesk_live_predispatch.py` additions only if needed
  - existing durable controller tests should continue passing

Avoid large changes to:

- `gateway/run.py` unless absolutely necessary
- `agent/orchestration_runtime.py` unless absolutely necessary
- status policy/classifier modules unless tests prove a need

## Required tests

Add focused tests for:

1. `test_durable_gate_off_preserves_existing_worker_start_behavior`
   - Durable store helper should not be called / no DB created when gate off.

2. `test_durable_worker_start_records_task_and_claimed_job`
   - Enable durable gate with temp DB path.
   - Start a live default worker with a patched subprocess runner.
   - Assert durable task exists, worker job was claimed/running and then completed.
   - Assert reviewer job exists on success.

3. `test_durable_worker_failure_records_error_not_presented`
   - Patched worker raises.
   - Durable worker job becomes failed/cancelled or error state per policy.
   - Task is not `done_presented`.

4. `test_recover_durable_frontdesk_store_requeues_expired_worker_job`
   - Create/claim a job with expired lease.
   - Call startup recovery helper.
   - Assert job is queued and task state reverts according to store policy.

5. Existing tests still pass:
   - `tests/agent/test_frontdesk_durable_controller.py`
   - `tests/agent/test_frontdesk_runtime_loop.py`
   - `tests/gateway/test_frontdesk_live_predispatch.py`
   - status/stop/session isolation suites.

## Verification commands

Run at minimum:

```bash
python -m py_compile \
  agent/frontdesk_live.py \
  agent/frontdesk_store.py \
  agent/orchestration_runtime.py \
  agent/orchestration_status.py \
  agent/task_registry.py

uv run pytest -o addopts='' \
  tests/agent/test_frontdesk_durable_controller.py \
  tests/agent/test_frontdesk_runtime_loop.py \
  tests/agent/test_orchestration_runtime.py \
  tests/agent/test_orchestration_status.py \
  tests/agent/test_control_plane.py \
  tests/agent/test_frontdesk_policy.py \
  tests/gateway/test_frontdesk_live_predispatch.py \
  tests/cli/test_frontdesk_live_predispatch.py \
  tests/tui_gateway/test_frontdesk_live_predispatch.py

git diff --check
```

Include any new test file in the pytest command.

## Deliverables

- Focused diff only.
- No commit; controller will review/import/commit.
- No push.
- No live gateway restart.
- `.codex-worker-summary.md` containing:
  - files changed,
  - durable bridge behavior,
  - feature gate name and default,
  - tests run and results,
  - remaining risks,
  - confirmation that live gateway was not restarted.

## Review traps to avoid

- Do not rely on `lease_owner` alone; use current `attempt` too.
- Do not allow terminal duplicate completion to bypass claim-token checks.
- Do not create SQLite files with permissive modes.
- Do not present user-facing completion before review pass.
- Do not break exact `/status` only routing.
- Do not claim recovery resumes work automatically; this pass should only requeue/report stale jobs unless explicitly implemented and tested.

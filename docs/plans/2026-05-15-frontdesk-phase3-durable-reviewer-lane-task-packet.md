# Task Packet: Frontdesk Phase 3 Durable Reviewer Lane

## Context

Base worktree: `/private/tmp/hermes-frontdesk-live-wiring-20260514-141525`

Current base commit should include:

- `1a1587237 feat(frontdesk): wire durable worker lane bridge`
- `a84a7d794 feat(frontdesk): add durable controller store foundation`
- `agent/frontdesk_store.py` with SQLite durable task/job/event/artifact state, claim-token safety, recovery, reviewer job enqueue on worker success.
- `agent/frontdesk_live.py` with default-off durable worker lane bridge.

Current state:

- Worker-shaped live requests can be mirrored into SQLite when `orchestration.frontdesk_durable_store_enabled` is enabled.
- Successful durable worker completion transitions task to `worker_done_pending_review` and queues a reviewer job.
- Reviewer job execution is not yet wired.
- Live gateway must not be restarted by the worker.

## Goal

Implement a **default-off durable reviewer lane foundation** that can claim queued reviewer jobs, inspect worker result/artifacts, write a review result, and transition tasks according to the review verdict without presenting/importing final output.

This is Phase 3 only. Keep it small and deterministic.

## Scope

1. Add a reviewer lane/helper in or near `agent/frontdesk_live.py` or a new focused module if cleaner.
2. Provide a function that can process one queued reviewer job from `FrontdeskStore`, e.g. `run_one_durable_frontdesk_review(...)`, without starting live gateway.
3. Claim exactly one queued reviewer job using current `lease_owner + attempt` semantics.
4. Read the linked task, worker job result, and artifact metadata.
5. Produce a deterministic review result for this pass. It can be a simple policy/review adapter rather than a full LLM reviewer, but must model verdicts clearly:
   - `passed`
   - `needs_iteration`
   - `blocked`
   - `rejected`
   - `unsafe`
6. On reviewer pass:
   - complete the reviewer job as succeeded/pass;
   - transition the task to `review_passed`;
   - do **not** mark `done_presented`;
   - do **not** import/apply artifacts.
7. On reviewer non-pass:
   - record reviewer result;
   - transition task to a non-presentable or review-needs-work state consistent with existing store constants;
   - do not enqueue unbounded loops.
8. Add JSON-safe status/result payloads with artifact pointers and concise summary.
9. Keep durable reviewer gate default-off or callable only from explicit helper/tests.
10. Do not change exact `/status` routing.
11. Do not restart live gateway.

## Non-goals

- No full `/review`, `/import`, `/discard` UX.
- No automatic apply/import to git, Obsidian, or RKB.
- No live gateway restart.
- No natural-language status heuristic changes.
- No broad rewrite of `OrchestrationRuntime`.
- No secrets/tokens/API keys persistence.
- No unbounded worker/reviewer iteration loop.

## Required invariants

- Reviewer jobs require current `lease_owner + attempt` for heartbeat/completion.
- Stale/recovered reviewer jobs cannot be completed by stale reviewers.
- Worker success remains only `worker_done_pending_review` until reviewer pass.
- Reviewer pass still does **not** mean final presentation/import.
- Reviewer reject/unsafe/blocked must be non-presentable.
- Duplicate reviewer completion must be idempotent only for current terminal token, and stale attempts must fail.
- SQLite DB/WAL/SHM permissions remain `0600`.
- Existing durable controller and worker bridge tests continue passing.
- Default-off live behavior is unchanged.

## Suggested files

Likely files:

- `agent/frontdesk_store.py` if small state/helper additions are needed.
- `agent/frontdesk_live.py` or new `agent/frontdesk_review.py` for reviewer helper.
- `tests/agent/test_frontdesk_durable_reviewer_lane.py`.

Avoid large changes to:

- `gateway/run.py` unless absolutely necessary.
- `agent/orchestration_runtime.py` unless absolutely necessary.
- status policy/classifier modules.

## Required tests

Add focused tests for:

1. `test_reviewer_pass_transitions_task_to_review_passed_not_presented`
   - create task + worker job;
   - complete worker success;
   - run one reviewer;
   - assert reviewer job terminal/pass;
   - assert task is `review_passed`, not `done_presented`.

2. `test_reviewer_reject_or_unsafe_is_non_presentable`
   - force reviewer result to `unsafe` or `rejected`;
   - assert task is error/non-presentable;
   - `mark_done_presented` must fail.

3. `test_reviewer_claim_token_blocks_stale_completion_after_recovery`
   - claim reviewer;
   - expire/recover lease;
   - assert stale attempt cannot complete;
   - new claim can complete.

4. `test_run_one_review_returns_none_when_no_queued_reviewer_job`
   - helper should be safe/no-op.

5. Existing tests still pass:
   - `tests/agent/test_frontdesk_durable_controller.py`
   - `tests/agent/test_frontdesk_durable_worker_lane.py`
   - `tests/agent/test_frontdesk_runtime_loop.py`
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
  tests/agent/test_frontdesk_durable_worker_lane.py \
  tests/agent/test_frontdesk_durable_reviewer_lane.py \
  tests/agent/test_frontdesk_runtime_loop.py \
  tests/agent/test_orchestration_runtime.py \
  tests/agent/test_orchestration_status.py \
  tests/agent/test_control_plane.py \
  tests/agent/test_frontdesk_policy.py \
  tests/gateway/test_frontdesk_live_predispatch.py \
  tests/cli/test_frontdesk_live_predispatch.py \
  tests/tui_gateway/test_frontdesk_live_predispatch.py \
  -q

git diff --check
```

## Deliverables

- Focused diff only.
- No commit; controller will review/import/commit.
- No push.
- No live gateway restart.
- `.codex-worker-summary.md` containing:
  - files changed,
  - reviewer lane behavior,
  - feature gate/call surface,
  - tests run and results,
  - remaining risks,
  - confirmation that live gateway was not restarted.

## Review traps to avoid

- Do not mark final done/presented after review pass.
- Do not allow reviewer completion without current claim token.
- Do not swallow reviewer/storage errors in a way that falsely reports pass.
- Do not change `/status` natural language routing.
- Do not start live gateway or modify live runtime config.
- Do not enqueue infinite review/worker loops.

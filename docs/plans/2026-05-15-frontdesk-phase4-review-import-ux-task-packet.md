# Task Packet: Frontdesk Phase 4 Review / Import UX Surface

## Context

Base worktree: `/private/tmp/hermes-frontdesk-live-wiring-20260514-141525`

Current base commit should include:

- `20f365af7 feat(frontdesk): add durable reviewer lane foundation`
- `1a1587237 feat(frontdesk): wire durable worker lane bridge`
- `a84a7d794 feat(frontdesk): add durable controller store foundation`

Current state:

- Durable SQLite store tracks tasks, jobs, events, artifacts.
- Durable worker lane bridge is default-off and can mirror live worker lifecycle.
- Successful worker completion queues reviewer job.
- Explicit reviewer helper can process one queued reviewer job and transition task to review_passed / failed / blocked / error without presentation/import.
- There is not yet a user/operator UX for listing durable tasks, reviewing review results, presenting summaries, importing/applying artifacts, or discarding results.
- Live gateway must not be restarted by the worker.

## Goal

Implement a **small explicit review/import control surface foundation** for durable frontdesk tasks. This phase should add safe library-level helpers and tests, not live Telegram command wiring yet.

The core invariant: **review pass is still not final done; presentation and import are explicit separate actions.**

## Scope

Implement Phase 4 foundation only:

1. Add helper(s) in a focused module or `agent/frontdesk_live.py` / `agent/frontdesk_store.py` that can:
   - list durable tasks with current task state, latest worker/reviewer job, artifact pointers;
   - read a single task detail by id;
   - present a review-passed task summary by marking `done_presented` only after review pass;
   - record import/discard decisions without applying arbitrary files.
2. If adding import/discard to store:
   - import should only mark artifact import status / task metadata as pending/imported decision, not copy/apply files yet;
   - discard should mark non-destructive discard/closed state or event, not delete artifacts by default.
3. Add explicit functions only. Do **not** wire natural-language commands or live Telegram slash commands in this pass.
4. Preserve exact `/status` routing and STOP/STEER invariants.
5. Add JSON-safe return payloads suitable for future Gateway/TUI/CLI formatting.
6. Keep behavior default-off / explicit-call only.
7. Do not restart live gateway.

## Non-goals

- No live gateway restart.
- No Telegram command wiring yet.
- No natural-language status/import/review heuristics.
- No arbitrary file apply/copy to repo, Obsidian, or RKB.
- No full reviewer LLM semantics.
- No destructive deletion of artifacts.
- No broad rewrite of OrchestrationRuntime.

## Required invariants

- A task can be marked presented only when state is `review_passed`.
- Worker done pending review must not be presentable/importable as final output.
- Review failed/blocked/rejected/unsafe tasks must not be presentable/importable as final output.
- Import/discard helpers must be idempotent.
- Import/discard decisions must be durable events/metadata, JSON-safe, and non-destructive.
- Artifact paths are treated as data/pointers, not executed or shell-expanded.
- Existing durable controller/worker/reviewer tests continue passing.
- SQLite DB/WAL/SHM permissions remain `0600`.

## Suggested files

Likely files:

- `agent/frontdesk_store.py` for artifact import/discard status helpers if needed.
- `agent/frontdesk_live.py` or new `agent/frontdesk_review_surface.py` for JSON-safe UX helper functions.
- `tests/agent/test_frontdesk_review_import_surface.py`.

Avoid large changes to:

- `gateway/run.py` unless absolutely necessary.
- `agent/orchestration_runtime.py` unless absolutely necessary.
- status policy/classifier modules.

## Required tests

Add focused tests for:

1. `test_list_durable_frontdesk_tasks_includes_review_state_and_artifacts`
   - create task -> worker success -> reviewer pass;
   - list helper returns task, latest jobs, artifact pointers, review result.

2. `test_present_review_passed_task_marks_done_presented`
   - review_passed task can be marked `done_presented`;
   - returned payload is JSON-safe and includes presented=True.

3. `test_present_before_review_pass_fails`
   - worker_done_pending_review task cannot be presented;
   - failed/unsafe review task cannot be presented.

4. `test_import_decision_is_idempotent_and_non_destructive`
   - mark artifact/task import decision;
   - repeated call returns same durable state or no duplicate destructive behavior;
   - artifact file path is not executed/copied/deleted.

5. `test_discard_decision_is_idempotent_and_non_destructive`
   - mark discard decision/event;
   - repeated call does not duplicate destructive action.

6. Existing tests still pass:
   - durable controller
   - durable worker lane
   - durable reviewer lane
   - frontdesk runtime/status/stop/session isolation suites.

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
  tests/agent/test_frontdesk_review_import_surface.py \
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
  - review/import surface behavior,
  - call surface / feature gate,
  - tests run and results,
  - remaining risks,
  - confirmation that live gateway was not restarted.

## Review traps to avoid

- Do not apply/copy/delete artifact files in this phase.
- Do not make worker success final.
- Do not import before review pass.
- Do not let artifact paths become shell commands.
- Do not change `/status` natural language routing.
- Do not add live gateway command wiring yet.

# Phase 6 Orchestration Observatory Notes

## Summary

Phase 6 adds a read-only orchestration observability layer in `agent/orchestration_status.py` with tests in `tests/agent/test_orchestration_status.py`.

It formats the existing task/worker substrates into concise, Telegram-friendly status boards:

- `build_snapshot(...)`
- `format_tasks(...)`
- `format_agents(...)`
- `format_overview(...)`
- `looks_like_orchestration_status_query(...)`
- `OrchestrationStatusFormatter` facade

## PURPOSE-FIT DESIGN RATIONALE

Woo compared the desired Hermes experience to a smooth Manus-like coordinator with worker agents behind it, and to Claude's agent view. The immediate need is visibility, not a heavy new runtime.

The module therefore gives Hermes a read-only "front desk board" over the already-built substrates:

- Phase 3 `TaskRegistry`: focused tasks, statuses, follow-ups, notes, artifacts, worker linkage.
- Phase 4 `WorkerLaneRegistry`: worker handles, result/error/cancel status, lane/task linkage.
- Phase 5 `FollowupRouter`: later natural-language status queries can call this board without mutating state.

This is enough to answer "what are you working on?", "what agents are running?", and "is anything blocked?" once a runtime registry is injected, without prematurely building a full dashboard or DB.

## WHAT YOU INTENTIONALLY DID NOT BUILD

This phase does not implement:

- Ralph runtime.
- LLM/model classifier.
- automatic Telegram/gateway natural-language routing.
- force kill / force cancel.
- public `delegate_task(background=True)` API.
- durable routing DB / SQLite schema.
- global singleton task/worker registry.
- broad CLI/TUI/gateway refactor.
- worker result delivery/synthesis pipeline.

It also does not wire `/tasks` and `/agents` into production command handlers yet. There is not yet a long-lived runtime task/worker registry for those commands to read, so adding command strings now would risk a misleading empty dashboard. The formatter is ready for a thin command handler once registry injection exists.

## RALPH/FUTURE FOCUSED-AGENT NOTES

A future Ralph/focused-agent layer can use this status module as its visibility surface:

```text
Ralph/focused task runtime
→ TaskRegistry + WorkerLaneRegistry
→ OrchestrationStatusFormatter
→ natural-language status / /tasks / /agents
```

This separates concerns cleanly:

- runtime owns work
- registry owns state
- router owns follow-up policy
- status formatter owns presentation
- Hermes main owns synthesis/accountability

## Validation

Controller-run validation:

```text
/Users/wookim/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/agent/test_orchestration_status.py \
  tests/agent/test_followup_router.py \
  tests/agent/test_worker_lanes.py \
  tests/agent/test_task_registry.py \
  tests/agent/test_pending_turn_queue.py -q

117 passed, 8 warnings
```

```text
/Users/wookim/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/cli/test_busy_queue_coalescing.py \
  tests/cli/test_busy_input_mode_command.py \
  tests/gateway/test_restart_drain.py \
  tests/gateway/test_session_race_guard.py -q

65 passed, 8 warnings
```

```text
/Users/wookim/.hermes/hermes-agent/venv/bin/python -m compileall -q \
  agent/orchestration_status.py agent/followup_router.py agent/worker_lanes.py agent/task_registry.py agent/pending_turn_queue.py cli.py gateway/run.py

git diff --check

passed
```

## Risks / Follow-up

- `/tasks` and `/agents` commands should be added only once there is an injected runtime registry or a clear session-local registry surface to inspect.
- Natural-language gateway status routing should be a later thin integration phase using `looks_like_orchestration_status_query(...)` plus `format_overview(...)`.
- Durable DB remains deferred until restart recovery becomes a concrete requirement.
- The formatter is intentionally read-only and best-effort; missing/foreign worker metadata should degrade to compact lines rather than fail the user-facing status request.

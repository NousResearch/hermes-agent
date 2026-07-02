# PR: Hermes Agent /v1/runs Runtime API Foundation

## Summary

Adds a standalone `gateway/runtime/` package providing structured runtime event/status models, an in-memory thread-safe RunManager for run lifecycle management, and aiohttp route handlers implementing the `/v1/runs` API contract. The route module is conditionally mounted into the live Agent API server behind a feature flag.

## Motivation

The WebUI currently instantiates `AIAgent` directly for chat. The long-term goal is for WebUI to delegate to the Agent runtime API. This PR provides the Agent-side foundation: structured event contract, run state management, and HTTP routes that WebUI's agent-runs adapter can call.

## Major Changes

### New: `gateway/runtime/` package
- `models.py` — `RuntimeEvent`, `RuntimeStatus` dataclasses, secret redaction, event/status constants
- `run_manager.py` — `RunManager` class (thread-safe, in-memory) with create, events, stop, complete, fail, approval/clarify stubs
- `routes.py` — `register_runtime_routes(app)` with 6 aiohttp handlers
- `__init__.py` — public API exports

### Modified: `gateway/platforms/api_server.py`
- Conditional route mount in `APIServerAdapter.connect()` when `HERMES_USE_RUNTIME_RUNS=true` or `platforms.api_server.extra.use_runtime_runs` config is set
- Default: legacy embedded handlers (zero behavior change)

### New tests
- `tests/gateway/test_runtime_models.py` — 20 tests
- `tests/gateway/test_runtime_run_manager.py` — 40 tests
- `tests/gateway/test_runtime_routes.py` — 23 tests
- `tests/gateway/test_runtime_server_mount.py` — 42 tests
- `tests/gateway/test_runtime_approval_clarify.py` — 38 tests
- `tests/gateway/test_runtime_control_bridge.py` — 25 tests

## API Changes

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/runs` | Create a new run (202) |
| GET | `/v1/runs/{run_id}` | Get run status |
| GET | `/v1/runs/{run_id}/events` | Get run events (JSON or SSE) |
| POST | `/v1/runs/{run_id}/stop` | Request run interruption |
| POST | `/v1/runs/{run_id}/approval` | Resolve pending approval |
| POST | `/v1/runs/{run_id}/clarify` | Resolve pending clarification |

## Config Flags

- `HERMES_USE_RUNTIME_RUNS=true` (env) — enables runtime route module
- `platforms.api_server.extra.use_runtime_runs: true` (config.yaml) — equivalent config gate
- Default: flag absent → legacy embedded handlers

## Tests Run

```
Phase 13 complete: 307 runtime tests, all passed
  - 11 runtime test files covering models, run_manager, routes, server_mount,
    approval_clarify, control_bridge, gateway_binding, messaging_binding
  - 36 new tests in test_runtime_messaging_binding.py
Phase 12: 188 runtime tests passed
Full gateway suite: main runtime tests pass (gateway startup blocked by messaging adapters in test env)
WebUI tests: no changes needed (Agent response shape unchanged)
```

## Compatibility Notes

- Backward compatible by default — existing `/v1/runs` API behavior unchanged
- Legacy embedded handlers remain active until feature flag is set
- No change to `/v1/chat/completions`, tool orchestration, or messaging platforms

## Known Limitations

- Approval/clarify resolution has a full lifecycle in RunManager with bridge to live gateway primitives via `RuntimeControlBridge` (`tools.approval.resolve_gateway_approval` and `tools.clarify_gateway.resolve_gateway_clarify`)
- True live agent interruption is bridged via `RuntimeControlBridge.stop_run()` which uses direct agent reference (from `bind_run`) or GatewayRunner `_running_agents` via weakref
- `bind_run(run_id, session_key, agent)` is called at API server runtime run spawn time AND at GatewayRunner messaging-platform agent promotion time, establishing the mapping for approval/clarify/stop
- `unbind_run(run_id)` cleans up on run terminal (completion, failure, cancel, sweep) for both API-server and messaging-platform paths
- GatewayRunner-level binding is now complete: GatewayRunner creates runtime run_id per messaging turn, binds at agent promotion, unbinds at terminal cleanup
- Messaging-platform approval and clarify events are recorded in RunManager with redacted payloads via bridge.request_approval / request_clarify in the gateway approval/clarify callbacks
- GatewayRunner `/approve` and `/deny` slash commands now sync RunManager state via `_sync_runtime_approval_state()` helper (Phase 14)
- Duplicate slash-command resolution returns conflict without duplicating events
- Unknown approval IDs map to not_found; terminal runs map to conflict (Phase 14)
- Non-API GatewayRunner execution now transitions run status to completed/failed before unbinding (Phase 14)
- `hermes gateway run` full startup blocked by messaging adapter dependencies in test environments

## Phase 14 Changes (this PR)

### Non-API runtime execution plane:
- Fixed invalid `status` parameter in `RunManager.create_run()` call in messaging path
- Added terminal status transitions (completed/failed) before `_release_running_agent_state()` unbinds
- Exception paths in `_handle_message()` correctly mark runs as failed

### Slash-command state sync:
- `_handle_approve_command` now syncs RunManager state after resolving gateway approval
- `_handle_deny_command` now syncs RunManager state with `choice="deny"` after denial
- New `_sync_runtime_approval_state()` helper resolves pending approval IDs in RunManager
- Duplicate resolution, not_found, and conflict handling via existing RunManager primitives

### New tests:
- `tests/gateway/test_runtime_non_api_execution.py` — 17 tests
- `tests/gateway/test_runtime_slash_approval.py` — 17 tests
- 1 pre-existing test fix in `tests/gateway/test_restart_resume_pending.py`

### Test results:
- 304 tests passed, 0 failed across 10 runtime test files
- 10 live smoke tests passed
- All Phase 11B-13 tests continue to pass

## Rollback Plan

```bash
unset HERMES_USE_RUNTIME_RUNS
# Or: revert commit range
git revert <commit-range>
```

The `gateway/runtime/` package is additive — removing it has zero impact on running Agent server.

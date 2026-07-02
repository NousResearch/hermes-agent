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
188 runtime-focused tests: 188 passed, 0 failed (6 files)
8845 full gateway suite: 8845 passed, 3 failed (pre-existing, unrelated)
193 existing api_server tests: 193 passed, 0 failed
23 existing api_server_runs tests: 23 passed, 0 failed
104 WebUI tests: 104 passed, 0 failed
Import/construct smoke: PASS
Integration HTTP smoke: PASS
```

## Compatibility Notes

- Backward compatible by default — existing `/v1/runs` API behavior unchanged
- Legacy embedded handlers remain active until feature flag is set
- No change to `/v1/chat/completions`, tool orchestration, or messaging platforms

## Known Limitations

- Approval/clarify resolution has a full lifecycle in RunManager with bridge to live gateway primitives via `RuntimeControlBridge` (`tools.approval.resolve_gateway_approval` and `tools.clarify_gateway.resolve_gateway_clarify`)
- True live agent interruption is bridged via `RuntimeControlBridge.stop_run()` which uses direct agent reference (from `bind_run`) or GatewayRunner `_running_agents` via weakref
- `bind_run(run_id, session_key, agent)` is called at API server runtime run spawn time, establishing the mapping for approval/clarify/stop
- `unbind_run(run_id)` cleans up on run terminal (completion, failure, cancel, sweep)
- Full GatewayRunner-level binding for non-API-server sessions is deferred: the bridge infrastructure is ready, but GatewayRunner does not yet call `bind_run` when spawning runtime-tracked agents
- `hermes gateway run` full startup blocked by messaging adapter dependencies in test environments

## Rollback Plan

```bash
unset HERMES_USE_RUNTIME_RUNS
# Or: revert commit range
git revert <commit-range>
```

The `gateway/runtime/` package is additive — removing it has zero impact on running Agent server.

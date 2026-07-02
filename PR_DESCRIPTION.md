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

## Phase 15 Changes

### Cross-Repo Integration Verification

Phase 15 audits the runtime architecture gap and adds integration contract tests between hermes-agent and hermes-webui.

**Execution-plane gap audit:**
- POST /v1/runs in the standalone `gateway/runtime/routes.py` module is control-plane-only
- The API server adapter has its OWN execution handler (separate route registration)
- Missing primitives documented: execute flag, AIAgent factory, background task manager, event streaming, live agent notification wiring
- Option B selected — execution explicitly deferred to a future "runtime executor" service

**New cross-repo contract tests:**
- `tests/gateway/test_runtime_cross_repo_contract.py` — 25 tests verifying Agent response shapes match WebUI agent-runs adapter expectations
- `tests/gateway/test_runtime_v1_runs_execution_gap.py` — 16 tests documenting the execution gap

**Verification results:**
- 345 tests passed, 0 failed across 12 runtime test files
- WebUI: 138 passed, 0 failed (default mode); 130 passed, 8 expected failures (agent-runs mode)
- All prior Phases 11B-14 behavior preserved

## Phase 16 Changes (this PR)

### Runtime Executor Service

Implements a dedicated runtime executor service that processes queued runs from RunManager through a configurable AgentFactory.

**New: `gateway/runtime/executor.py`**
- `RuntimeExecutor` — `execute_run(run_id)`, `run_once()`, `start()`/`stop()` (background poll), `cancel_run(run_id)`
- `AgentFactory` (Protocol) — pluggable agent creation interface
- `FakeAgentFactory` — deterministic fake for unit tests
- `SessionKeyFactory` — generates session keys for executor-owned runs
- Error redaction via `agent.redact.redact_sensitive_text`
- Resource cleanup in `finally` blocks

**Modified: `gateway/runtime/run_manager.py`**
- `claim_queued_run(run_id)` — atomic worker-safe claim (queued → running)
- `list_runs()` — internal snapshot for executor polling

**Modified: `gateway/runtime/routes.py`**
- Added optional `executor` parameter to `register_runtime_routes()`
- POST /v1/runs with `execute: true` body flag spawns background executor task
- Backward compatible: without executor or execute flag, route stays control-plane-only

**Modified: `gateway/runtime/__init__.py`**
- Exported `RuntimeExecutor`, `AgentFactory`, `FakeAgentFactory`, `SessionKeyFactory`

### Key Design Decisions

1. **Executor is optional** — routes work without it, existing behavior unchanged
2. **AgentFactory is a pluggable protocol** — tests use FakeAgentFactory, production needs DefaultAgentFactory with credentials
3. **POST /v1/runs default is still control-plane-only** — caller must opt in with `execute: true`
4. **Executor uses RunManager as source of truth** — status/events driven through RunManager, never bypassed
5. **Executor uses RuntimeControlBridge** for live binding and stop/cancel
6. **Resource cleanup in finally** — unbind always fires even on failure/cancellation
7. **Secrets redacted** — all error messages pass through `redact_sensitive_text`

### New Tests

- `tests/gateway/test_runtime_executor.py` — 30 tests (construction, lifecycle, bindings, events, cancellation, approval/clarify, background loop, factories)
- `tests/gateway/test_runtime_executor_routes.py` — 8 tests (opt-in routes, backward compat, failure events, stop, approval/clarify via routes)
- `tests/gateway/test_runtime_v1_runs_execution_gap.py` — updated (asyncio.create_task now expected)

### Test Results

```
14 runtime test files: 383 passed, 0 failed
  - test_runtime_models.py: 20 tests
  - test_runtime_run_manager.py: 40 tests
  - test_runtime_routes.py: 23 tests
  - test_runtime_server_mount.py: 42 tests
  - test_runtime_approval_clarify.py: 38 tests
  - test_runtime_control_bridge.py: 33 tests
  - test_runtime_gateway_binding.py: 38 tests
  - test_runtime_messaging_binding.py: 36 tests
  - test_runtime_non_api_execution.py: 17 tests
  - test_runtime_slash_approval.py: 17 tests
  - test_runtime_cross_repo_contract.py: 25 tests
  - test_runtime_v1_runs_execution_gap.py: 16 tests
  - test_runtime_executor.py: 30 tests (new)
  - test_runtime_executor_routes.py: 8 tests (new)
```

### Smoke Test Results

8/8 deterministic smoke tests passed:
1. POST /v1/runs execute=true → queued
2. GET /v1/runs/{run_id} → completed
3. GET events → 3 events
4. Backward compat: no execute flag stays queued
5. Failure path: no traceback leak
6. Stop/cancel: cancelled
7. Approval: mechanism works
8. No secret leak

## Phase 17 Changes (this PR)

### DefaultAgentFactory Implementation

Implements a production-capable `AgentFactory` that constructs real `AIAgent` instances from gateway configuration.

**New: `gateway/runtime/agent_factory.py`**
- `DefaultAgentFactory` — constructs AIAgent using `_resolve_runtime_agent_kwargs()` + `_resolve_gateway_model()`
- `create_default_agent_factory()` — convenience function for wiring
- `create_runtime_executor_with_default_factory()` — one-liner to create wired executor
- Supports dependency injection via `agent_kwargs` for tests
- Validates: missing API key, missing provider, missing model → clean RuntimeError (no secrets)
- Error redaction via `agent.redact.redact_sensitive_text`

**Modified: `gateway/runtime/executor.py`**
- `execute_run()` now handles synchronous AIAgent.run_conversation (real AIAgent) and async agents (FakeAgentFactory) via `asyncio.iscoroutinefunction()` detection
- Sync methods wrapped in `run_in_executor` to avoid blocking the event loop

**Sync/async design:**
| Agent type | run_conversation | Handling |
|---|---|---|
| FakeAgentFactory._FakeAgent | `async def` | `await` directly |
| Real AIAgent | `def` (sync) | `run_in_executor` |

### Key Design Decisions

1. **DefaultAgentFactory reuses existing GatewayRunner credential resolution** — no duplicate logic
2. **Sync/async detection is automatic** — agents declare their nature via coroutine vs def
3. **FakeAgentFactory unchanged** — all 30 existing executor tests pass
4. **Real AIAgent execution verified** with DeepSeek (deepseek-v4-flash)
5. **Errors are redacted** — `agent_creation_failed` replaces raw stack traces

### New Tests

- `tests/gateway/test_runtime_default_agent_factory.py` — 15 tests (construction, validation, integration with executor, helper functions)

### Test Results

```
15 runtime test files: 398 passed, 0 failed
  - test_runtime_default_agent_factory.py: 15 tests (new)
  - All 383 Phase 16 tests preserved
```

### Smoke Test Results

**Deterministic (6/6 passed):**
1. FakeAgentFactory execution → completed
2. No factory → not_supported
3. DefaultAgentFactory injection → clean creation failure
4. Missing credentials → clean RuntimeError
5. Approval/clarify compatibility → triggered and resolved
6. Stop/cancel → cancelled

**Real-credential (passed):**
- Provider: DeepSeek (deepseek-v4-flash)
- POST /v1/runs execute:true → completed

### Enablement

```bash
# Runtime executor with DefaultAgentFactory:
export HERMES_USE_RUNTIME_RUNS=true
# Then wire executor:
#   executor = create_runtime_executor_with_default_factory(run_manager, control_bridge=cb)
#   register_runtime_routes(app, executor=executor)

# Required config:
#   config.yaml: model.provider (e.g. deepseek), model.default (e.g. deepseek-chat)
#   .env: DEEPSEEK_API_KEY=<key>
```

### Phase 18 — Cross-Repo Live HTTP Smoke Harness

**New files:**
- `scripts/standalone_runtime_server.py` — Minimal aiohttp server for runtime smoke
- `scripts/smoke_runtime_executor_live.sh` — Agent-only live smoke (7 tests)
- `scripts/smoke_cross_repo.sh` — Combined Agent + WebUI cross-repo smoke (11 tests)
- `tests/gateway/test_runtime_live_http_smoke.py` — 11 pytest tests
- `docs/runtime-live-smoke.md` — Documentation

**Smoke verified:**
1. Agent direct POST /v1/runs execute:true → completed + done events
2. WebUI proxied run status → terminal state
3. WebUI proxied run events → done event present
4. WebUI cancel/stop → proxies correctly
5. WebUI runtime capabilities → agent-runs mode
6. WebUI deployment health → agent-runs adapter
7. Agent approval/clarify → action_not_found (no pending action)

**Run:**
```bash
cd hermes-agent
scripts/smoke_cross_repo.sh --fake    # deterministic (no credentials)
DEEPSEEK_API_KEY=<key> scripts/smoke_cross_repo.sh  # real DeepSeek
```

**Results:**
- Agent tests: 409 passed, 0 failed (16 files)
- WebUI tests (default): 146 passed, 0 failed
- WebUI tests (agent-runs env): 138 passed, 8 expected failures
- Agent-only smoke (--fake): 7/7 PASSED
- Cross-repo smoke (--fake): 11/11 PASSED
- Real DeepSeek smoke: SKIPPED (no key in this env)

### Phase 19 Additions

- `standalone_runtime_server.py --fake` now wires `FakeAgentFactory` with `request_approval`/`request_clarify` callbacks, generating `approval.requested` and `clarify.requested` events during deterministic execution.
- Smoke scripts verify approval/clarify event presence in run events.
- `docs/messaging-adapter-live-smoke.md` documents credential matrix for 18 adapters, smoke steps, and secret redaction.

**Approval/clarify live pending-action smoke:** Partial — events verified; full e2e lifecycle resolution deferred (fake agent completes immediately after requesting approval, requiring a delay-based or pause-before-complete mechanism for non-terminal resolution).

## Rollback Plan

```bash
unset HERMES_USE_RUNTIME_RUNS
# Or: revert commit range
git revert <commit-range>
```

The `gateway/runtime/` package is additive — removing it has zero impact on running Agent server. Smoke scripts are standalone and do not affect production.

---

## Phase 20 -- Real Smoke Readiness

Date: 2026-07-02

### Verification summary

- Deterministic Agent-only runtime smoke: PASSED, 7 passed, 0 failed, 1 skipped.
- Deterministic cross-repo Agent to WebUI smoke: PASSED, 11 passed, 0 failed.
- DEEPSEEK_API_KEY: not present in the active environment.
- Real DeepSeek Agent-only smoke: SKIPPED.
- Real DeepSeek cross-repo smoke: SKIPPED.
- Provider/model: N/A because no real credential smoke ran.
- WebUI proxied status/events: PASSED via deterministic cross-repo smoke.
- Cancel/stop: PASSED via deterministic smoke.
- Selected reference messaging adapter: Telegram.
- Reference messaging adapter live smoke: SKIPPED because Telegram credentials and a safe test chat were unavailable.
- Approval/clarify full e2e resolve-while-non-terminal: deferred.
- API-server runtime path: preserved.
- Messaging-platform runtime binding: preserved.
- Slash-command state sync: preserved.
- RuntimeExecutor: preserved.
- DefaultAgentFactory: preserved.

### Remaining deferred items

1. Run real DeepSeek Agent-only smoke with DEEPSEEK_API_KEY.
2. Run real DeepSeek cross-repo Agent to WebUI smoke with DEEPSEEK_API_KEY.
3. Run Telegram reference messaging-adapter live smoke with TELEGRAM_BOT_TOKEN and a safe private test chat.
4. Complete full approval/clarify e2e resolution while the run is non-terminal.

---

## Phase 21 -- Pauseable Pending-Action Smoke

Date: 2026-07-02

### Summary

Phase 21 adds deterministic fake-mode delay support so approval and clarify pending actions can be resolved while an executor-owned run is still non-terminal.

### Changes

- Added fake-mode delay support to the runtime fake agent path.
- Added --fake-delay-seconds to scripts/standalone_runtime_server.py.
- Added scripts/smoke_runtime_pending_actions.sh.
- No production-only injection endpoints were added.
- Real DefaultAgentFactory execution remains unchanged.

### Verification

- Pending approval and clarify IDs are visible while terminal=false.
- Approval resolution removes apr-fake-001.
- Clarify resolution removes clar-fake-001.
- approval.resolved and clarify.resolved are each appended exactly once.
- The delayed fake run completes after pending-action resolution.
- Existing deterministic runtime and cross-repo smokes continue passing.

---

## Phase 21 -- Final Verification Results

Date: 2026-07-02

### Summary

Phase 21 completed deterministic pauseable pending-action smoke coverage for executor-owned runtime runs.

### Verification

- Pauseable pending-action smoke: PASSED.
- Existing Agent deterministic smoke: PASSED, 7 passed, 0 failed.
- Cross-repo deterministic smoke: PASSED, 11 passed, 0 failed.
- Agent focused runtime tests: PASSED, 150 passed, 0 failed.
- WebUI focused default tests: PASSED, 77 passed.
- WebUI agent-runs env focused tests: 69 passed, 8 expected failures.
  - The expected failures are direct/journal runtime route assertions from tests/test_runtime_routes.py under forced agent-runs mode.

### Preserved behavior

- RuntimeExecutor remains the execution owner for execute:true runtime runs.
- RuntimeControlBridge remains preserved.
- DefaultAgentFactory real-provider execution remains unchanged.
- API-server runtime path remains preserved.
- Messaging-platform runtime binding remains preserved.
- Slash-command state sync remains preserved.
- No production-only pending-action injection endpoint was added.

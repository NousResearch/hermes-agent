# Runtime Live Smoke Harness

Repeatable cross-repo live HTTP smoke harness for the Hermes Agent
`RuntimeExecutor` and WebUI `agent-runs` adapter.

## Files

| File | Purpose |
|------|---------|
| `scripts/standalone_runtime_server.py` | Minimal aiohttp server with runtime routes + executor |
| `scripts/smoke_runtime_executor_live.sh` | Agent-only live smoke (direct HTTP) |
| `scripts/smoke_agent_runs_live.sh` (in hermes-webui) | WebUI agent-runs live smoke |
| `scripts/smoke_cross_repo.sh` | Combined Agent + WebUI cross-repo smoke |
| `tests/gateway/test_runtime_live_http_smoke.py` | Pytest tests for smoke harness construction |
| `tests/test_agent_runs_live_http_smoke.py` (in hermes-webui) | Pytest tests for WebUI smoke harness |
| `docs/runtime-live-smoke.md` | This file |

## Agent Environment

```bash
# Start the standalone runtime server (fake/deterministic mode):
cd hermes-agent
python3 scripts/standalone_runtime_server.py --port 8642 --fake

# Or with real DeepSeek credentials:
DEEPSEEK_API_KEY=<key> python3 scripts/standalone_runtime_server.py --port 8642

# Run the Agent-only smoke:
scripts/smoke_runtime_executor_live.sh           # auto-detects DEEPSEEK_API_KEY
scripts/smoke_runtime_executor_live.sh --fake    # deterministic mode
```

## WebUI Environment

```bash
# Start WebUI in agent-runs mode (requires Agent server on :8642):
cd hermes-webui
HERMES_WEBUI_RUNTIME_ADAPTER=agent-runs \
HERMES_WEBUI_AGENT_RUNS_BASE_URL=http://127.0.0.1:8642 \
HERMES_WEBUI_PASSWORD=test-password \
HERMES_WEBUI_PORT=8789 \
python3 server.py

# Run the WebUI smoke (requires Agent server on :8642 or AGENT_BASE_URL):
scripts/smoke_agent_runs_live.sh
```

## Cross-Repo Smoke

```bash
cd hermes-agent
# Combined smoke (starts both servers):
scripts/smoke_cross_repo.sh
# Or with real credentials:
DEEPSEEK_API_KEY=<key> scripts/smoke_cross_repo.sh
```

## What Is Verified

1. **Agent direct POST /v1/runs execute:true** — creates a run, executes it,
   reaches completed status, emits done events.
2. **WebUI proxied run status** — GET /api/runs/{run_id} returns terminal state.
3. **WebUI proxied run events** — GET /api/runs/{run_id}/events contains done.
4. **Cancel/stop** — POST /v1/runs/{run_id}/stop transitions to cancelled.
5. **Runtime capabilities** — GET /api/runtime/capabilities shows agent-runs.
6. **Deployment health** — GET /api/deployment/health shows agent-runs adapter.
7. **Approval/clarify** — control-plane endpoints respond correctly (returns
   action_not_found when no pending action exists).

## Real-Credential Smoke

When `DEEPSEEK_API_KEY` is set, the DefaultAgentFactory resolves live
credentials and executes a real AIAgent call. If the key is missing, smoke
falls back to deterministic mode with `FakeAgentFactory`.

## Approval/Clarify Live Pending-Action Smoke

No deterministic pending-action trigger exists without production-only test
injection endpoints. Approval/clarify are verified by:
- Contract/unit tests (`test_runtime_approval_clarify.py`)
- RunManager-level smoke
- Live smoke verifies endpoints respond but returns action_not_found
  (expected for runs without pending actions)

## CI Usage

```bash
# Deterministic (no credentials needed):
cd hermes-agent && scripts/smoke_cross_repo.sh --fake

# With credentials (if DEEPSEEK_API_KEY is available):
cd hermes-agent && scripts/smoke_cross_repo.sh
```

The harness respects `SKIP_REAL=1` to skip real-credential execution
when credentials are present but should not be used in CI.

## Phase 19 Updates

- `--fake` mode now generates `approval.requested` and `clarify.requested` events
  during execution (wired via `FakeAgentFactory` callbacks).
- Smokes 6 and 7 verify that run events contain approval/clarify events.
- Full e2e lifecycle resolution (approve while run is non-terminal) remains
  deferred — the fake agent completes immediately, putting the run in terminal
  state before resolution can complete. A future improvement can add a
  pause-before-complete mechanism or delay-based polling window.
- Real DeepSeek smoke was skipped in this environment because `DEEPSEEK_API_KEY`
  was not set. Run with `DEEPSEEK_API_KEY=<key>` to exercise the live
  `DefaultAgentFactory` path.
- Messaging-adapter live-smoke requirements are documented in
  `docs/messaging-adapter-live-smoke.md`.

---

## Phase 20 -- Real DeepSeek Smoke Readiness

Date: 2026-07-02

### Result

- Deterministic Agent-only runtime smoke: PASSED.
  - Command: scripts/smoke_runtime_executor_live.sh
  - Result: 7 passed, 0 failed, 1 skipped.
  - Port: 8642.
  - Cleanup: standalone runtime server stopped.
- Deterministic cross-repo Agent to WebUI smoke: PASSED.
  - Command: scripts/smoke_cross_repo.sh
  - Result: 11 passed, 0 failed.
  - Ports: Agent 8642, WebUI 8789.
  - Cleanup: Agent and WebUI smoke servers stopped.
- Real DeepSeek Agent-only smoke: SKIPPED.
  - Reason: DEEPSEEK_API_KEY was not present in the active environment.
- Real DeepSeek cross-repo smoke: SKIPPED.
  - Reason: DEEPSEEK_API_KEY was not present in the active environment.
- Provider/model: N/A because no real credential smoke ran.
- WebUI proxied status/events: PASSED via deterministic cross-repo smoke.
- Cancel/stop: PASSED via deterministic smoke.
- Secret leakage: no secret values were printed or committed during credential checks.

### Future real DeepSeek smoke requirement

Set DEEPSEEK_API_KEY only in a local secret store or active shell.

Preferred real-smoke target:

- Provider: DeepSeek
- Model: deepseek-v4-flash
- Prompt: Return exactly: hermes runtime real smoke ok

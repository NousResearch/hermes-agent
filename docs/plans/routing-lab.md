# Routing Lab Plan

> For Hermes: use this worktree only for experiments, implementation tests, and PR preparation.

Goal: Test routing-related changes safely in an isolated git worktree without modifying the active production config.

Architecture:
- Code changes happen only in this worktree branch.
- The pre-change config backup is stored as a fixture/reference file, not restored over ~/.hermes/config.yaml.
- Every PR that comes out of this lab must include implementation tests or explicit validation coverage for the changed behavior.

Scope:
- Worktree path: ~/.hermes/worktrees/routing-lab
- Branch: chore/routing-lab
- Config fixture: tests/fixtures/config.pre-remove-routing.yaml

Rules:
1. Do not overwrite ~/.hermes/config.yaml from this lab.
2. Use fixture/temp config files for routing tests.
3. Before opening a PR, run the relevant implementation tests.
4. A PR is only considered ready if the targeted behavior is covered by tests and the test run is green.
5. Keep experimental commits in this branch or its child branches until validated.

PR test policy:
- For every accepted change, add or update implementation tests.
- Prefer focused pytest coverage for the exact routing/fallback/delegation behavior touched.
- Record the exact test command used in the PR notes.

Initial checklist:
- [x] Create isolated worktree
- [x] Copy config backup as fixture
- [ ] Identify first routing change to test
- [ ] Add implementation test(s)
- [ ] Run tests
- [ ] Prepare PR

---

## FASE 3: Non-invasive Telemetry Integration

### Activation
```bash
export HERMES_ROUTING_TELEMETRY=1
```
Set this env var before launching hermes from the worktree. Events are written to:
```
~/.hermes/router/telemetry.jsonl
```

### Query telemetry
```bash
python scripts/cost_tracker.py summary
```

### Design
- `agent/smart_model_routing.py` exposes `instrument_resolve_turn_route()` function
- Auto-instrumentation at import time if `HERMES_ROUTING_TELEMETRY=1`
- Idempotent: calling `instrument_resolve_turn_route()` multiple times is a no-op
- `wrap_resolve_turn_route()` decorator in `agent/routing_telemetry.py` captures:
  - success/failure
  - latency_ms
  - model/provider from result dict
  - turn_kind: "smart_route" if label contains "smart route", else "primary"

### Tests
- `tests/routing/test_lab_integration.py` — 5 tests covering env-var trigger, idempotency, event recording

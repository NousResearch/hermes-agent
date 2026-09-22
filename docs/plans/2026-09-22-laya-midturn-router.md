# Laya Mid-Turn Router Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a safe, opt-in same-runtime mid-turn route selection after completed tool outcomes.

**Architecture:** Extend the existing allowlist router with a separate redacted tool-outcome resolver. Add a narrow agent-loop helper that can alter only request-local model/effort/overrides when target and active runtime identities match. Restore request-local state in the public conversation wrapper.

**Tech Stack:** Python 3.13, Hermes `AIAgent` turn phases, pytest through `scripts/run_tests.sh`.

---

### Task 1: Specify the pure resolver contract

**Objective:** Write behavior tests for disabled, redacted tool-outcome routing and strong-route fail-closed semantics.

**Files:**
- Modify: `tests/hermes_cli/test_model_router.py`
- Modify: `hermes_cli/model_router.py`

**Step 1: Write failing tests** for `resolve_midturn_route()` proving it sends only bounded redacted goal/outcome state and pins exceptional decisions to `strong_route`.

**Step 2: Run RED**

Run: `scripts/run_tests.sh tests/hermes_cli/test_model_router.py -k midturn`

Expected: FAIL because the resolver does not exist.

**Step 3: Implement minimal resolver** using the existing loopback controller, confidence checks, route allowlist, and runtime resolver.

**Step 4: Run GREEN**

Run: `scripts/run_tests.sh tests/hermes_cli/test_model_router.py -k midturn`

Expected: PASS.

### Task 2: Specify and implement same-runtime application

**Objective:** Prove the agent-loop helper rejects cross-runtime targets and changes only model, effort, and request overrides for a compatible next request.

**Files:**
- Create: `agent/midturn_model_router.py`
- Create: `tests/agent/test_midturn_model_router.py`

**Step 1: Write failing tests** for compatibility, no controller when disabled, strong pinning, and state restoration.

**Step 2: Run RED**

Run: `scripts/run_tests.sh tests/agent/test_midturn_model_router.py`

Expected: FAIL because the helper does not exist.

**Step 3: Implement minimal request-local helper**; do not call `switch_model()` and do not mutate prompt/tools/history/fallback state.

**Step 4: Run GREEN**

Run: `scripts/run_tests.sh tests/agent/test_midturn_model_router.py`

Expected: PASS.

### Task 3: Wire only the completed-tool boundary

**Objective:** Call the helper after a successful tool round and restore state around every conversation exit.

**Files:**
- Modify: `agent/conversation_loop.py`
- Modify: `agent/turn_tool_round.py`
- Modify: `hermes_cli/config_defaults.py`
- Modify: `tests/agent/test_midturn_model_router.py`

**Step 1: Write a failing loop integration test** with tool calls followed by text, asserting model/effort transition, preserved prompt/tools/history, and restoration after the turn.

**Step 2: Run RED**

Run: `scripts/run_tests.sh tests/agent/test_midturn_model_router.py -k integration`

Expected: FAIL because the loop does not call the helper.

**Step 3: Wire the helper** after `run_tool_round()` returns `continue`; use `try/finally` in the public wrapper to restore request-local fields. Add disabled nested config defaults.

**Step 4: Run GREEN and focused regression tests**

Run: `scripts/run_tests.sh tests/agent/test_midturn_model_router.py tests/hermes_cli/test_model_router.py`

Expected: PASS.

### Task 4: Verify and commit locally

**Objective:** Run the specified focused suite and inspect repository state without activating production configuration.

**Files:** all changed files above.

**Step 1:** Run `scripts/run_tests.sh tests/agent/test_midturn_model_router.py tests/hermes_cli/test_model_router.py`.

**Step 2:** Run `git diff --check` and `git status --short`.

**Step 3:** Commit only verified changes with `git commit -m "feat: add opt-in midturn model router"`; do not push.

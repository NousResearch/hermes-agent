# CTX residue changes

Date: 2026-07-10

## Header

No live-gateway restart was performed. I did not touch
`~/.hermes/hermes-agent` or `~/.hermes/runtime`.

Active-risk assessment: the branch contains the `_run_agent` choke-point fix
for the confirmed recursive/shortcut turn-entry class. The added RC3 tests did
not reproduce pool-thread residue through the wrapped gateway/tool seams.

## Code changes

- Added a log-only tool-submit invariant check in
  `agent/tool_executor.py`.
  It compares the currently bound `HERMES_SESSION_KEY` against the executing
  agent's `_gateway_session_key` immediately before concurrent tool jobs are
  submitted. This is the runtime net requested for the worker/tool seam.
- Added worker-thread residue regressions in
  `tests/gateway/test_session_context_inheritance.py`.
  They poison a size-1 executor thread with a foreign ContextVar binding and
  assert on the same subprocess-env surface the incident exposed.
- Added a focused unit for the new tool-submit warning.

## Diagnosis summary

RC3 pool residue was ruled out for these two wrapped seams:

- `GatewayRunner._run_in_executor_with_context` uses `copy_context()` plus
  `ctx.run`, so a recycled gateway pool thread's own ContextVars are not used.
- `tools.thread_context.propagate_context_to_thread` uses `copy_context()` plus
  `ctx.run`, so a recycled tool worker's own ContextVars are not used.

The confirmed fix remains the `_run_agent` wrapper that binds from the current
call's `source`, `session_key`, `session_id`, and `event_message_id` before any
agent execution or executor snapshot, then restores by token on exit.

## Gates run

- `scripts/run_tests.sh tests/gateway/test_session_context_inheritance.py`
  passed: 14 tests.
- `scripts/run_tests.sh tests/tools/test_local_env_session_leak.py`
  passed: 12 tests.
- `scripts/run_tests.sh tests/agent/test_close_interrupted_tool_sequence.py tests/agent/test_shell_hooks.py tests/tools/test_execute_code_approval_cluster.py tests/tools/test_approved_command_clean_slate.py`
  passed: 92 tests.
- `scripts/run_tests.sh tests/tools/test_interrupt.py tests/agent/test_budget_grace_gate.py`
  passed: 24 tests.
- `scripts/run_tests.sh tests/tools/test_tool_search.py`
  passed: 39 tests.
- `python -m py_compile agent/tool_executor.py tests/gateway/test_session_context_inheritance.py`
  passed.

## Not completed

- I did not implement the requested `execute_code` live-agent namespace because
  this checkout's `execute_code` path executes user code in subprocess/remote
  environments, not an in-process namespace.

# CTX residue diagnosis

Date: 2026-07-10
Worktree: `/Users/alexgierczyk/.hermes/worktrees/ctx-residue`
Branch: `fix/executor-ctx-residue`

## Result

Confirmed root cause in this checkout is the turn-entry / recursive `_run_agent`
binding class, not recycled executor-thread residue.

The current branch already contains the structural `_run_agent` wrapper fix for
that class. I added the missing RC3 worker-thread gates and they pass against
the fixed checkout, explicitly ruling out the two wrapped executor seams as the
active residue mechanism here:

- `tests/gateway/test_session_context_inheritance.py::test_gateway_executor_context_run_overrides_reused_thread_residue_bidirectionally`
- `tests/gateway/test_session_context_inheritance.py::test_tool_thread_context_wrapper_overrides_reused_thread_residue_bidirectionally`

Both tests poison a size-1 worker thread with a foreign binding, then assert on
the subprocess-env surface (`_make_run_env` / `_inject_session_context_env`)
from inside the reused worker. The gateway worker path observes the current
turn's binding because `GatewayRunner._run_in_executor_with_context` schedules
`ctx.run`. The tool-worker path observes the agent thread's binding because
`propagate_context_to_thread` snapshots the parent context and runs the target
under `ctx.run`.

## Re-grounded anchors

- Handler entry reset: `gateway/run.py:10672-10685`.
  This clears inherited task ContextVars before the pre-bind window.
- Top-level handler bind: `gateway/run.py:12926-12929`.
  Normal message handling builds `SessionContext` and calls `_set_session_env`.
- Shared bind helper: `gateway/run.py:17223-17272`.
  `_set_session_vars_for_source` binds source/chat/thread/user/session/message
  identity and async-delivery capability.
- Gateway executor wrapper: `gateway/run.py:17279-17288`.
  It captures `copy_context()` in the async task and schedules `ctx.run`.
- `_run_agent` choke point: `gateway/run.py:19282-19318`.
  It resets then binds from this call's own `source`, `session_key`,
  `session_id`, and `event_message_id` before `_run_agent_inner`.
- Existing entry warning: `gateway/run.py:19389-19401`.
  It logs when the bound session key disagrees with the currently executing
  turn's source-derived key.
- Recursive queued follow-up: `gateway/run.py:22392-22402`.
  It re-enters `_run_agent`; the choke point above is what makes nested
  follow-ups restore/bind correctly.
- Tool context wrapper: `tools/thread_context.py:64-105`.
  It captures `contextvars.copy_context()` on the agent/tool-submit thread and
  runs the tool target under that context.
- Tool submit warning added here: `agent/tool_executor.py:417-431`,
  called at `agent/tool_executor.py:804-805`.

## Mechanism

The live symptom was a coherent foreign identity in terminal subprocess env,
while the task-entry mismatch warning did not fire. The worker-thread surface
had to be tested directly because MainThread ContextVar reads are not a valid
gate for this failure class.

The deterministic residue tests show that a stale binding directly written onto
a reused worker thread does not leak through either wrapped seam:

1. Gateway worker seam: `_run_in_executor_with_context` snapshots the current
   async task after the turn bind, then calls `ctx.run` on the pool thread.
   A poisoned pool thread is overridden for the duration of the job.
2. Tool worker seam: `propagate_context_to_thread` snapshots the agent worker
   thread at submit time and executes `_run_tool` under `ctx.run`. A poisoned
   daemon worker is overridden for the duration of the tool job.

The confirmed bad path is therefore the caller-to-`_run_agent` path where
recursive or shortcut turns can bypass the top-level message handler's bind.
The fixed choke point is `_run_agent` itself. It is before `_run_agent_inner`
and before any `_run_in_executor_with_context` snapshot, satisfying the
bind-precedes-snapshot invariant. It uses token-based reset/restore semantics:
inner recursive exits restore the outer turn's binding rather than clearing it.

## Failing-first tests for the confirmed path

Existing tests in this branch cover the confirmed mechanism and would fail if
the `_run_agent` rebinding wrapper were removed:

- `test_run_agent_rebinds_full_turn_context_before_inner_dispatch`
- `test_recursive_run_agent_rebinds_queued_cross_session_followup`
- `test_recursive_run_agent_rebinds_steer_fallback_message_anchor`
- `test_run_agent_composes_session_binding_with_profile_scope`

New RC3 tests cover the worker-thread residue suspicion and runtime net:

- `test_gateway_executor_context_run_overrides_reused_thread_residue_bidirectionally`
- `test_tool_thread_context_wrapper_overrides_reused_thread_residue_bidirectionally`
- `test_tool_submit_warns_on_agent_context_mismatch`

## Execute-code probe note

The brief also requested exposing the live `agent` object to executed code when
`execute_code` runs in-process for the top-level agent. In this checkout,
`tools/code_execution_tool.py` runs user code through local subprocess sandbox
or remote environment dispatch; I found no in-process user-code namespace to
bind without changing the execution model. I did not add a fake in-process path.

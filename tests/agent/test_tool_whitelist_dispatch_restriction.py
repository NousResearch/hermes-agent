"""A restricted fork must be denied by the real tool-executor worker.

Regression for #151307.  The whitelist is installed by the calling fork, but
``_run_sequential_tool_execution_middleware`` invokes the policy gate on a
worker.  A thread-local whitelist vanishes there, allowing the tool body.
"""

import threading

from agent import tool_executor
from hermes_cli.plugins import clear_thread_tool_whitelist, set_thread_tool_whitelist


class _Allow:
    allows_execution = True


class _Guardrails:
    def before_call(self, _name, _args):
        return _Allow()


class _Agent:
    session_id = "whitelist-worker-regression"
    tool_progress_callback = None
    _interrupt_requested = False

    def __init__(self):
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._tool_guardrails = _Guardrails()

    def _touch_activity(self, _message):
        pass


def test_restricted_fork_cannot_run_terminal_on_tool_worker(monkeypatch):
    """The same real executor can run an allowed call but must not run a denied one."""
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 10.0)
    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call", lambda *_a, **_k: None)
    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda *_a, **_k: None)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda *_a, **_k: [])

    reached = threading.Event()

    def execute(_args):
        reached.set()
        return "terminal body executed"

    def dispatch():
        return tool_executor._run_sequential_tool_execution_middleware(
            _Agent(),
            function_name="terminal",
            function_args={},
            effective_task_id="t",
            tool_call_id="whitelist-worker-regression",
            execute=execute,
        )

    # Positive control: the fixture reaches a real tool body when unrestricted.
    unrestricted = dispatch()
    assert unrestricted.result == "terminal body executed"
    assert reached.is_set()

    reached.clear()
    set_thread_tool_whitelist({"memory"}, deny_msg_fmt="review denied {tool_name}")
    try:
        restricted = dispatch()
    finally:
        clear_thread_tool_whitelist()

    assert not reached.is_set(), "a forbidden terminal ran on the tool worker"
    assert restricted.blocked is True
    assert "review denied terminal" in str(restricted.result)

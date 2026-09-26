"""The thread tool whitelist must survive the tool-executor worker hop.

``_run_sequential_tool_execution_middleware`` runs every tool on a daemon worker thread wrapped
in ``propagate_context_to_thread`` (ContextVars only). While the whitelist lived in
``threading.local`` the worker saw no whitelist at all, so ``/btw`` and the background-review
fork could reach ANY tool body (#98479). These tests cross the real executor seam.
"""

import threading

import agent.tool_executor as tool_executor
from agent.tool_executor import _run_sequential_tool_execution_middleware
from hermes_cli.plugins import clear_thread_tool_whitelist, set_thread_tool_whitelist


class _AllowAll:
    allows_execution = True


class _Guardrails:
    def before_call(self, name, args):
        return _AllowAll()


class _FakeAgent:
    def __init__(self):
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._interrupt_requested = False
        self._tool_guardrails = _Guardrails()
        self.session_id = "s"

    def _touch_activity(self, msg):
        pass


def _dispatch(name, execute, monkeypatch):
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 10.0)
    monkeypatch.setattr(tool_executor, "_emit_terminal_post_tool_call", lambda agent, **kw: None)
    monkeypatch.setattr(tool_executor, "_begin_tool_execution", lambda agent, ref, idx: None)
    return _run_sequential_tool_execution_middleware(
        _FakeAgent(), function_name=name, function_args={"path": "x"},
        effective_task_id="t", tool_call_id="call_1", execute=execute)


def test_whitelist_denies_tool_body_across_the_worker_hop(monkeypatch):
    ran = threading.Event()
    seen_thread = {}

    def execute(args):
        seen_thread["name"] = threading.current_thread().name
        ran.set()
        return "BODY RAN"

    set_thread_tool_whitelist({"read_file"}, deny_msg_fmt="denied {tool_name}")
    try:
        managed = _dispatch("terminal", execute, monkeypatch)
    finally:
        clear_thread_tool_whitelist()

    assert not ran.is_set(), f"tool body executed on {seen_thread.get('name')}"
    assert managed.blocked is True
    assert "denied terminal" in str(managed.result)


def test_whitelisted_tool_body_still_runs_on_the_worker(monkeypatch):
    seen_thread = {}

    def execute(args):
        seen_thread["name"] = threading.current_thread().name
        return "ok"

    set_thread_tool_whitelist({"read_file"})
    try:
        managed = _dispatch("read_file", execute, monkeypatch)
    finally:
        clear_thread_tool_whitelist()

    assert managed.result == "ok"
    # The point of the test: the body ran on the executor's worker, not inline on this thread.
    assert seen_thread["name"] != threading.current_thread().name

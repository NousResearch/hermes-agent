"""P-0106: a delegate_task sequential-deadline timeout must not kill a healthy batch.

Under agent.deadline's 420 s sequential-call deadline, a nested orchestrator's
delegate_task call "times out" while its children keep running on daemon
threads. Legacy behavior cancels the worker and interrupts its thread — the
children lose their completion path and the orchestrator re-dispatches,
multiplying tasks and spend. The guard (tools.delegate_tool_timeout) polls the
batch's live transcripts; fresh activity defers the timeout.

Invariant tests: observable outcomes (result type, no interrupt fan-out,
registry hygiene), never internals like call counts.

Note: on current origin/main ``delegate_task`` IS exempt from the sequential
deadline (``_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS``) - the primary fix. The
timeout-path tests here reinstate a deadline (monkeypatching the exemption
away) to pin the deferred/fatal mechanics for the regression class "the
exemption is lost again"; the pin test locks the exemption itself in place.
"""

import threading
import time
from pathlib import Path

import pytest

import agent.tool_executor as tool_executor
import tools.delegate_tool_timeout as guard
from agent.tool_executor import (
    _ManagedToolResult,
    _ToolTimeoutResult,
    _run_sequential_tool_execution_middleware,
)

_LIVE_DIR = Path(__file__).parent / "_p0106_live_dir"


def _reset_live_dir():
    _LIVE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in _LIVE_DIR.iterdir():
        stale.unlink()


class _FakeAgent:
    def __init__(self):
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._interrupt_requested = False
        self.activity = []

    def _touch_activity(self, msg):
        self.activity.append(msg)


@pytest.fixture()
def fake_agent():
    return _FakeAgent()


@pytest.fixture(autouse=True)
def _fast_polls(monkeypatch):
    monkeypatch.setattr(tool_executor, "_SEQUENTIAL_INTERRUPT_POLL_SECONDS", 0.05)
    emitted = []
    monkeypatch.setattr(
        tool_executor,
        "_emit_terminal_post_tool_call",
        lambda agent, **kw: emitted.append(kw),
    )
    yield emitted


@pytest.fixture(autouse=True)
def clean_registry_and_dir():
    guard._ACTIVE_DELEGATIONS.clear()
    _reset_live_dir()
    yield
    guard._ACTIVE_DELEGATIONS.clear()


@pytest.fixture()
def _delegate_deadline_reinstated(monkeypatch):
    """Simulate the P-0106 regression class: delegate_task no longer exempt."""
    monkeypatch.setattr(tool_executor, "_SEQUENTIAL_DEADLINE_EXEMPT_TOOLS", frozenset())


def _late_middleware(release, register_id=None, touch_log=False):
    """Fake inner middleware: blocks until released; optionally registers the batch."""
    def _run(agent_arg, **kwargs):
        if register_id is not None:
            guard.register_delegation(register_id, str(_LIVE_DIR))
        if touch_log:
            _LIVE_DIR.joinpath("task-1.log").write_text("child working\n", encoding="utf-8")
        release.wait(30)
        return _ManagedToolResult(result="late", args={}, middleware_trace=[], blocked=False, dispatched=True)
    return _run


def _release_after(seconds, release):
    def _target():
        time.sleep(seconds)
        release.set()
    threading.Thread(target=_target, daemon=True).start()


def test_fresh_child_activity_defers_timeout(monkeypatch, fake_agent, _fast_polls, _delegate_deadline_reinstated):
    """Batch alive (recent log write) -> deferred: notice tells the model the batch
    was NOT re-dispatched and to poll the transcripts instead."""
    release = threading.Event()

    def _fake_middleware(agent_arg, **kwargs):
        guard.register_delegation("deleg-fresh", str(_LIVE_DIR))
        _LIVE_DIR.joinpath("task-1.log").write_text("child working\n", encoding="utf-8")
        release.wait(30)
        return _ManagedToolResult(result="late", args={}, middleware_trace=[], blocked=False, dispatched=True)

    monkeypatch.setattr(tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware)
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 0.2)

    _release_after(0.6, release)

    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="delegate_task",
        function_args={"tasks": [{"goal": "x"}]},
        effective_task_id="t",
        tool_call_id="call_fresh",
        execute=lambda a: "unused",
    )

    assert isinstance(managed.result, _ToolTimeoutResult)
    notice = str(managed.result)
    assert "NOT re-dispatched" in notice
    assert "action='list'" in notice
    assert any(kw.get("error_type") == "delegate_still_active" for kw in _fast_polls)


def test_stale_batch_times_out_fatal(monkeypatch, fake_agent, _fast_polls, _delegate_deadline_reinstated):
    """No recent activity anywhere in the batch -> legacy fatal timeout behavior."""
    release = threading.Event()

    def _fake_middleware(agent_arg, **kwargs):
        guard.register_delegation("deleg-stale", str(_LIVE_DIR))
        release.wait(30)
        return _ManagedToolResult(result="late", args={}, middleware_trace=[], blocked=False, dispatched=True)

    monkeypatch.setattr(tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware)
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 0.2)

    _release_after(0.6, release)

    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="delegate_task",
        function_args={"tasks": [{"goal": "x"}]},
        effective_task_id="t",
        tool_call_id="call_stale",
        execute=lambda a: "unused",
    )

    assert isinstance(managed.result, _ToolTimeoutResult)
    assert "timed out after" in str(managed.result)
    assert any(kw.get("error_type") == "tool_timeout" for kw in _fast_polls)


def test_unregistered_batch_is_fatal(monkeypatch, fake_agent, _fast_polls, _delegate_deadline_reinstated):
    """Guard has no entry for the worker -> fatal timeout, never assumed-alive."""
    release = threading.Event()

    def _fake_middleware(agent_arg, **kwargs):
        release.wait(30)
        return _ManagedToolResult(result="late", args={}, middleware_trace=[], blocked=False, dispatched=True)

    monkeypatch.setattr(tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware)
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 0.2)

    _release_after(0.6, release)

    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="delegate_task",
        function_args={"tasks": [{"goal": "x"}]},
        effective_task_id="t",
        tool_call_id="call_unreg",
        execute=lambda a: "unused",
    )

    assert isinstance(managed.result, _ToolTimeoutResult)
    assert "timed out after" in str(managed.result)


def test_guard_failure_falls_back_to_fatal(monkeypatch, fake_agent, _fast_polls, _delegate_deadline_reinstated):
    """A failing guard module must not mask the timeout: the executor wrapper
    swallows the import/call failure and legacy fatal behavior survives."""
    monkeypatch.setitem(
        __import__("sys").modules, "tools.delegate_tool_timeout", None  # import -> ImportError
    )
    release = threading.Event()

    def _fake_middleware(agent_arg, **kwargs):
        release.wait(30)
        return _ManagedToolResult(result="late", args={}, middleware_trace=[], blocked=False, dispatched=True)

    monkeypatch.setattr(tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware)
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 0.2)

    _release_after(0.6, release)

    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="delegate_task",
        function_args={"tasks": [{"goal": "x"}]},
        effective_task_id="t",
        tool_call_id="call_boom",
        execute=lambda a: "unused",
    )

    assert isinstance(managed.result, _ToolTimeoutResult)
    assert "timed out after" in str(managed.result)


def test_non_delegate_tools_keep_legacy_timeout(monkeypatch, fake_agent, _fast_polls):
    """The guard must not change timeout behavior for ordinary tools."""
    release = threading.Event()

    def _fake_middleware(agent_arg, **kwargs):
        release.wait(30)
        return _ManagedToolResult(result="late", args={}, middleware_trace=[], blocked=False, dispatched=True)

    monkeypatch.setattr(tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware)
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: 0.2)

    _release_after(0.6, release)

    managed = _run_sequential_tool_execution_middleware(
        fake_agent,
        function_name="web_search",
        function_args={},
        effective_task_id="t",
        tool_call_id="call_web",
        execute=lambda a: "unused",
    )

    assert isinstance(managed.result, _ToolTimeoutResult)
    assert "timed out after" in str(managed.result)
    assert any(kw.get("error_type") == "tool_timeout" for kw in _fast_polls)


def test_registration_round_trip():
    """register/unregister around a sync batch: entry present during, gone after."""
    tid = threading.get_ident()
    guard.register_delegation("deleg-rt", str(_LIVE_DIR))
    assert guard.active_delegation_id(tid) == "deleg-rt"
    assert guard.maybe_defer_sequential_timeout(tid) is False
    _LIVE_DIR.joinpath("task-1.log").write_text("work\n", encoding="utf-8")
    assert guard.maybe_defer_sequential_timeout(tid) is True
    guard.unregister_delegation(tid)
    assert guard.active_delegation_id(tid) is None


def test_deferred_batch_hygiene_after_executor_cleanup():
    """A deferred timeout's registry entry must not outlive the batch."""
    tid = threading.get_ident()
    guard.register_delegation("deleg-hyg", str(_LIVE_DIR))
    assert guard.active_delegation_id(tid) == "deleg-hyg"
    tool_executor._delegate_timeout_guard("unregister_delegation", tid)
    assert guard.active_delegation_id(tid) is None


def test_delegate_task_exempt_from_sequential_deadline():
    """P-0106 primary fix pin: delegate_task carries no sequential deadline on main.

    The 420s deadline firing on delegate_task is the P-0106 root cause; this
    catches any reversion (it was re-applied upstream once already).
    """
    assert "delegate_task" in tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS


def test_non_delegate_tool_is_not_exempt():
    """Only delegate_task is exempt - ordinary tools keep their deadline."""
    assert "web_search" not in tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS
    assert "terminal" not in tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS

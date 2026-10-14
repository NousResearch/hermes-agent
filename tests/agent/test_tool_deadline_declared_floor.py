"""A tool's own timeout must outlive the executor's generic tool deadline.

``_DEFAULT_CONCURRENT_TOOL_TIMEOUT_S`` is 420s, but several tools are handed a longer
budget by their own schema and config: ``terminal`` advertises a 600s foreground cap
(``FOREGROUND_MAX_TIMEOUT``) and honours ``TERMINAL_TIMEOUT``, ``process_manage
action=wait`` blocks for its ``timeout`` clamped by the same setting. Before
``ToolEntry.deadline_floor`` the generic guard fired first and returned
``Error executing tool '<name>': timed out after 420.0s`` with no output at all, where
the tool's own timeout would have returned the partial output it had collected.
"""

import threading
import time

import pytest

from agent import tool_executor
from agent.tool_executor import (
    _ManagedToolResult,
    _ToolTimeoutResult,
    _run_sequential_tool_execution_middleware,
)
from tools.registry import registry


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
def _quiet_post_tool_call(monkeypatch):
    monkeypatch.setattr(tool_executor, "_SEQUENTIAL_INTERRUPT_POLL_SECONDS", 0.05)
    monkeypatch.setattr(
        tool_executor, "_emit_terminal_post_tool_call", lambda agent, **kw: None
    )


@pytest.fixture()
def slow_tool():
    """Register a tool that blocks ~1s, with a settable declared floor."""
    declared = {"floor": None}
    name = "spec_slow_tool"
    registry.register(
        name=name, toolset="testing", schema={"name": name, "parameters": {}},
        handler=lambda **kw: "unused", emoji="🧪",
        deadline_floor=lambda args: declared["floor"],
    )
    try:
        yield name, declared
    finally:
        registry.deregister(name)


def _run(agent, name, args, monkeypatch, *, generic, work_seconds):
    monkeypatch.setattr(tool_executor, "_resolve_sequential_tool_timeout", lambda: generic)

    def _fake_middleware(agent_arg, **kwargs):
        # health: allow HX013 -- a call that outlives the generic deadline IS the behaviour under test
        time.sleep(work_seconds)
        return _ManagedToolResult(
            result="real result", args={}, middleware_trace=[], blocked=False, dispatched=True,
        )

    monkeypatch.setattr(tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware)
    return _run_sequential_tool_execution_middleware(
        agent, function_name=name, function_args=args, effective_task_id="t",
        tool_call_id="c1", execute=None,
    )


class TestSequentialRunnerHonoursTheFloor:
    def test_undeclared_tool_still_dies_on_the_generic_deadline(self, fake_agent, monkeypatch, slow_tool):
        """Control: containment for a tool that declares nothing is unchanged."""
        name, declared = slow_tool
        declared["floor"] = None
        result = _run(fake_agent, name, {}, monkeypatch, generic=0.3, work_seconds=5)
        assert isinstance(result.result, _ToolTimeoutResult)
        assert "timed out after 0.3s" in str(result.result)

    def test_declared_floor_outlives_a_shorter_generic_deadline(self, fake_agent, monkeypatch, slow_tool):
        """The regression: the same call now survives to return its real result."""
        name, declared = slow_tool
        declared["floor"] = 2.0
        result = _run(fake_agent, name, {}, monkeypatch, generic=0.3, work_seconds=1.0)
        assert not isinstance(result.result, _ToolTimeoutResult)
        assert result.result == "real result"

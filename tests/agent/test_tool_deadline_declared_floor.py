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

import agent.tool_executor as tool_executor
from agent.tool_executor import (
    _ManagedToolResult,
    _ToolTimeoutResult,
    _deadline_with_floor,
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
        time.sleep(work_seconds)
        return _ManagedToolResult(
            result="real result", args={}, middleware_trace=[], blocked=False, dispatched=True,
        )

    monkeypatch.setattr(tool_executor, "_run_agent_tool_execution_middleware", _fake_middleware)
    return _run_sequential_tool_execution_middleware(
        agent, function_name=name, function_args=args, effective_task_id="t",
        tool_call_id="c1", execute=None,
    )


class TestDeadlineWithFloor:
    def test_floor_raises_a_shorter_generic_deadline(self):
        assert _deadline_with_floor(420.0, 605.0) == 605.0

    def test_floor_never_shortens_a_longer_generic_deadline(self):
        assert _deadline_with_floor(900.0, 605.0) == 900.0

    def test_no_declaration_leaves_the_generic_deadline_alone(self):
        assert _deadline_with_floor(420.0, None) == 420.0

    def test_a_floor_never_introduces_a_deadline_where_none_was_set(self):
        assert _deadline_with_floor(None, 605.0) is None


class TestRegistryDeclaration:
    def test_undeclared_tool_has_no_floor(self):
        assert registry.get_deadline_floor("definitely_not_a_tool", {}) is None

    def test_declared_value_is_returned(self, slow_tool):
        name, declared = slow_tool
        declared["floor"] = 42.0
        assert registry.get_deadline_floor(name, {}) == 42.0

    @pytest.mark.parametrize("bad", [0, -1, "nope", float("nan"), float("inf")])
    def test_unusable_declarations_degrade_to_no_floor(self, slow_tool, bad):
        name, declared = slow_tool
        declared["floor"] = bad
        assert registry.get_deadline_floor(name, {}) is None

    def test_a_raising_declaration_does_not_break_dispatch(self):
        name = "spec_exploding_floor"
        registry.register(
            name=name, toolset="testing", schema={"name": name, "parameters": {}},
            handler=lambda **kw: "unused",
            deadline_floor=lambda args: 1 / 0,
        )
        try:
            assert registry.get_deadline_floor(name, {}) is None
        finally:
            registry.deregister(name)


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


class TestTerminalDeclaration:
    def test_foreground_default_declares_the_configured_terminal_timeout(self, monkeypatch):
        from tools.terminal_tool import _terminal_deadline_floor

        monkeypatch.setenv("TERMINAL_TIMEOUT", "1800")
        assert _terminal_deadline_floor({"command": "x"}) == 1800.0

    def test_explicit_foreground_timeout_is_declared(self, monkeypatch):
        from tools.terminal_tool import _terminal_deadline_floor

        monkeypatch.setenv("TERMINAL_TIMEOUT", "180")
        assert _terminal_deadline_floor({"command": "x", "timeout": 600}) == 600.0

    def test_over_cap_foreground_declares_nothing_because_it_is_promoted(self):
        from tools.terminal_tool import FOREGROUND_MAX_TIMEOUT, _terminal_deadline_floor

        assert _terminal_deadline_floor({"command": "x", "timeout": FOREGROUND_MAX_TIMEOUT + 1}) is None

    def test_background_spawn_declares_nothing_because_it_returns_at_once(self):
        from tools.terminal_tool import _terminal_deadline_floor

        assert _terminal_deadline_floor({"command": "x", "background": True, "timeout": 3600}) is None

    def test_a_foreground_call_above_the_generic_deadline_is_no_longer_clipped(self, monkeypatch):
        """The reported shape: terminal(timeout=600) under the 420s default."""
        from tools.terminal_tool import _terminal_deadline_floor

        monkeypatch.setenv("TERMINAL_TIMEOUT", "180")
        floor = tool_executor._declared_deadline_floor("terminal", {"command": "x", "timeout": 600})
        assert floor is not None and floor > 600.0
        assert _deadline_with_floor(420.0, floor) > 600.0
        assert _terminal_deadline_floor({"command": "x", "timeout": 600}) == 600.0


class TestProcessManageDeclaration:
    def test_wait_without_timeout_declares_the_configured_default(self, monkeypatch):
        from tools.process_registry import _process_deadline_floor

        monkeypatch.setenv("TERMINAL_TIMEOUT", "1800")
        assert _process_deadline_floor({"action": "wait", "session_id": "p"}) == 1800.0

    def test_wait_timeout_is_clamped_the_same_way_wait_clamps_it(self, monkeypatch):
        from tools.process_registry import _process_deadline_floor

        monkeypatch.setenv("TERMINAL_TIMEOUT", "600")
        assert _process_deadline_floor({"action": "wait", "timeout": 900}) == 600.0
        assert _process_deadline_floor({"action": "wait", "timeout": 450}) == 450.0

    @pytest.mark.parametrize("action", ["poll", "log", "kill", "list", "write"])
    def test_non_blocking_actions_declare_nothing(self, action):
        from tools.process_registry import _process_deadline_floor

        assert _process_deadline_floor({"action": action, "session_id": "p"}) is None

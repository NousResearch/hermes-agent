import threading
from unittest.mock import MagicMock

import gateway.run as gateway_run
from tools import delegate_tool


def _clear_active_subagents():
    with delegate_tool._active_subagents_lock:
        delegate_tool._active_subagents.clear()


def test_interrupt_all_subagents_interrupts_registered_children():
    _clear_active_subagents()
    child_a = MagicMock()
    child_b = MagicMock()

    try:
        with delegate_tool._active_subagents_lock:
            delegate_tool._active_subagents["sa-a"] = {"agent": child_a}
            delegate_tool._active_subagents["sa-b"] = {"agent": child_b}

        count = delegate_tool.interrupt_all_subagents("Gateway restarting")

        assert count == 2
        child_a.interrupt.assert_called_once_with("Gateway restarting")
        child_b.interrupt.assert_called_once_with("Gateway restarting")
    finally:
        _clear_active_subagents()


def test_run_single_child_returns_when_parent_is_interrupted(monkeypatch):
    _clear_active_subagents()
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 5.0)

    released = threading.Event()

    class Parent:
        _interrupt_requested = True
        _current_task_id = None

    class Child:
        tool_progress_callback = None
        _delegate_saved_tool_names = []
        _credential_pool = None
        _subagent_id = "sa-parent-interrupt"
        _delegate_depth = 1
        _parent_subagent_id = None
        _delegate_role = "leaf"
        model = "test-model"
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_estimated_cost_usd = 0.0
        session_reasoning_tokens = 0

        def __init__(self):
            self.interrupt_message = None
            self.closed = False

        def run_conversation(self, user_message, task_id):
            released.wait(timeout=5)
            return {
                "final_response": "",
                "completed": False,
                "interrupted": True,
                "api_calls": 0,
                "messages": [],
            }

        def interrupt(self, message=None):
            self.interrupt_message = message
            released.set()

        def close(self):
            self.closed = True

        def get_activity_summary(self):
            return {"api_call_count": 0, "max_iterations": 1, "current_tool": None}

    parent = Parent()
    child = Child()

    try:
        result = delegate_tool._run_single_child(
            0,
            "do work",
            child=child,
            parent_agent=parent,
        )
    finally:
        released.set()
        _clear_active_subagents()

    assert result["status"] == "interrupted"
    assert result["exit_reason"] == "interrupted"
    assert child.interrupt_message == "Parent agent interrupted"
    assert child.closed is True


def test_gateway_interrupt_running_agents_also_interrupts_registered_subagents(monkeypatch):
    calls = []

    def _fake_interrupt_all(reason):
        calls.append(reason)
        return 1

    monkeypatch.setattr(delegate_tool, "interrupt_all_subagents", _fake_interrupt_all)

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._running_agents = {}

    gateway_run.GatewayRunner._interrupt_running_agents(runner, "Gateway restarting")

    assert calls == ["Gateway restarting"]

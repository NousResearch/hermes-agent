"""Regression tests for Kanban worker completion cleanup."""

from __future__ import annotations

import json
from types import SimpleNamespace

import run_agent
from run_agent import AIAgent


class _AllowAllGuardrails:
    def before_call(self, _name, _args):
        return SimpleNamespace(allows_execution=True)


class _NoSubdirHints:
    def check_tool_call(self, _name, _args):
        return ""


def _tool_call(name: str, args: dict | None = None, call_id: str | None = None):
    return SimpleNamespace(
        id=call_id or f"call_{name}",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(args or {}),
        ),
    )


def _minimal_agent():
    agent = object.__new__(AIAgent)
    agent._interrupt_requested = False
    agent._tool_guardrails = _AllowAllGuardrails()
    agent._turns_since_memory = 0
    agent._iters_since_skill = 0
    agent.quiet_mode = True
    agent.verbose_logging = False
    agent.log_prefix = ""
    agent.log_prefix_chars = 120
    agent.tool_progress_callback = None
    agent.tool_start_callback = None
    agent.tool_complete_callback = None
    agent.tool_delay = 0
    agent.session_id = "test-session"
    agent.valid_tool_names = None
    agent._checkpoint_mgr = SimpleNamespace(enabled=False)
    agent._memory_manager = None
    agent._context_engine_tool_names = set()
    agent.context_compressor = None
    agent._subdirectory_hints = _NoSubdirHints()
    agent._current_tool = None
    agent._append_guardrail_observation = lambda _n, _a, result, failed=False: result
    agent._apply_pending_steer_to_tool_results = lambda _messages, _count: None
    agent._touch_activity = lambda _note: None
    agent._should_emit_quiet_tool_messages = lambda: False
    agent._should_start_quiet_spinner = lambda: False
    agent._vprint = lambda *args, **kwargs: None
    agent._print_fn = lambda *args, **kwargs: None
    return agent


def test_kanban_complete_success_detection_is_exact():
    assert AIAgent._kanban_complete_tool_succeeded(
        "kanban_complete", '{"ok": true, "task_id": "t_1"}'
    )
    assert not AIAgent._kanban_complete_tool_succeeded(
        "kanban_complete", '{"ok": false, "error": "blocked"}'
    )
    assert not AIAgent._kanban_complete_tool_succeeded("terminal", '{"ok": true}')
    assert not AIAgent._kanban_complete_tool_succeeded("kanban_complete", "not-json")


def test_kanban_worker_stops_after_successful_complete(monkeypatch):
    """A dispatched worker must not execute tool calls after completion."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_done")
    invoked: list[str] = []

    def fake_handle_function_call(name, args, *unused_args, **unused_kwargs):
        invoked.append(name)
        if name == "kanban_complete":
            return json.dumps({"ok": True, "task_id": "t_done"})
        raise AssertionError(f"post-completion tool executed: {name}")

    monkeypatch.setattr(run_agent, "handle_function_call", fake_handle_function_call)

    agent = _minimal_agent()
    messages: list[dict] = []
    assistant_message = SimpleNamespace(
        tool_calls=[
            _tool_call("kanban_complete", {"summary": "done"}, "call_complete"),
            _tool_call("terminal", {"command": "touch should_not_exist"}, "call_terminal"),
        ]
    )

    agent._execute_tool_calls_sequential(assistant_message, messages, "task-id")

    assert invoked == ["kanban_complete"]
    assert [m["name"] for m in messages] == ["kanban_complete", "terminal"]
    assert json.loads(messages[0]["content"])["ok"] is True
    assert "skipped" in messages[1]["content"].lower()
    assert "kanban_complete succeeded" in messages[1]["content"]

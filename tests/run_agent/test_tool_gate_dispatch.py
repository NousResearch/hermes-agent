"""Integration: the tool-approval gate fires at the real dispatch choke point.

Verifies that ``execute_tool_calls_sequential`` consults ``check_tool_approval``
before executing, and that a deferred ("staged") decision short-circuits the
tool — the underlying ``handle_function_call`` must never run, and the agent
sees a non-error staged result.
"""

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names):
    return [
        {"type": "function",
         "function": {"name": n, "description": f"{n} tool",
                      "parameters": {"type": "object", "properties": {}}}}
        for n in names
    ]


def _tool_call(name, arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _make_agent(*tool_names):
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=5,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def test_staged_decision_blocks_execution_sequential():
    agent = _make_agent("echo_tool")
    assistant_message = SimpleNamespace(
        content="", tool_calls=[_tool_call("echo_tool", '{"msg": "hi"}')])
    messages: list = []

    staged = {"approved": False, "status": "staged",
              "message": "Queued for approval; continuing.",
              "pending_id": "pX", "card_id": "tY"}

    hfc = MagicMock(return_value='{"ok": true}')
    with (
        patch("tools.approval.check_tool_approval", return_value=staged) as gate,
        patch("run_agent.handle_function_call", hfc),
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task1")

    gate.assert_called_once()
    # The gated tool must NOT have executed.
    hfc.assert_not_called()
    # The agent sees exactly one tool result carrying the staged decision.
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["status"] == "staged"
    assert "approval" in payload["message"].lower()


def test_blocked_decision_is_error_sequential():
    agent = _make_agent("echo_tool")
    assistant_message = SimpleNamespace(
        content="", tool_calls=[_tool_call("echo_tool", "{}")])
    messages: list = []

    blocked = {"approved": False, "status": "blocked",
               "message": "BLOCKED: user denied. Do NOT retry."}

    hfc = MagicMock(return_value='{"ok": true}')
    with (
        patch("tools.approval.check_tool_approval", return_value=blocked),
        patch("run_agent.handle_function_call", hfc),
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task1")

    hfc.assert_not_called()
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["status"] == "blocked"
    assert "error" in payload  # blocked → error key (hard failure)


def test_staged_decision_blocks_execution_concurrent():
    agent = _make_agent("echo_tool")
    assistant_message = SimpleNamespace(
        content="",
        tool_calls=[_tool_call("echo_tool", '{"i": 1}', "c1"),
                    _tool_call("echo_tool", '{"i": 2}', "c2")])
    messages: list = []

    staged = {"approved": False, "status": "staged",
              "message": "Queued for approval; continuing.",
              "pending_id": "pX", "card_id": "tY"}

    hfc = MagicMock(return_value='{"ok": true}')
    with (
        patch("tools.approval.check_tool_approval", return_value=staged),
        patch("run_agent.handle_function_call", hfc),
    ):
        agent._execute_tool_calls_concurrent(assistant_message, messages, "task1")

    # Gate ran in invoke_tool for both calls → neither executed.
    hfc.assert_not_called()
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    for tm in tool_msgs:
        assert json.loads(tm["content"])["status"] == "staged"


def test_allowed_decision_executes_sequential():
    agent = _make_agent("echo_tool")
    assistant_message = SimpleNamespace(
        content="", tool_calls=[_tool_call("echo_tool", "{}")])
    messages: list = []

    hfc = MagicMock(return_value='{"ok": true}')
    with (
        patch("tools.approval.check_tool_approval",
              return_value={"approved": True, "message": None}),
        patch("run_agent.handle_function_call", hfc),
    ):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task1")

    hfc.assert_called_once()

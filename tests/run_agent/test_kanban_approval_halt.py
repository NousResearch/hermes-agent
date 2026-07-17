"""Conversation-loop behavior when a Kanban card is parked for approval."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.execution_context import (
    ExecutionRole,
    bind_agent_execution_context,
    issue_kanban_approval_pause_token,
    kanban_approval_pending_metadata as _real_pending_metadata,
    reset_agent_execution_context,
)
from run_agent import AIAgent


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": name,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _agent(hermes_home, *names: str) -> AIAgent:
    hermes_home.mkdir(parents=True, exist_ok=True)
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs(*names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.agent_init.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=10,
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


def _tool_call(name: str, call_id: str, arguments: str = "{}") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _response(*, content="", finish_reason="stop", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        model="test/model",
        usage=None,
    )


def _pending_result() -> str:
    return json.dumps(
        {
            "status": "kanban_approval_pending",
            "kanban_approval_pending": True,
            "request_id": "apr_123",
            "display_target": "redacted target",
            "description": "requires operator approval",
            "error": "",
        }
    )


def _signed_pending_result(monkeypatch) -> dict:
    request_id = "ka_1234567890abcdef12345678"
    display_target = "redacted target"
    description = "requires operator approval"
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "7")
    monkeypatch.setenv("HERMES_PROFILE", "worker")
    return {
        "status": "kanban_approval_pending",
        "kanban_approval_pending": True,
        "request_id": request_id,
        "display_target": display_target,
        "description": description,
        "_hermes_kanban_pause_token": issue_kanban_approval_pause_token(
            request_id=request_id,
            task_id="task-1",
            run_id="7",
            profile="worker",
            display_target=display_target,
            description=description,
        ),
    }


@pytest.fixture(autouse=True)
def _accept_synthetic_pending_marker(monkeypatch):
    """Keep loop tests focused on halting; broker trust has separate tests."""

    def parse(result):
        payload = json.loads(result) if isinstance(result, str) else result
        if not isinstance(payload, dict):
            return None
        if payload.get("kanban_approval_pending") is not True:
            return None
        return dict(payload)

    monkeypatch.setattr(
        "agent.execution_context.kanban_approval_pending_metadata",
        parse,
    )


def test_sequential_pause_answers_remaining_calls_without_dispatching_them(tmp_path):
    agent = _agent(tmp_path / ".hermes", "terminal", "read_file")
    assistant = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call("terminal", "call-1"),
            _tool_call("read_file", "call-2"),
        ],
    )
    messages = []

    with patch("run_agent.handle_function_call", return_value=_pending_result()) as dispatch:
        agent._execute_tool_calls_sequential(assistant, messages, "task-1")

    dispatch.assert_called_once()
    assert [message["tool_call_id"] for message in messages] == ["call-1", "call-2"]
    first = json.loads(messages[0]["content"])
    second = json.loads(messages[1]["content"])
    assert first["kanban_approval_pending"] is True
    assert second["reason"] == "kanban_approval_pending"
    assert agent._kanban_approval_pending["request_id"] == "apr_123"


def test_conversation_halts_after_persisted_tool_result_without_second_model_call(tmp_path):
    agent = _agent(tmp_path / ".hermes", "terminal")
    agent.client.chat.completions.create.side_effect = [
        _response(
            finish_reason="tool_calls",
            tool_calls=[_tool_call("terminal", "call-1")],
        ),
        AssertionError("approval pause made an extra model call"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value=_pending_result()),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("do the gated action")

    assert agent.client.chat.completions.create.call_count == 1
    assert result["turn_exit_reason"] == "kanban_approval_pending"
    assert result["kanban_approval_pending"] is True
    assert result["approval_request"]["request_id"] == "apr_123"
    assert result["completed"] is False
    assert result["failed"] is False
    assert [message["role"] for message in result["messages"][-3:]] == [
        "assistant",
        "tool",
        "assistant",
    ]
    assert json.loads(result["messages"][-2]["content"])[
        "kanban_approval_pending"
    ] is True


def test_concurrent_pre_hook_pause_skips_every_sibling_before_submit(
    tmp_path, monkeypatch
):
    agent = _agent(tmp_path / ".hermes", "read_file", "write_file")
    assistant = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call("read_file", "call-1", '{"path":"/tmp/one"}'),
            _tool_call("write_file", "call-2", '{"path":"/tmp/two"}'),
        ],
    )
    messages = []
    hook_calls = []

    def _pre_tool(tool_name, _args, **kwargs):
        hook_calls.append((tool_name, kwargs.get("tool_call_id")))
        if tool_name == "write_file":
            return json.loads(_pending_result())
        return None

    monkeypatch.setattr("hermes_cli.plugins.resolve_pre_tool_block", _pre_tool)
    agent._invoke_tool = MagicMock(
        side_effect=AssertionError("sibling submitted after card was parked")
    )

    agent._execute_tool_calls_concurrent(assistant, messages, "task-1")

    agent._invoke_tool.assert_not_called()
    assert hook_calls == [("read_file", "call-1"), ("write_file", "call-2")]
    assert [message["tool_call_id"] for message in messages] == ["call-1", "call-2"]
    first = json.loads(messages[0]["content"])
    second = json.loads(messages[1]["content"])
    assert first["reason"] == "kanban_approval_pending"
    assert second["kanban_approval_pending"] is True
    assert agent._kanban_approval_pending["request_id"] == "apr_123"


def test_sequential_plugin_pause_preserves_verified_control_until_halt(
    tmp_path, monkeypatch,
):
    agent = _agent(tmp_path / ".hermes", "write_file")
    agent._execution_role = ExecutionRole.KANBAN_OWNER
    assistant = SimpleNamespace(
        content="",
        tool_calls=[_tool_call("write_file", "call-1", '{"path":"/tmp/one"}')],
    )
    messages = []
    pending = _signed_pending_result(monkeypatch)
    monkeypatch.setattr(
        "agent.execution_context.kanban_approval_pending_metadata",
        _real_pending_metadata,
    )
    monkeypatch.setattr(
        "hermes_cli.plugins.resolve_pre_tool_block",
        lambda *_args, **_kwargs: pending,
    )

    token = bind_agent_execution_context(agent)
    try:
        agent._execute_tool_calls_sequential(assistant, messages, "task-1")
    finally:
        reset_agent_execution_context(token)

    assert agent._kanban_approval_pending["request_id"] == pending["request_id"]
    transcript = json.loads(messages[0]["content"])
    assert transcript["kanban_approval_pending"] is True
    assert "_hermes_kanban_pause_token" not in transcript


def test_segmented_pause_skips_every_later_segment_without_dispatch(
    tmp_path, monkeypatch,
):
    agent = _agent(tmp_path / ".hermes", "terminal", "read_file")
    agent._execution_role = ExecutionRole.KANBAN_OWNER
    pending = _signed_pending_result(monkeypatch)
    monkeypatch.setattr(
        "agent.execution_context.kanban_approval_pending_metadata",
        _real_pending_metadata,
    )

    assistant = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call("terminal", "call-gated", '{"command":"rm gated"}'),
            _tool_call(
                "read_file",
                "call-read-1",
                json.dumps({"path": str(tmp_path / "one.txt")}),
            ),
            _tool_call(
                "read_file",
                "call-read-2",
                json.dumps({"path": str(tmp_path / "two.txt")}),
            ),
            _tool_call("terminal", "call-final", '{"command":"touch must-not-run"}'),
        ],
    )
    messages = []
    hook_calls = []

    def _pre_tool(tool_name, _args, **kwargs):
        hook_calls.append((tool_name, kwargs.get("tool_call_id")))
        return pending if kwargs.get("tool_call_id") == "call-gated" else None

    monkeypatch.setattr("hermes_cli.plugins.resolve_pre_tool_block", _pre_tool)

    with (
        patch(
            "run_agent.handle_function_call",
            side_effect=AssertionError("a tool ran after the card was parked"),
        ) as dispatch,
        patch.object(
            agent,
            "_invoke_tool",
            side_effect=AssertionError("a concurrent tool ran after the card was parked"),
        ) as concurrent_dispatch,
    ):
        token = bind_agent_execution_context(agent)
        try:
            agent._execute_tool_calls(assistant, messages, "task-1")
        finally:
            reset_agent_execution_context(token)

    assert hook_calls == [("terminal", "call-gated")]
    dispatch.assert_not_called()
    concurrent_dispatch.assert_not_called()
    assert [message["tool_call_id"] for message in messages] == [
        "call-gated",
        "call-read-1",
        "call-read-2",
        "call-final",
    ]

    gated = json.loads(messages[0]["content"])
    skipped = [json.loads(message["content"]) for message in messages[1:]]
    assert gated["request_id"] == pending["request_id"]
    assert gated["outcome"] == "approval_pending"
    assert all(result["status"] == "skipped" for result in skipped)
    assert all(result["reason"] == "kanban_approval_pending" for result in skipped)
    assert all(result["request_id"] == pending["request_id"] for result in skipped)
    assert all("paused pending approval" in result["message"] for result in skipped)
    assert all(message["effect_disposition"] == "none" for message in messages)


def test_concurrent_plugin_pause_preserves_verified_control_until_halt(
    tmp_path, monkeypatch,
):
    agent = _agent(tmp_path / ".hermes", "read_file", "write_file")
    agent._execution_role = ExecutionRole.KANBAN_OWNER
    assistant = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call("read_file", "call-1", '{"path":"/tmp/one"}'),
            _tool_call("write_file", "call-2", '{"path":"/tmp/two"}'),
        ],
    )
    messages = []
    pending = _signed_pending_result(monkeypatch)
    monkeypatch.setattr(
        "agent.execution_context.kanban_approval_pending_metadata",
        _real_pending_metadata,
    )

    def _pre_tool(tool_name, _args, **_kwargs):
        return pending if tool_name == "write_file" else None

    monkeypatch.setattr("hermes_cli.plugins.resolve_pre_tool_block", _pre_tool)
    agent._invoke_tool = MagicMock(
        side_effect=AssertionError("sibling submitted after card was parked")
    )

    token = bind_agent_execution_context(agent)
    try:
        agent._execute_tool_calls_concurrent(assistant, messages, "task-1")
    finally:
        reset_agent_execution_context(token)

    agent._invoke_tool.assert_not_called()
    assert agent._kanban_approval_pending["request_id"] == pending["request_id"]
    assert [message["tool_call_id"] for message in messages] == ["call-1", "call-2"]
    assert all(
        "_hermes_kanban_pause_token" not in message["content"]
        for message in messages
    )


def test_plugin_approval_pause_remains_structured_and_receives_tool_args(monkeypatch):
    from hermes_cli.plugins import resolve_pre_tool_block

    seen = {}
    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_a, **_k: [
            {"action": "approve", "message": "sensitive write", "rule_key": "ssh"}
        ],
    )

    def _approval(tool_name, reason, **kwargs):
        seen.update(
            tool_name=tool_name,
            reason=reason,
            rule_key=kwargs.get("rule_key"),
            tool_args=kwargs.get("tool_args"),
        )
        return json.loads(_pending_result())

    monkeypatch.setattr("tools.approval.request_tool_approval", _approval)

    result = resolve_pre_tool_block("write_file", {"path": "/tmp/file"})

    assert isinstance(result, dict)
    assert result["kanban_approval_pending"] is True
    assert seen == {
        "tool_name": "write_file",
        "reason": "sensitive write",
        "rule_key": "ssh",
        "tool_args": {"path": "/tmp/file"},
    }

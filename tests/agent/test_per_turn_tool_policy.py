"""Regression tests for the per-turn ``tool_policy=none`` contract."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import shell_hooks
from agent.chat_completion_helpers import build_api_kwargs
from run_agent import AIAgent
from tests.agent.test_turn_context import _FakeAgent, _build


def _tool_def(name: str = "read_file") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _make_agent() -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[_tool_def()]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=4,
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


def _direct_response(content: str = "direct answer"):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _tool_response(name: str = "read_file"):
    call = SimpleNamespace(
        id="call_forbidden",
        type="function",
        function=SimpleNamespace(name=name, arguments='{"path":"/tmp/private"}'),
    )
    message = SimpleNamespace(content="", tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_pre_llm_hook_records_none_policy_on_current_turn():
    agent = _FakeAgent()
    hook_result = [{"context": "bounded customer context", "tool_policy": "none"}]

    with (
        patch("agent.auxiliary_client.set_runtime_main"),
        patch("hermes_cli.plugins.invoke_hook", return_value=hook_result),
    ):
        context = _build(agent)

    assert context.plugin_user_context == "bounded customer context"
    assert context.tool_policy == "none"


@pytest.mark.parametrize(
    ("api_mode", "provider", "base_url", "transport_name"),
    [
        (
            "chat_completions",
            "openrouter",
            "https://openrouter.ai/api/v1",
            "ChatCompletionsTransport",
        ),
        (
            "anthropic_messages",
            "anthropic",
            "https://api.anthropic.com",
            "AnthropicTransport",
        ),
        (
            "codex_responses",
            "openai-codex",
            "https://chatgpt.com/backend-api/codex",
            "ResponsesApiTransport",
        ),
    ],
)
def test_real_transports_remove_tools_without_mutating_agent(
    api_mode: str,
    provider: str,
    base_url: str,
    transport_name: str,
):
    agent = _make_agent()
    agent.api_mode = api_mode
    agent.provider = provider
    agent.base_url = base_url
    agent._base_url_lower = base_url.lower()
    agent._base_url_hostname = base_url.split("/", 3)[2]
    agent._anthropic_base_url = base_url if api_mode == "anthropic_messages" else None
    agent._is_anthropic_oauth = False
    original_tools = agent.tools
    messages = [{"role": "user", "content": "answer directly"}]

    restricted = build_api_kwargs(agent, messages, tool_policy="none")
    normal = build_api_kwargs(agent, messages)

    assert type(agent._get_transport()).__name__ == transport_name
    assert restricted.get("tools", []) == []
    assert agent.tools is original_tools
    assert len(normal.get("tools", [])) == 1


def test_normal_turn_preserves_legacy_build_kwargs_override_signature():
    agent = _make_agent()
    agent.client.chat.completions.create.return_value = _direct_response()

    def legacy_build(api_messages):
        return {"model": "test/model", "messages": api_messages}

    agent.__dict__["_build_api_kwargs"] = legacy_build

    with (
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("ordinary turn")

    assert result["completed"] is True


def test_none_policy_is_local_to_one_turn_and_context_is_ephemeral():
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _direct_response("first"),
        _direct_response("second"),
    ]
    pre_llm_calls = 0

    def invoke(event: str, **_kwargs):
        nonlocal pre_llm_calls
        if event != "pre_llm_call":
            return []
        pre_llm_calls += 1
        if pre_llm_calls == 1:
            return [{"context": "PRIVATE_EPHEMERAL_CONTEXT", "tool_policy": "none"}]
        return []

    with (
        patch("hermes_cli.plugins.invoke_hook", side_effect=invoke),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        first = agent.run_conversation("first question")
        second = agent.run_conversation("second question")

    assert first["completed"] is True
    assert second["completed"] is True
    assert len(agent.client.chat.completions.create.call_args_list) == 2
    first_request, second_request = agent.client.chat.completions.create.call_args_list
    assert first_request.kwargs.get("tools", []) == []
    assert len(second_request.kwargs.get("tools", [])) == 1
    assert "PRIVATE_EPHEMERAL_CONTEXT" in str(first_request.kwargs["messages"])
    assert "PRIVATE_EPHEMERAL_CONTEXT" not in str(second_request.kwargs["messages"])
    assert "PRIVATE_EPHEMERAL_CONTEXT" not in str(first["messages"])
    assert len(agent.tools) == 1


def test_none_policy_rejects_unexpected_tool_call_before_handler():
    agent = _make_agent()
    agent.client.chat.completions.create.return_value = _tool_response()

    with (
        patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "bounded context", "tool_policy": "none"}],
        ),
        patch("run_agent.handle_function_call") as handler,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("answer without tools")

    handler.assert_not_called()
    assert result["completed"] is False
    assert result["partial"] is True
    assert result["error"] == "tool_call_rejected_by_policy"
    assert result["turn_exit_reason"] == "tool_policy_violation"


def test_pre_api_hook_observes_effective_zero_tool_request():
    agent = _make_agent()
    agent.client.chat.completions.create.return_value = _direct_response()
    request_events: list[dict] = []

    def invoke(event: str, **kwargs):
        if event == "pre_llm_call":
            return [{"context": "bounded context", "tool_policy": "none"}]
        if event == "pre_api_request":
            request_events.append(kwargs)
        return []

    with (
        patch("hermes_cli.plugins.invoke_hook", side_effect=invoke),
        patch("hermes_cli.plugins.has_hook", side_effect=lambda event: event == "pre_api_request"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("answer directly")

    assert result["completed"] is True
    assert len(request_events) == 1
    assert request_events[0]["tool_policy"] == "none"
    assert request_events[0]["tool_count"] == 0


def test_shell_pre_llm_hook_preserves_none_policy():
    parsed = shell_hooks._parse_response(
        "pre_llm_call",
        '{"context":"bounded context","tool_policy":"none"}',
    )

    assert parsed == {"context": "bounded context", "tool_policy": "none"}


def test_shell_pre_llm_hook_accepts_none_policy_without_context():
    parsed = shell_hooks._parse_response(
        "pre_llm_call",
        '{"tool_policy":"none"}',
    )

    assert parsed == {"tool_policy": "none"}

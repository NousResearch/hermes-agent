"""Regression tests for malformed non-empty final response recovery."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from tests.run_agent.test_run_agent import (
    _mock_response,
    _mock_tool_call,
)


def _tool_schema(name: str = "terminal") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _make_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[_tool_schema()]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="http://127.0.0.1:9090/v1",
            model="dflash",
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


def test_degenerate_final_after_tool_calls_gets_regeneration_prompt():
    agent = _make_agent()
    tool_call = _mock_tool_call(name="terminal", arguments="{}")
    malformed = (
        "Now I have a clear picture. The key pattern is:\n\n"
        "1. signal: killed, exit code: -1\n"
        "2. process exited but not StateStopping\n"
        "3. no valid JSON data found in stream\n"
        "4. No d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    )

    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", tool_calls=[tool_call], finish_reason="tool_calls"),
        _mock_response(content=malformed, finish_reason="stop"),
        _mock_response(content="The backend did not die at that timestamp.", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value='{"ok": true}'),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("diagnose dflash")

    assert agent.client.chat.completions.create.call_count == 3
    second_retry_kwargs = agent.client.chat.completions.create.call_args_list[2].kwargs
    retry_messages = second_retry_kwargs["messages"]
    assert retry_messages[-1]["role"] == "user"
    assert "Regenerate a concise, complete final answer" in retry_messages[-1]["content"]
    assert retry_messages[-2]["content"] == "[malformed final response omitted]"
    assert result["final_response"] == "The backend did not die at that timestamp."
    assert "No d!!!!" not in result["final_response"]


def test_open_connector_final_after_tool_calls_gets_regeneration_prompt():
    agent = _make_agent()
    tool_call = _mock_tool_call(name="terminal", arguments="{}")
    malformed = (
        "So the current health run doesn't show `missing_checkout` for "
        "meshboard or cosmic-core anymore. The tasks were auto-generated "
        "from a previous health run. Let me pick a task that's genuinely "
        "actionable. Let me look at the `profile-unavailable-launcher-exit-rc-prefix` "
        "task since it's small and concrete:\n\n"
        "Actually, let me pick something more straightforward. The "
        "health-doctor tasks for missing checkouts are stale (the repos exist "
        "now). Let me pick the `meshboard-store-outputs-flag` task -- it's a "
        "small schema extension. But"
    )

    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", tool_calls=[tool_call], finish_reason="tool_calls"),
        _mock_response(content=malformed, finish_reason="stop"),
        _mock_response(content="I found the task and claimed it.", finish_reason="stop"),
    ]

    with (
        patch("run_agent.handle_function_call", return_value='{"ok": true}'),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("pick a MeshBoard task")

    assert agent.client.chat.completions.create.call_count == 3
    retry_messages = agent.client.chat.completions.create.call_args_list[2].kwargs["messages"]
    assert retry_messages[-1]["role"] == "user"
    assert "Regenerate a concise, complete final answer" in retry_messages[-1]["content"]
    assert retry_messages[-2]["content"] == "[malformed final response omitted]"
    assert result["final_response"] == "I found the task and claimed it."


def test_complete_final_with_tool_calls_is_not_malformed():
    agent = _make_agent()
    content = (
        "I found the stale health-doctor tasks and selected a different "
        "MeshBoard task. The next step is to claim the task and implement the "
        "schema change. Let me know if you want a narrower task instead."
    )
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
    ]

    assert agent._detect_malformed_tool_final_response(
        content,
        "stop",
        messages,
    ) is None


def test_degenerate_final_repeats_activates_fallback_provider():
    agent = _make_agent()
    agent._fallback_chain = [
        {
            "provider": "openrouter",
            "model": "openai/gpt-test-fallback",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "fallback-key",
        }
    ]
    tool_call = _mock_tool_call(name="terminal", arguments="{}")
    malformed = (
        "I found the issue after reviewing logs and state. "
        "The final line is broken!!!!!!!!!!!!!!!!!!!!!!!!"
    )

    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", tool_calls=[tool_call], finish_reason="tool_calls"),
        _mock_response(content=malformed, finish_reason="stop"),
        _mock_response(content=malformed, finish_reason="stop"),
        _mock_response(content="Fallback produced a clean answer.", finish_reason="stop"),
    ]
    fallback_client = MagicMock()
    fallback_client.api_key = "fallback-key"
    fallback_client.base_url = "https://openrouter.ai/api/v1"
    fallback_client.chat = agent.client.chat

    with (
        patch("run_agent.handle_function_call", return_value='{"ok": true}'),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, "openai/gpt-test-fallback"),
        ),
        patch.object(agent, "_create_request_openai_client", return_value=fallback_client),
    ):
        result = agent.run_conversation("diagnose dflash")

    assert agent.client.chat.completions.create.call_count == 4
    assert agent.provider == "openrouter"
    assert agent.model == "openai/gpt-test-fallback"
    assert result["final_response"] == "Fallback produced a clean answer."

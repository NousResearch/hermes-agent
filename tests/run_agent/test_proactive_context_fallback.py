"""Tests for proactive pre-call fallback when the active model cannot fit context."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _mock_response(content="ok"):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(*, fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent._cached_system_prompt = "You are helpful."
        agent.tool_delay = 0
        return agent


def test_proactive_context_fallback_happens_before_first_api_call():
    agent = _make_agent(
        fallback_model={"provider": "openrouter", "model": "fallback/model"},
    )
    primary_client = MagicMock()
    primary_client.chat.completions.create.return_value = _mock_response("primary should not run")
    agent.client = primary_client
    agent.context_compressor.context_length = 100000
    agent.context_compressor.threshold_tokens = 80000

    fallback_client = MagicMock()
    fallback_client.api_key = "fallback-key"
    fallback_client.base_url = "https://openrouter.ai/api/v1"
    fallback_client.chat.completions.create.return_value = _mock_response("fallback success")

    history = [
        {"role": "user", "content": f"message {i}"} if i % 2 == 0 else {"role": "assistant", "content": f"reply {i}"}
        for i in range(8)
    ]

    with (
        patch("run_agent.estimate_request_tokens_rough", return_value=150000),
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(fallback_client, "fallback/model")),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
        patch.object(agent, "_compress_context") as mock_compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("latest message", conversation_history=history)

    assert result["completed"] is True
    assert result["final_response"] == "fallback success"
    assert primary_client.chat.completions.create.call_count == 0
    assert fallback_client.chat.completions.create.call_count == 1
    mock_compress.assert_not_called()
    assert agent.provider == "openrouter"
    assert agent.model == "fallback/model"

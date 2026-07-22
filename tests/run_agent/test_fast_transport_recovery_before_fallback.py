"""Regression coverage for #69186.

Fast SDK/socket connection errors should reach the existing primary-client
recovery cycle before fallback.  Stale-stream failures remain eligible for
bounded eager fallback so #22277 does not regress.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


class APIConnectionError(Exception):
    pass


def _tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def _response(content: str):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test-model", usage=None)


def _agent():
    fallback_chain = [
        {
            "provider": "deepseek",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
        }
    ]
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key-abcdef12",
            base_url="https://primary.example.com/v1",
            provider="custom",
            model="primary-model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_chain,
        )
    agent.client = MagicMock()
    agent._api_max_retries = 3
    return agent


def _fallback_client():
    client = MagicMock()
    client.api_key = "fallback-key-abcdef12"
    client.base_url = "https://api.deepseek.com"
    client._custom_headers = None
    client.default_headers = None
    return client


def _run(agent, api_call, *, fallback_client=None):
    patches = [
        patch.object(agent, "_interruptible_api_call", side_effect=api_call),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("run_agent.OpenAI", return_value=MagicMock()),
        patch("agent.agent_runtime_helpers.time.sleep"),
        patch("agent.conversation_loop.time.sleep"),
        patch("agent.model_metadata.get_model_context_length", return_value=200000),
    ]
    if fallback_client is not None:
        patches.extend([
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(fallback_client, "deepseek-chat"),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda model, provider: model,
            ),
        ])

    entered = []
    try:
        for ctx in patches:
            entered.append(ctx)
            ctx.__enter__()
        return agent.run_conversation("hello")
    finally:
        for ctx in reversed(entered):
            ctx.__exit__(None, None, None)


def test_fast_connection_errors_recover_primary_before_fallback():
    """Three fast failures exhaust retries, then a rebuilt primary succeeds."""
    agent = _agent()
    calls = []

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        if len(calls) <= 3:
            raise APIConnectionError("Connection error.")
        return _response("Recovered on rebuilt primary")

    with (
        patch.object(
            agent,
            "_try_recover_primary_transport",
            wraps=agent._try_recover_primary_transport,
        ) as recover,
        patch.object(
            agent,
            "_try_activate_fallback",
            wraps=agent._try_activate_fallback,
        ) as fallback,
    ):
        result = _run(agent, api_call, fallback_client=_fallback_client())

    assert result["completed"] is True
    assert result["final_response"] == "Recovered on rebuilt primary"
    assert calls == [("custom", "primary-model")] * 4
    recover.assert_called_once()
    fallback.assert_not_called()


def test_stale_stream_timeout_keeps_bounded_eager_fallback():
    """A stale-detector-derived timeout still falls back after one retry."""
    agent = _agent()
    agent._consecutive_stale_streams = 1
    calls = []

    def api_call(_kwargs):
        calls.append((agent.provider, agent.model))
        if agent.provider == "custom":
            raise APIConnectionError("Connection closed after stale-stream kill.")
        return _response("Recovered via fallback")

    result = _run(agent, api_call, fallback_client=_fallback_client())

    assert result["completed"] is True
    assert result["final_response"] == "Recovered via fallback"
    assert calls == [
        ("custom", "primary-model"),
        ("custom", "primary-model"),
        ("deepseek", "deepseek-chat"),
    ]
    assert agent._fallback_activated is True

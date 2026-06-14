"""Regression tests for anthropic_messages partial-stream recovery."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_constants import PARTIAL_STREAM_STUB_ID


def _make_agent():
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://api.anthropic.com/v1",
            model="claude-sonnet-test",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent.api_mode = "anthropic_messages"
    agent.provider = "anthropic"
    agent._interrupt_requested = False
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _make_anthropic_partial_stub(
    text: str,
    dropped_tool_names: list[str] | None = None,
):
    return SimpleNamespace(
        id=PARTIAL_STREAM_STUB_ID,
        type="message",
        role="assistant",
        model="claude-sonnet-test",
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="max_tokens",
        stop_sequence=None,
        usage=None,
        _dropped_tool_names=dropped_tool_names,
    )


def _make_anthropic_response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        id="msg_test_01",
        type="message",
        role="assistant",
        model="claude-sonnet-test",
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=None,
    )


def test_interruptible_streaming_returns_anthropic_shaped_partial_stub(monkeypatch):
    """Malformed Anthropic finalization after delivery must not fall back to OpenAI shape."""
    from agent.transports import get_transport

    agent = _make_agent()
    agent.stream_delta_callback = lambda text: None

    events = [
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text="Recovered partial answer"),
        )
    ]

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.__iter__ = MagicMock(return_value=iter(events))
    mock_stream.get_final_message.side_effect = IndexError("list index out of range")
    mock_stream.response = None

    agent._anthropic_client = MagicMock()
    agent._anthropic_client.messages.stream.return_value = mock_stream

    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    response = agent._interruptible_streaming_api_call({})

    assert response.id == PARTIAL_STREAM_STUB_ID
    assert isinstance(response.content, list)
    assert response.content[0].type == "text"
    assert response.content[0].text == "Recovered partial answer"
    assert response.stop_reason == "max_tokens"
    transport = get_transport("anthropic_messages")
    assert transport.validate_response(response) is True
    assert transport.normalize_response(response).finish_reason == "length"


def test_run_conversation_continues_after_anthropic_partial_stub():
    """A valid Anthropic-shaped stub should trigger bounded continuation, not invalid retries."""
    agent = _make_agent()
    partial_stub = _make_anthropic_partial_stub("Recovered partial answer from stream")
    continuation = _make_anthropic_response("and here is the rest.")

    with (
        patch.object(
            agent,
            "_interruptible_api_call",
            side_effect=[partial_stub, continuation],
        ) as mock_api,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("continue")

    assert mock_api.call_count == 2
    assert result["completed"] is True
    assert result["api_calls"] == 2
    assert "Recovered partial answer from stream" in result["final_response"]
    assert "and here is the rest." in result["final_response"]

"""Regression coverage for incomplete native Anthropic Messages streams (#121320)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _stream_cm(final_message, events):
    stream = MagicMock()
    stream.__iter__ = MagicMock(return_value=iter(events))
    stream.get_final_message = MagicMock(return_value=final_message)
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=stream)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _event(event_type, **fields):
    return SimpleNamespace(type=event_type, **fields)


def _text_delta(text):
    return _event(
        "content_block_delta",
        delta=SimpleNamespace(type="text_delta", text=text),
    )


def _thinking_delta(thinking):
    return _event(
        "content_block_delta",
        delta=SimpleNamespace(type="thinking_delta", thinking=thinking),
    )


def _agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://api.anthropic.com",
        model="claude-test",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "anthropic_messages"
    agent._interrupt_requested = False
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = "test-key"
    agent._create_request_anthropic_client = lambda *args, **kwargs: agent._anthropic_client
    return agent


@pytest.mark.parametrize(
    ("dropped_event", "expected_callback"),
    [
        (_thinking_delta("PARTIAL-THOUGHT"), "RECOVERED"),
    ],
)
def test_anthropic_eof_before_message_stop_retries_without_delivering_partial_output(
    monkeypatch, dropped_event, expected_callback,
):
    """A final SDK snapshot cannot turn an unterminated SSE response into success."""
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "1")
    agent = _agent()
    delivered = []
    agent.stream_delta_callback = delivered.append
    dropped = SimpleNamespace(content=[SimpleNamespace(type="text", text="HALF-ANSWER")], stop_reason="end_turn")
    recovered = SimpleNamespace(content=[SimpleNamespace(type="text", text="RECOVERED")], stop_reason="end_turn")
    agent._anthropic_client.messages.stream.side_effect = [
        _stream_cm(dropped, [_event("message_start"), dropped_event]),
        _stream_cm(recovered, [_event("message_start"), _text_delta("RECOVERED"), _event("message_stop")]),
    ]

    response = agent._interruptible_streaming_api_call({"model": "claude-test"})

    assert response is recovered
    assert agent._anthropic_client.messages.stream.call_count == 2
    assert delivered == [expected_callback]


def test_anthropic_message_stop_accepts_completed_stream_without_retry():
    agent = _agent()
    completed = SimpleNamespace(content=[SimpleNamespace(type="text", text="done")], stop_reason="end_turn")
    agent._anthropic_client.messages.stream.return_value = _stream_cm(
        completed,
        [_event("message_start"), _text_delta("done"), _event("message_stop")],
    )

    assert agent._interruptible_streaming_api_call({"model": "claude-test"}) is completed
    assert agent._anthropic_client.messages.stream.call_count == 1


def test_auxiliary_anthropic_eof_before_message_stop_raises_empty_stream():
    from agent.anthropic_adapter import _stream_final_message
    from agent.errors import EmptyStreamError

    partial = SimpleNamespace(content=[SimpleNamespace(type="text", text="HALF")], stop_reason="end_turn")
    with pytest.raises(EmptyStreamError, match="message_stop"):
        _stream_final_message(
            lambda **kwargs: _stream_cm(partial, [_event("message_start"), _text_delta("HALF")]),
            {"model": "claude-test"}, "", None, None,
        )

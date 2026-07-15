from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.anthropic_adapter import create_anthropic_message
from agent.errors import EmptyStreamError


OCI_ANTHROPIC_URL = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/anthropic"


def _empty_snapshot_stream_cm():
    cm = MagicMock()
    stream = MagicMock()
    stream.__iter__ = MagicMock(return_value=iter(()))
    stream.get_final_message = MagicMock(side_effect=AssertionError())
    cm.__enter__ = MagicMock(return_value=stream)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def test_create_message_falls_back_when_sdk_stream_has_no_final_snapshot():
    client = MagicMock()
    client.messages.stream.return_value = _empty_snapshot_stream_cm()
    expected = SimpleNamespace(content=[], stop_reason="end_turn")
    client.messages.create.return_value = expected

    response = create_anthropic_message(
        client,
        {
            "model": "claude-opus-4-8",
            "messages": [{"role": "user", "content": "Reply exactly OK"}],
            "max_tokens": 16,
        },
    )

    assert response is expected
    client.messages.stream.assert_called_once()
    client.messages.create.assert_called_once()


@pytest.mark.parametrize(
    "empty_final",
    [None, SimpleNamespace(content=[], stop_reason=None)],
)
def test_create_message_falls_back_for_empty_final_message_shapes(empty_final):
    client = MagicMock()
    cm = _empty_snapshot_stream_cm()
    get_final_message = cm.__enter__.return_value.get_final_message
    get_final_message.side_effect = None
    get_final_message.return_value = empty_final
    client.messages.stream.return_value = cm
    expected = SimpleNamespace(content=[], stop_reason="end_turn")
    client.messages.create.return_value = expected

    response = create_anthropic_message(
        client,
        {"model": "claude-opus-4-8", "messages": [], "max_tokens": 16},
    )

    assert response is expected
    client.messages.create.assert_called_once()


def test_create_message_does_not_hide_nonempty_application_assertions():
    client = MagicMock()
    cm = _empty_snapshot_stream_cm()
    cm.__enter__.return_value.get_final_message.side_effect = AssertionError("bad app state")
    client.messages.stream.return_value = cm

    with pytest.raises(AssertionError, match="bad app state"):
        create_anthropic_message(
            client,
            {"model": "claude-opus-4-8", "messages": [], "max_tokens": 16},
        )

    client.messages.create.assert_not_called()


def test_create_message_does_not_hide_blank_assertion_before_final_message():
    client = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(side_effect=AssertionError())
    client.messages.stream.return_value = cm

    with pytest.raises(AssertionError):
        create_anthropic_message(
            client,
            {"model": "claude-opus-4-8", "messages": [], "max_tokens": 16},
        )

    client.messages.create.assert_not_called()


def test_oci_empty_stream_disables_streaming_without_retrying(monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    agent = AIAgent(
        api_key="test-key",
        base_url=OCI_ANTHROPIC_URL,
        model="claude-opus-4-8",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = "test-key"
    agent._anthropic_client.messages.stream.return_value = _empty_snapshot_stream_cm()

    with pytest.raises(EmptyStreamError):
        agent._interruptible_streaming_api_call({})

    assert agent._disable_streaming is True
    agent._anthropic_client.messages.stream.assert_called_once()


def test_other_third_party_empty_stream_keeps_transient_retry_behavior(monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    agent = AIAgent(
        api_key="test-key",
        base_url="https://anthropic-proxy.example.com/v1",
        model="claude-opus-4-8",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = "test-key"
    agent._rebuild_anthropic_client = MagicMock()
    agent._anthropic_client.messages.stream.return_value = _empty_snapshot_stream_cm()

    with pytest.raises(EmptyStreamError):
        agent._interruptible_streaming_api_call({})

    assert getattr(agent, "_disable_streaming", False) is False
    assert agent._anthropic_client.messages.stream.call_count == 3

"""Regression test: /retry must return the agent response, not None.

Before the fix in PR #441, _handle_retry_command() called
_handle_message(retry_event) but discarded its return value with `return None`,
so users never received the final response.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from gateway.run import GatewayRunner
from gateway.platforms.base import MessageEvent, MessageType


@pytest.fixture
def gateway(tmp_path):
    config = MagicMock()
    config.sessions_dir = tmp_path
    config.max_context_messages = 20
    gw = GatewayRunner.__new__(GatewayRunner)
    gw.config = config
    gw.session_store = MagicMock()
    return gw


@pytest.mark.asyncio
async def test_retry_returns_response_not_none(gateway):
    """_handle_retry_command must return the inner handler response, not None."""
    gateway.session_store.get_or_create_session.return_value = MagicMock(
        session_id="test-session"
    )
    gateway.session_store.load_transcript.return_value = [
        {"role": "user", "content": "Hello Hermes"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    gateway.session_store.rewrite_transcript = MagicMock()
    expected_response = "Hi there! (retried)"
    gateway._handle_message = AsyncMock(return_value=expected_response)
    event = MessageEvent(
        text="/retry",
        message_type=MessageType.TEXT,
        source=MagicMock(),
    )
    result = await gateway._handle_retry_command(event)
    assert result is not None, "/retry must not return None"
    assert result == expected_response


@pytest.mark.asyncio
async def test_retry_no_previous_message(gateway):
    """If there is no previous user message, return early with a message."""
    gateway.session_store.get_or_create_session.return_value = MagicMock(
        session_id="test-session"
    )
    gateway.session_store.load_transcript.return_value = []
    event = MessageEvent(
        text="/retry",
        message_type=MessageType.TEXT,
        source=MagicMock(),
    )
    result = await gateway._handle_retry_command(event)
    assert result == "No previous message to retry."


@pytest.mark.asyncio
async def test_retry_preserves_event_context_when_rebuilding_message(gateway):
    gateway.session_store.get_or_create_session.return_value = MagicMock(
        session_id="test-session"
    )
    gateway.session_store.load_transcript.return_value = [
        {"role": "user", "content": "原始问题"},
        {"role": "assistant", "content": "旧回复"},
    ]
    gateway.session_store.rewrite_transcript = MagicMock()

    captured = {}

    async def fake_handle_message(event):
        captured["event"] = event
        return "retried"

    gateway._handle_message = AsyncMock(side_effect=fake_handle_message)
    event = MessageEvent(
        text="/retry",
        message_type=MessageType.TEXT,
        source=MagicMock(),
        raw_message={"platform": "qq"},
        message_id="qq-1",
        metadata={"explicit_addressed": True, "address_reason": "bot_mention"},
        reply_to_message_id="bot-msg-1",
        reply_to_text="上一条",
    )

    result = await gateway._handle_retry_command(event)

    assert result == "retried"
    retried_event = captured["event"]
    assert retried_event.text == "原始问题"
    assert retried_event.raw_message == {"platform": "qq"}
    assert retried_event.message_id == "qq-1"
    assert retried_event.metadata["explicit_addressed"] is True
    assert retried_event.reply_to_message_id == "bot-msg-1"
    assert retried_event.reply_to_text == "上一条"

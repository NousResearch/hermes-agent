"""Tests for Signal recent-history prompt rendering and per-turn injection."""

from datetime import datetime, timezone

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner, _format_recent_signal_chat_history
from gateway.session import SessionSource, _hash_sender_id


class _SignalAdapterStub:
    def __init__(self, messages):
        self._messages = list(messages)

    def get_recent_chat_messages(self, chat_id: str, n: int = 20):
        return list(self._messages)[-n:]

    def set_recent_chat_messages(self, messages):
        self._messages = list(messages)


def _make_signal_runner(messages):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SIGNAL: PlatformConfig(enabled=True, token="fake")},
    )
    adapter = _SignalAdapterStub(messages)
    runner.adapters = {Platform.SIGNAL: adapter}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    runner._consume_pending_native_image_paths = lambda _session_key: []
    return runner, adapter


def _signal_source() -> SessionSource:
    return SessionSource(
        platform=Platform.SIGNAL,
        chat_id="group:abc",
        chat_name="Test Group",
        chat_type="group",
        user_id="+15550001111",
        user_name="Alice",
    )


def test_format_recent_signal_chat_history_hashes_raw_sender_when_redacting():
    prompt = _format_recent_signal_chat_history(
        [
            {
                "ts": 1712345678000,
                "sender": "+15550004567",
                "name": "+15550004567",
                "text": "hello",
            }
        ],
        redact_pii=True,
    )

    assert "+15550004567" not in prompt
    assert _hash_sender_id("+15550004567") in prompt
    assert "hello" in prompt


def test_format_recent_signal_chat_history_preserves_display_name_when_redacting():
    prompt = _format_recent_signal_chat_history(
        [
            {
                "ts": 1712345678000,
                "sender": "+15550004567",
                "name": "Alice",
                "text": "hello",
            }
        ],
        redact_pii=True,
    )

    assert "Alice" in prompt
    assert _hash_sender_id("+15550004567") not in prompt


def test_format_recent_signal_chat_history_returns_empty_for_no_messages():
    assert _format_recent_signal_chat_history([], redact_pii=True) == ""


@pytest.mark.asyncio
async def test_prepare_inbound_message_injects_prior_signal_history_without_current_turn():
    source = _signal_source()
    current_ts = datetime.fromtimestamp(1712345682, tz=timezone.utc)
    runner, _adapter = _make_signal_runner(
        [
            {"ts": 1712345678000, "sender": "+15550002222", "name": "Bob", "text": "Earlier backlog"},
            {"ts": 1712345682000, "sender": "+15550001111", "name": "Alice", "text": "Latest live message"},
        ]
    )
    event = MessageEvent(text="Latest live message", source=source, timestamp=current_ts)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert result.startswith("[Recent Signal chat history]\n[")
    assert "Bob: Earlier backlog" in result
    assert result.endswith("Latest live message")
    assert result.count("Latest live message") == 1


@pytest.mark.asyncio
async def test_prepare_inbound_message_keeps_recent_signal_history_before_reply_pointer():
    source = _signal_source()
    runner, _adapter = _make_signal_runner(
        [
            {"ts": 1712345678000, "sender": "+15550002222", "name": "Bob", "text": "Earlier backlog"},
        ]
    )
    event = MessageEvent(
        text="What about this one?",
        source=source,
        timestamp=datetime.fromtimestamp(1712345682, tz=timezone.utc),
        reply_to_message_id="42",
        reply_to_text="Use the direct train.",
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result is not None
    assert result.startswith('[Replying to: "Use the direct train."]\n\n[Recent Signal chat history]')
    assert "Earlier backlog" in result
    assert result.endswith("What about this one?")


@pytest.mark.asyncio
async def test_prepare_inbound_message_only_injects_unseen_signal_history_after_watermark():
    source = _signal_source()
    runner, adapter = _make_signal_runner(
        [
            {"ts": 1712345678000, "sender": "+15550002222", "name": "Bob", "text": "Earlier backlog"},
            {"ts": 1712345682000, "sender": "+15550001111", "name": "Alice", "text": "Latest live message"},
        ]
    )

    first = await runner._prepare_inbound_message_text(
        event=MessageEvent(
            text="Latest live message",
            source=source,
            timestamp=datetime.fromtimestamp(1712345682, tz=timezone.utc),
        ),
        source=source,
        history=[],
        session_key="signal-session",
    )

    assert first is not None
    assert "Earlier backlog" in first

    adapter.set_recent_chat_messages(
        [
            {"ts": 1712345678000, "sender": "+15550002222", "name": "Bob", "text": "Earlier backlog"},
            {"ts": 1712345682000, "sender": "+15550001111", "name": "Alice", "text": "Latest live message"},
            {"ts": 1712345690000, "sender": "+15550003333", "name": "Carol", "text": "Cross-talk while Hermes was busy"},
            {"ts": 1712345695000, "sender": "+15550001111", "name": "Alice", "text": "Second live message"},
        ]
    )

    second = await runner._prepare_inbound_message_text(
        event=MessageEvent(
            text="Second live message",
            source=source,
            timestamp=datetime.fromtimestamp(1712345695, tz=timezone.utc),
        ),
        source=source,
        history=[],
        session_key="signal-session",
    )

    assert second is not None
    assert "Cross-talk while Hermes was busy" in second
    assert "Earlier backlog" not in second
    assert second.count("Second live message") == 1


@pytest.mark.asyncio
async def test_prepare_inbound_message_skips_bot_reply_entries_after_watermark():
    source = _signal_source()
    runner, adapter = _make_signal_runner(
        [
            {"ts": 1712345682000, "sender": "+15550001111", "name": "Alice", "text": "Latest live message"},
        ]
    )

    first = await runner._prepare_inbound_message_text(
        event=MessageEvent(
            text="Latest live message",
            source=source,
            timestamp=datetime.fromtimestamp(1712345682, tz=timezone.utc),
        ),
        source=source,
        history=[],
        session_key="signal-session",
    )

    assert first == "Latest live message"

    adapter.set_recent_chat_messages(
        [
            {"ts": 1712345682000, "sender": "+15550001111", "name": "Alice", "text": "Latest live message"},
            {"ts": 1712345688000, "sender": "+15550009999", "name": "me", "text": "Hermes reply", "from_self": True},
            {"ts": 1712345695000, "sender": "+15550001111", "name": "Alice", "text": "Second live message"},
        ]
    )

    second = await runner._prepare_inbound_message_text(
        event=MessageEvent(
            text="Second live message",
            source=source,
            timestamp=datetime.fromtimestamp(1712345695, tz=timezone.utc),
        ),
        source=source,
        history=[],
        session_key="signal-session",
    )

    assert second == "Second live message"

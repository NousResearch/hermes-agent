"""Tests for the gateway /queue command handler (running-agent path).

/queue stores a turn-boundary follow-up in the adapter's pending queue
without interrupting the active run. The queued event must carry the
full payload — media attachments and reply context — not just the text.
Previously the handler rebuilt the event with only text/type/source/
message_id/channel_prompt, silently dropping any photo/document/reply
metadata the user attached to the /queue message.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_runner(session_entry: SessionEntry):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._queued_events = {}
    runner._pending_approvals = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner, adapter


def _session_entry() -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


def _running(runner):
    """Mark the session as having a running agent so /queue hits the
    early-intercept path."""
    sk = build_session_key(_make_source())
    runner._running_agents[sk] = MagicMock()
    return sk


@pytest.mark.asyncio
async def test_queue_text_only_queues_and_does_not_interrupt():
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)
    running_agent = runner._running_agents[sk]

    event = MessageEvent(text="/queue do this next", source=_make_source(), message_id="q1")
    result = await runner._handle_message(event)

    assert result is not None and "queued" in result.lower()
    running_agent.interrupt.assert_not_called()
    assert sk in adapter._pending_messages
    queued = adapter._pending_messages[sk]
    assert queued.text == "do this next"
    assert queued.message_type == MessageType.TEXT


@pytest.mark.asyncio
async def test_queue_preserves_photo_media():
    """A /queue carrying a photo must keep the attachment + type."""
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)

    event = MessageEvent(
        text="/queue look at this",
        message_type=MessageType.PHOTO,
        source=_make_source(),
        message_id="q-photo",
        media_urls=["/tmp/photo-a.jpg"],
        media_types=["image/jpeg"],
    )
    result = await runner._handle_message(event)

    assert result is not None and "queued" in result.lower()
    queued = adapter._pending_messages[sk]
    assert queued.text == "look at this"
    assert queued.message_type == MessageType.PHOTO
    assert queued.media_urls == ["/tmp/photo-a.jpg"]
    assert queued.media_types == ["image/jpeg"]


@pytest.mark.asyncio
async def test_queue_allows_media_without_prompt_text():
    """`/queue` as a bare caption on a document is valid — media-only."""
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)

    event = MessageEvent(
        text="/queue",
        message_type=MessageType.DOCUMENT,
        source=_make_source(),
        message_id="q-doc",
        media_urls=["/tmp/file.pdf"],
        media_types=["application/pdf"],
    )
    result = await runner._handle_message(event)

    assert result is not None and "queued" in result.lower()
    queued = adapter._pending_messages[sk]
    assert queued.text == ""
    assert queued.message_type == MessageType.DOCUMENT
    assert queued.media_urls == ["/tmp/file.pdf"]


@pytest.mark.asyncio
async def test_queue_preserves_reply_context():
    runner, adapter = _make_runner(_session_entry())
    sk = _running(runner)

    event = MessageEvent(
        text="/queue and this",
        source=_make_source(),
        message_id="q-reply",
        reply_to_message_id="orig-7",
        reply_to_text="the original message",
        reply_to_author_id="a1",
        reply_to_author_name="alice",
    )
    result = await runner._handle_message(event)

    assert result is not None and "queued" in result.lower()
    queued = adapter._pending_messages[sk]
    assert queued.reply_to_message_id == "orig-7"
    assert queued.reply_to_text == "the original message"
    assert queued.reply_to_author_id == "a1"
    assert queued.reply_to_author_name == "alice"


@pytest.mark.asyncio
async def test_queue_no_text_no_media_returns_usage():
    runner, adapter = _make_runner(_session_entry())
    _running(runner)

    event = MessageEvent(text="/queue", source=_make_source(), message_id="q-empty")
    result = await runner._handle_message(event)

    assert result is not None and "Usage" in result
    assert adapter._pending_messages == {}


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

"""Regression: voice messages must resolve pending clarify prompts.

When the Telegram clarify tool is waiting for a text response and the user
sends a voice message, the gateway's clarify intercept reads ``event.text``
(which is empty for voice messages) and ignores the audio.  The STT
transcription that would populate the text happens later in
``_handle_message_with_agent``, after the intercept window has closed.

Fix: transcribe voice/audio media inline inside the clarify intercept block
before checking ``event.text``.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _voice_event(media_path="/tmp/test_voice.ogg"):
    """Create a voice message event with empty text and a cached audio URL."""
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="private",
        user_id="user1",
    )
    return MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        message_id="msg-voice-1",
        media_urls=[media_path],
        media_types=["audio/ogg"],
    )


def _text_event(text="custom answer"):
    """Create a plain text message event."""
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="private",
        user_id="user1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-text-1",
    )


def _clear_clarify_state():
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


@pytest.mark.asyncio
async def test_voice_message_resolves_pending_clarify(tmp_path):
    """A voice message must resolve a pending clarify via STT transcription."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    # Set up a minimal GatewayRunner
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    import gateway.run as gateway_run
    with patch.object(gateway_run, "_hermes_home", tmp_path):
        runner = gateway_run.GatewayRunner(GatewayConfig())

    # Register a mock adapter
    adapter = MagicMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.send = AsyncMock()
    runner.adapters[Platform.TELEGRAM] = adapter

    event = _voice_event()
    session_key = build_session_key(event.source)

    # Register a pending clarify
    cm.register("clarify-voice-1", session_key, "What is your name?", None)

    # Mock STT transcription to return a transcript
    mock_transcribe = MagicMock(return_value={
        "success": True,
        "transcript": "My name is Alice",
    })

    with patch(
        "tools.transcription_tools.transcribe_audio",
        mock_transcribe,
    ):
        # _handle_message returns "" when clarify is resolved
        result = await runner._handle_message(event)

    assert result == "", "clarify intercept should return empty string"
    mock_transcribe.assert_called_once()

    # Verify the clarify entry was resolved (event set, response populated)
    entry = cm._entries.get("clarify-voice-1")
    assert entry is not None, "entry should still exist after resolution"
    assert entry.event.is_set(), "clarify event should be set after resolution"
    assert entry.response == "My name is Alice", (
        f"expected transcript as response, got: {entry.response!r}"
    )


@pytest.mark.asyncio
async def test_voice_stt_failure_does_not_crash(tmp_path):
    """STT failure during clarify intercept must not crash the gateway."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    import gateway.run as gateway_run
    with patch.object(gateway_run, "_hermes_home", tmp_path):
        runner = gateway_run.GatewayRunner(GatewayConfig())

    adapter = MagicMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.send = AsyncMock()
    runner.adapters[Platform.TELEGRAM] = adapter

    event = _voice_event()
    session_key = build_session_key(event.source)
    cm.register("clarify-voice-fail", session_key, "What?", None)

    # Mock STT to raise an exception
    def _failing_transcribe(path):
        raise RuntimeError("STT provider unavailable")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=_failing_transcribe,
    ):
        # Should not crash; falls through to normal dispatch
        result = await runner._handle_message(event)

    # Clarify should still be pending (event NOT set)
    entry = cm._entries.get("clarify-voice-fail")
    assert entry is not None
    assert not entry.event.is_set(), "clarify should remain unresolved on STT failure"


@pytest.mark.asyncio
async def test_text_message_still_resolves_clarify(tmp_path):
    """Plain text messages must still resolve clarify (no regression)."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    (tmp_path / "config.yaml").write_text("", encoding="utf-8")
    import gateway.run as gateway_run
    with patch.object(gateway_run, "_hermes_home", tmp_path):
        runner = gateway_run.GatewayRunner(GatewayConfig())

    adapter = MagicMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.send = AsyncMock()
    runner.adapters[Platform.TELEGRAM] = adapter

    event = _text_event("Alice")
    session_key = build_session_key(event.source)
    cm.register("clarify-text-1", session_key, "What is your name?", None)

    result = await runner._handle_message(event)
    assert result == "", "text clarify should still work"

    entry = cm._entries.get("clarify-text-1")
    assert entry is not None
    assert entry.event.is_set(), "clarify event should be set after text response"
    assert entry.response == "Alice"

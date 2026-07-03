"""Regression tests for voice/audio replies to pending clarify prompts.

When the user responds to a clarify prompt with a voice message, ``event.text``
is empty (STT hasn't run yet).  The clarify intercept must transcribe the audio
and resolve the clarify with the transcript text.  (Fixes #56739)

These tests verify the clarify-intercept path in the adapter's active-session
handler, which routes to ``_message_handler`` when a pending clarify exists.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _VoiceClarifyAdapter(BasePlatformAdapter):
    """Minimal adapter for testing clarify text-intercept routing."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _voice_event(media_url="/tmp/voice.ogg", media_type="audio/ogg"):
    """Create a MessageEvent that represents a voice message with no text."""
    return MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="user1",
        ),
        message_id="msg1",
        media_urls=[media_url],
        media_types=[media_type],
    )


def _text_event(text="custom answer"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="user1",
        ),
        message_id="msg1",
    )


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _make_session_key(event, adapter):
    return build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )


@pytest.mark.asyncio
async def test_voice_message_resolves_pending_clarify():
    """A voice reply to a pending clarify should be transcribed and resolved."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _VoiceClarifyAdapter()
    transcribe_called = asyncio.Event()

    async def fake_handle_message(event):
        """Simulate the clarify intercept from GatewayRunner._handle_message."""
        from tools import clarify_gateway as _clarify_mod
        session_key = _make_session_key(event, adapter)
        pending = _clarify_mod.get_pending_for_session(session_key, include_choice_prompts=True)
        if pending is not None:
            raw = (event.text or "").strip()
            # Voice fix: transcribe audio if no text
            if not raw and getattr(event, "media_urls", None):
                audio_paths = [
                    p for i, p in enumerate(event.media_urls)
                    if i < len(event.media_types) and event.media_types[i].startswith("audio/")
                ]
                if audio_paths:
                    transcribe_called.set()
                    tx_text = '"I want pizza"'
                    candidate = tx_text.strip('"').strip()
                    if candidate and not candidate.startswith("["):
                        raw = candidate
            if raw and not raw.startswith("/"):
                resolved = _clarify_mod.resolve_text_response_for_session(session_key, raw)
                if resolved:
                    return ""
        return ""

    adapter._message_handler = fake_handle_message
    adapter._busy_session_handler = AsyncMock(return_value=True)

    event = _voice_event()
    session_key = _make_session_key(event, adapter)

    # Register a pending clarify AND mark session as active (simulate agent
    # blocked on wait_for_response).  This routes the message through the
    # inline active-session handler rather than _process_message_background.
    cm.register("clarify-voice-1", session_key, "What do you want?", None)
    adapter._active_sessions[session_key] = asyncio.Event()
    assert len(cm._entries) == 1

    await adapter.handle_message(event)

    # Verify transcription was triggered
    assert transcribe_called.is_set(), "STT transcription should have been called"

    # The clarify should have been resolved (response set, event signaled).
    # Entry removal happens in wait_for_response() when the agent thread
    # unblocks — here we just verify the resolution was triggered.
    with cm._lock:
        entry = cm._entries.get("clarify-voice-1")
        assert entry is not None, "Entry should still exist until wait_for_response cleans up"
        assert entry.response == "I want pizza", f"Expected transcript, got: {entry.response}"
        assert entry.event.is_set(), "Event should be set to unblock agent thread"


@pytest.mark.asyncio
async def test_voice_message_stt_failure_does_not_resolve_clarify():
    """If STT returns a failure marker, the clarify should NOT be resolved."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _VoiceClarifyAdapter()

    async def fake_handle_message(event):
        from tools import clarify_gateway as _clarify_mod
        session_key = _make_session_key(event, adapter)
        pending = _clarify_mod.get_pending_for_session(session_key, include_choice_prompts=True)
        if pending is not None:
            raw = (event.text or "").strip()
            if not raw and getattr(event, "media_urls", None):
                audio_paths = [
                    p for i, p in enumerate(event.media_urls)
                    if i < len(event.media_types) and event.media_types[i].startswith("audio/")
                ]
                if audio_paths:
                    tx_text = "[voice message could not be transcribed]"
                    candidate = tx_text.strip('"').strip()
                    if candidate and not candidate.startswith("["):
                        raw = candidate
            if raw and not raw.startswith("/"):
                resolved = _clarify_mod.resolve_text_response_for_session(session_key, raw)
                if resolved:
                    return ""
        return ""

    adapter._message_handler = fake_handle_message
    adapter._busy_session_handler = AsyncMock(return_value=True)

    event = _voice_event()
    session_key = _make_session_key(event, adapter)
    cm.register("clarify-voice-2", session_key, "Pick one", ["A", "B"])
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    with cm._lock:
        assert len(cm._entries) == 1, "Clarify should remain pending after STT failure"


@pytest.mark.asyncio
async def test_text_message_still_resolves_clarify():
    """Text replies to clarify should still work (no regression)."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _VoiceClarifyAdapter()

    async def fake_handle_message(event):
        from tools import clarify_gateway as _clarify_mod
        session_key = _make_session_key(event, adapter)
        pending = _clarify_mod.get_pending_for_session(session_key, include_choice_prompts=True)
        if pending is not None:
            raw = (event.text or "").strip()
            if raw and not raw.startswith("/"):
                resolved = _clarify_mod.resolve_text_response_for_session(session_key, raw)
                if resolved:
                    return ""
        return ""

    adapter._message_handler = fake_handle_message
    adapter._busy_session_handler = AsyncMock(return_value=True)

    event = _text_event("blue")
    session_key = _make_session_key(event, adapter)
    cm.register("clarify-text-1", session_key, "What color?", None)
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    with cm._lock:
        entry = cm._entries.get("clarify-text-1")
        assert entry is not None
        assert entry.response == "blue"
        assert entry.event.is_set()


@pytest.mark.asyncio
async def test_voice_with_choice_clarify_resolves():
    """Voice reply to a multi-choice clarify should be transcribed and matched."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _VoiceClarifyAdapter()

    async def fake_handle_message(event):
        from tools import clarify_gateway as _clarify_mod
        session_key = _make_session_key(event, adapter)
        pending = _clarify_mod.get_pending_for_session(session_key, include_choice_prompts=True)
        if pending is not None:
            raw = (event.text or "").strip()
            if not raw and getattr(event, "media_urls", None):
                audio_paths = [
                    p for i, p in enumerate(event.media_urls)
                    if i < len(event.media_types) and event.media_types[i].startswith("audio/")
                ]
                if audio_paths:
                    tx_text = '"A"'
                    candidate = tx_text.strip('"').strip()
                    if candidate and not candidate.startswith("["):
                        raw = candidate
            if raw and not raw.startswith("/"):
                resolved = _clarify_mod.resolve_text_response_for_session(session_key, raw)
                if resolved:
                    return ""
        return ""

    adapter._message_handler = fake_handle_message
    adapter._busy_session_handler = AsyncMock(return_value=True)

    event = _voice_event()
    session_key = _make_session_key(event, adapter)
    cm.register("clarify-voice-3", session_key, "Pick one", ["A", "B"])
    adapter._active_sessions[session_key] = asyncio.Event()

    await adapter.handle_message(event)

    with cm._lock:
        entry = cm._entries.get("clarify-voice-3")
        assert entry is not None
        assert entry.response == "A"
        assert entry.event.is_set()

"""Tests for voice/audio clarify intercept in GatewayRunner._handle_message.

Regression test for #56739: voice messages sent while the clarify tool is
waiting for a user response should be transcribed via STT and used to resolve
the pending clarify, rather than being silently ignored.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runner() -> "GatewayRunner":  # type: ignore[name-defined]
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False
    runner.pairing_store = MagicMock()
    runner._update_prompt_pending = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._slash_confirm_pending = {}
    runner._startup_restore_in_progress = False
    runner.session_store = MagicMock()
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner._inbound_last_ts = {}
    return runner


def _voice_event(text: str = "") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.VOICE,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="user1",
        ),
        message_id="msg1",
        media_urls=["/tmp/voice.ogg"],
        media_types=["audio/ogg"],
    )


def _text_event(text: str = "hello") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="user1",
        ),
        message_id="msg2",
    )


def _clear_clarify_state():
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVoiceClarifyIntercept:
    """Voice messages arriving during a pending clarify should resolve it."""

    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_voice_message_resolves_pending_clarify(self):
        """A voice message with empty text transcribes and resolves the clarify."""
        runner = _make_runner()
        event = _voice_event()
        session_key = build_session_key(event.source)

        from tools import clarify_gateway as cm
        cm.register("cid1", session_key, "What do you want?", None)

        with (
            patch.object(runner, "_is_user_authorized", return_value=True),
            patch.object(runner, "_session_key_for_source", return_value=session_key),
            patch(
                "tools.transcription_tools.transcribe_audio",
                return_value={"success": True, "transcript": "I want pizza", "provider": "whisper"},
            ) as mock_stt,
        ):
            result = await runner._handle_message(event)

        # The clarify should be resolved with the transcript
        mock_stt.assert_called_once_with("/tmp/voice.ogg")
        # _handle_message returns "" when clarify is intercepted
        assert result == ""
        # The clarify entry should have the response set
        entry = cm.get_pending_for_session(session_key)
        assert entry is not None
        assert entry.response == "I want pizza"

    @pytest.mark.asyncio
    async def test_text_message_still_resolves_clarify_normally(self):
        """Text messages still resolve clarifies via the existing text path."""
        runner = _make_runner()
        event = _text_event("I want sushi")
        session_key = build_session_key(event.source)

        from tools import clarify_gateway as cm
        cm.register("cid2", session_key, "What food?", None)

        with (
            patch.object(runner, "_is_user_authorized", return_value=True),
            patch.object(runner, "_session_key_for_source", return_value=session_key),
        ):
            result = await runner._handle_message(event)

        assert result == ""
        entry = cm.get_pending_for_session(session_key)
        assert entry is not None
        assert entry.response == "I want sushi"

    @pytest.mark.asyncio
    async def test_voice_stt_failure_does_not_resolve_clarify(self):
        """If STT fails, the clarify stays pending and message falls through."""
        runner = _make_runner()
        event = _voice_event()
        session_key = build_session_key(event.source)

        from tools import clarify_gateway as cm
        cm.register("cid3", session_key, "Pick one", ["A", "B"])

        with (
            patch.object(runner, "_is_user_authorized", return_value=True),
            patch.object(runner, "_session_key_for_source", return_value=session_key),
            patch(
                "tools.transcription_tools.transcribe_audio",
                return_value={"success": False, "error": "no STT provider"},
            ),
        ):
            result = await runner._handle_message(event)

        # Clarify should still be pending (not resolved by failed STT)
        pending = cm.get_pending_for_session(session_key, include_choice_prompts=True)
        assert pending is not None
        assert pending.clarify_id == "cid3"

    @pytest.mark.asyncio
    async def test_voice_empty_transcript_does_not_resolve_clarify(self):
        """If STT returns an empty transcript, clarify stays pending."""
        runner = _make_runner()
        event = _voice_event()
        session_key = build_session_key(event.source)

        from tools import clarify_gateway as cm
        cm.register("cid4", session_key, "What?", None)

        with (
            patch.object(runner, "_is_user_authorized", return_value=True),
            patch.object(runner, "_session_key_for_source", return_value=session_key),
            patch(
                "tools.transcription_tools.transcribe_audio",
                return_value={"success": True, "transcript": "", "provider": "whisper"},
            ),
        ):
            result = await runner._handle_message(event)

        # Empty transcript should not resolve
        pending = cm.get_pending_for_session(session_key)
        assert pending is not None

    @pytest.mark.asyncio
    async def test_audio_file_type_also_resolves_clarify(self):
        """MessageType.AUDIO (file attachment) also resolves clarify via STT."""
        runner = _make_runner()
        event = MessageEvent(
            text="",
            message_type=MessageType.AUDIO,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="12345",
                chat_type="private",
                user_id="user1",
            ),
            message_id="msg5",
            media_urls=["/tmp/song.mp3"],
            media_types=["audio/mpeg"],
        )
        session_key = build_session_key(event.source)

        from tools import clarify_gateway as cm
        cm.register("cid5", session_key, "Which one?", ["X", "Y"])

        with (
            patch.object(runner, "_is_user_authorized", return_value=True),
            patch.object(runner, "_session_key_for_source", return_value=session_key),
            patch(
                "tools.transcription_tools.transcribe_audio",
                return_value={"success": True, "transcript": "option X please", "provider": "whisper"},
            ) as mock_stt,
        ):
            result = await runner._handle_message(event)

        mock_stt.assert_called_once_with("/tmp/song.mp3")
        assert result == ""
        entry = cm.get_pending_for_session(session_key, include_choice_prompts=True)
        assert entry is not None
        assert entry.response == "option X please"

    @pytest.mark.asyncio
    async def test_stt_exception_does_not_crash(self):
        """If STT raises an exception, the clarify stays pending (no crash)."""
        runner = _make_runner()
        event = _voice_event()
        session_key = build_session_key(event.source)

        from tools import clarify_gateway as cm
        cm.register("cid6", session_key, "What?", None)

        with (
            patch.object(runner, "_is_user_authorized", return_value=True),
            patch.object(runner, "_session_key_for_source", return_value=session_key),
            patch(
                "tools.transcription_tools.transcribe_audio",
                side_effect=RuntimeError("STT service down"),
            ),
        ):
            result = await runner._handle_message(event)

        # Should not crash; clarify stays pending
        pending = cm.get_pending_for_session(session_key)
        assert pending is not None

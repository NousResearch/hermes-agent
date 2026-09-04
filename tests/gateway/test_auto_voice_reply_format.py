"""Tests for gateway auto-TTS voice reply audio format selection."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class TestAutoVoiceReplyFormat:


    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "platform",
        [Platform.MATRIX, Platform.FEISHU, Platform.WHATSAPP, Platform.SIGNAL],
    )
    async def test_opus_platform_auto_voice_reply_requests_ogg(self, platform):
        """Every OPUS_VOICE_PLATFORMS member gets an explicit .ogg output path.

        Regression for #14841 (Matrix) / #45557 (Feishu): _send_voice_reply
        hardcoded .ogg for Telegram only, so Matrix/Feishu voice replies were
        synthesized as MP3 and delivered as plain attachments instead of
        native voice bubbles.
        """
        runner = _make_runner()
        adapter = _make_adapter(platform)
        runner.adapters[platform] = adapter
        event = _make_event(platform)
        requested_paths = []

        def fake_tts(*, text, output_path):
            requested_paths.append(output_path)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fake ogg opus")
            return json.dumps({
                "success": True,
                "file_path": output_path,
                "provider": "gemini",
                "voice_compatible": True,
            })

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=fake_tts):
            await runner._send_voice_reply(event, "hello from auto tts")

        assert requested_paths and requested_paths[0].endswith(".ogg")
        adapter.send_voice.assert_awaited_once()
        assert adapter.send_voice.await_args.kwargs["audio_path"].endswith(".ogg")

    def test_should_send_voice_reply_streamed_global_auto_tts_fires(self):
        """Streamed reply + global voice.auto_tts (no /voice opt-in) sends voice.

        Regression for the #51867/#23983 remainder: when streaming consumed
        the text, the base adapter's auto-TTS gets text_content=None, and the
        runner path used to consult only self._voice_mode — so a chat relying
        purely on the global voice.auto_tts default silently lost its voice
        reply.
        """
        runner = _make_runner()
        adapter = _make_adapter(Platform.TELEGRAM)
        adapter._should_auto_tts_for_chat = MagicMock(return_value=True)
        runner.adapters[Platform.TELEGRAM] = adapter
        voice_event = _make_event(
            Platform.TELEGRAM, chat_id="123", message_type=MessageType.VOICE
        )

        assert runner._should_send_voice_reply(
            voice_event, "hello", [], already_sent=True
        ) is True

    def test_should_send_voice_reply_voice_only_still_requires_voice_input(self):
        """Explicit voice_only must not widen to text input (#73508 regression).

        Persisted voice_only mode is synced into the adapter as an explicit
        auto-TTS opt-in, so adapter_auto_tts is True for this chat. The
        chat-level mode stays authoritative: text input gets no voice reply,
        voice input still does.
        """
        runner = _make_runner()
        runner._voice_mode["telegram:123"] = "voice_only"
        adapter = _make_adapter(Platform.TELEGRAM)
        adapter._should_auto_tts_for_chat = MagicMock(return_value=True)
        runner.adapters[Platform.TELEGRAM] = adapter
        event = _make_event(Platform.TELEGRAM, chat_id="123")

        assert runner._should_send_voice_reply(event, "hello", []) is False

        voice_event = _make_event(Platform.TELEGRAM, chat_id="123", message_type=MessageType.VOICE)
        assert runner._should_send_voice_reply(voice_event, "hello", [], already_sent=True) is True

    def test_voice_only_catchup_text_keeps_voice_reply_after_lease(self):
        """Catch-up TEXT after a voice inbound must still answer in voice (#94660).

        Lease contention replaces the original VOICE MessageEvent with a
        synthetic catch-up/resume turn. voice_only must not treat that as
        text-in → text-out.
        """
        runner = _make_runner()
        chat_id = "15551234567"
        runner._voice_mode[f"whatsapp:{chat_id}"] = "voice_only"
        adapter = _make_adapter(Platform.WHATSAPP)
        adapter._should_auto_tts_for_chat = MagicMock(return_value=True)
        runner.adapters[Platform.WHATSAPP] = adapter

        voice = _make_event(Platform.WHATSAPP, chat_id=chat_id, message_type=MessageType.VOICE)
        session_key = runner._session_key_for_source(voice.source)
        runner._remember_voice_inbound(session_key, voice)

        catchup = _make_event(Platform.WHATSAPP, chat_id=chat_id, message_type=MessageType.TEXT)
        catchup.text = (
            "[System note: A new message has arrived. The conversation "
            "history contains pending tool outputs from an interrupted turn. "
            "IGNORE those pending results. Address the user's NEW message "
            "below FIRST. Do NOT re-execute old tool calls from the history.]\n\n"
            "transcribed voice: hello"
        )
        catchup.internal = True
        runner._restore_voice_type_on_catchup(session_key, catchup)

        assert catchup.message_type == MessageType.VOICE
        assert runner._should_send_voice_reply(catchup, "hello", [], already_sent=True) is True

        later_text = _make_event(Platform.WHATSAPP, chat_id=chat_id, message_type=MessageType.TEXT)
        later_text.text = "actually typed this"
        runner._remember_voice_inbound(session_key, later_text)
        assert runner._should_send_voice_reply(later_text, "hello", []) is False

    def test_resume_pending_synthetic_inherits_voice_stamp(self):
        runner = _make_runner()
        chat_id = "15551234567"
        voice = _make_event(Platform.WHATSAPP, chat_id=chat_id, message_type=MessageType.VOICE)
        session_key = runner._session_key_for_source(voice.source)
        runner._remember_voice_inbound(session_key, voice)

        resume = _make_event(Platform.WHATSAPP, chat_id=chat_id, message_type=MessageType.TEXT)
        resume.text = ""
        resume.internal = True
        runner._restore_voice_type_on_catchup(session_key, resume)
        assert resume.message_type == MessageType.VOICE

        heartbeat = _make_event(Platform.WHATSAPP, chat_id=chat_id, message_type=MessageType.TEXT)
        heartbeat.text = "heartbeat tick"
        heartbeat.internal = True
        runner._restore_voice_type_on_catchup(session_key, heartbeat)
        assert heartbeat.message_type == MessageType.TEXT
        assert runner._should_send_voice_reply(heartbeat, "hello", []) is False

def _make_runner() -> GatewayRunner:
    with patch("gateway.run.GatewayRunner._load_voice_modes", return_value={}):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._voice_mode = {}
        runner._pending_voice_reply_sessions = set()
        runner.adapters = {}
        runner.session_store = None
    return runner


def _make_adapter(platform: Platform) -> MagicMock:
    adapter = MagicMock()
    adapter.platform = platform
    adapter.send_voice = AsyncMock()
    return adapter


def _make_event(platform: Platform, chat_id: str = "123", message_type: MessageType = MessageType.TEXT) -> MessageEvent:
    return MessageEvent(
        text="trigger",
        source=SessionSource(
            platform=platform,
            chat_id=chat_id,
            user_id="u1",
            user_name="User",
        ),
        message_type=message_type,
        message_id="456",
    )

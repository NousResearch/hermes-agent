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
    async def test_telegram_auto_voice_reply_requests_ogg_for_native_voice_bubble(self):
        """Telegram auto-TTS should request OGG/Opus so send_voice sends a voice bubble."""
        runner = _make_runner()
        adapter = _make_adapter(Platform.TELEGRAM)
        runner.adapters[Platform.TELEGRAM] = adapter
        event = _make_event(Platform.TELEGRAM)
        requested_paths = []

        def fake_tts(*, text, output_path):
            requested_paths.append(output_path)
            assert output_path.endswith(".ogg")
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

        assert requested_paths
        assert requested_paths[0].endswith(".ogg")
        adapter.send_voice.assert_awaited_once()
        assert adapter.send_voice.await_args.kwargs["audio_path"].endswith(".ogg")

    @pytest.mark.asyncio
    async def test_non_telegram_auto_voice_reply_keeps_mp3_default(self):
        """Non-Telegram platforms should keep the current MP3 default."""
        runner = _make_runner()
        adapter = _make_adapter(Platform.SLACK)
        runner.adapters[Platform.SLACK] = adapter
        event = _make_event(Platform.SLACK)
        requested_paths = []

        def fake_tts(*, text, output_path):
            requested_paths.append(output_path)
            assert output_path.endswith(".mp3")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fake mp3")
            return json.dumps({
                "success": True,
                "file_path": output_path,
                "provider": "gemini",
                "voice_compatible": False,
            })

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=fake_tts):
            await runner._send_voice_reply(event, "hello from auto tts")

        assert requested_paths
        assert requested_paths[0].endswith(".mp3")
        adapter.send_voice.assert_awaited_once()
        assert adapter.send_voice.await_args.kwargs["audio_path"].endswith(".mp3")
    def test_should_send_voice_reply_uses_global_auto_tts_adapter_default(self):
        """voice.auto_tts=true should make normal text replies get voice too."""
        runner = _make_runner()
        adapter = _make_adapter(Platform.TELEGRAM)
        adapter._should_auto_tts_for_chat = MagicMock(return_value=True)
        runner.adapters[Platform.TELEGRAM] = adapter
        event = _make_event(Platform.TELEGRAM, chat_id="123")

        assert runner._should_send_voice_reply(event, "hello", []) is True
        adapter._should_auto_tts_for_chat.assert_called_once_with("123")

    def test_should_send_voice_reply_honors_explicit_voice_off_over_global_auto_tts(self):
        """A chat-level /voice off remains a hard override."""
        runner = _make_runner()
        runner._voice_mode["telegram:123"] = "off"
        adapter = _make_adapter(Platform.TELEGRAM)
        adapter._should_auto_tts_for_chat = MagicMock(return_value=True)
        runner.adapters[Platform.TELEGRAM] = adapter
        event = _make_event(Platform.TELEGRAM, chat_id="123")

        assert runner._should_send_voice_reply(event, "hello", []) is False

    def test_should_send_voice_reply_voice_only_still_requires_voice_input(self):
        runner = _make_runner()
        runner._voice_mode["telegram:123"] = "voice_only"
        event = _make_event(Platform.TELEGRAM, chat_id="123")

        assert runner._should_send_voice_reply(event, "hello", []) is False

        voice_event = _make_event(Platform.TELEGRAM, chat_id="123", message_type=MessageType.VOICE)
        assert runner._should_send_voice_reply(voice_event, "hello", [], already_sent=True) is True


def _make_runner() -> GatewayRunner:
    with patch("gateway.run.GatewayRunner._load_voice_modes", return_value={}):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner._voice_mode = {}
        runner.adapters = {}
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

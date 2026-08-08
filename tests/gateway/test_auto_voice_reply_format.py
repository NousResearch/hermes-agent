"""Tests for gateway auto-TTS voice reply audio format selection."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
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

    @pytest.mark.asyncio
    async def test_discord_synthetic_auto_voice_reply_uses_live_text_channel_binding(self):
        """A synthetic outer event must not turn live VC TTS into an attachment.

        An internal completion can be interrupted by a Discord voice turn. The
        runner drains that pending voice turn recursively, then routes its reply
        through the original internal event, which has no raw guild metadata.
        The current Discord adapter still owns the text-channel-to-VC binding.
        """
        from plugins.platforms.discord.adapter import DiscordAdapter

        runner = _make_runner()
        adapter = DiscordAdapter(
            PlatformConfig(enabled=True, token="fake-token")
        )
        runner.adapters[Platform.DISCORD] = adapter

        guild_id = 111
        chat_id = "222"
        voice_client = MagicMock()
        voice_client.is_connected.return_value = True
        adapter._voice_clients[guild_id] = voice_client
        adapter._voice_text_channels[guild_id] = int(chat_id)
        adapter.play_in_voice_channel = AsyncMock(return_value=True)
        adapter.send_voice = AsyncMock()

        # Models the original internal completion event retained by the outer
        # delivery frame after the pending voice turn is recursively drained.
        event = _make_event(
            Platform.DISCORD,
            chat_id=chat_id,
            message_type=MessageType.TEXT,
        )
        event.internal = True
        event.raw_message = None

        def fake_tts(*, text, output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fake mp3")
            return json.dumps({
                "success": True,
                "file_path": output_path,
                "provider": "test",
            })

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=fake_tts):
            await runner._send_voice_reply(event, "voice follow-up response")

        adapter.send_voice.assert_not_awaited()
        adapter.play_in_voice_channel.assert_awaited_once()
        playback_call = adapter.play_in_voice_channel.await_args
        assert playback_call is not None
        assert playback_call.args[0] == guild_id
        assert Path(playback_call.args[1]).suffix == ".mp3"


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

"""Tests for gateway auto-TTS voice reply audio format selection."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
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

        def fake_tts(*, text, output_path, request_timeout_s=None):
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

    def test_should_send_voice_reply_skips_channels_that_cannot_deliver_voice(self):
        """A channel whose adapter cannot carry audio gets no auto-TTS at all.

        Regression for the email channel: the email adapter never overrides
        ``send_voice``, so it inherits the base adapter's warning-only fallback.
        Auto-TTS therefore synthesized audio that ``_deliver_voice_reply`` threw
        away — while the FINAL TEXT reply waited behind it. Against a slow TTS
        backend that is the TTS client's whole retry budget (~30 min), which is
        how every email answer ended up arriving half an hour late.
        """
        runner = _make_runner()
        adapter = _NoVoiceAdapter.__new__(_NoVoiceAdapter)
        adapter.platform = Platform.EMAIL
        adapter._auto_tts_default = True
        adapter._auto_tts_enabled_chats = set()
        adapter._auto_tts_disabled_chats = set()
        runner.adapters[Platform.EMAIL] = adapter
        event = _make_event(Platform.EMAIL, chat_id="nca@fastmail.com")

        assert adapter._should_auto_tts_for_chat("nca@fastmail.com") is True
        assert runner._should_send_voice_reply(event, "hello", []) is False

        # The guard is structural, not a mode: an explicit /voice all cannot re-enable it either.
        runner._voice_mode["email:nca@fastmail.com"] = "all"
        assert runner._should_send_voice_reply(event, "hello", []) is False


class _NoVoiceAdapter(BasePlatformAdapter):
    """Stands in for the email adapter: it does not override ``send_voice``.

    Only the abstract methods are filled in; the instance is built with
    ``__new__`` and the three auto-TTS attributes the guard reads.
    """

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def get_chat_info(self, chat_id):
        return {}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return None


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

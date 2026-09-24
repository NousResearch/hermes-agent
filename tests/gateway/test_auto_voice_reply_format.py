"""Tests for gateway auto-TTS voice reply audio format selection."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
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

    @pytest.mark.asyncio
    async def test_wav_native_provider_auto_voice_reply_is_real_opus(self, tmp_path, monkeypatch):
        """A WAV-native provider's auto voice reply must hold Ogg/OPUS, not just be named .ogg.

        Regression (Telegram replies arriving as audio attachments instead of voice bubbles):
        the WAV sidecar was converted with a bare ``ffmpeg -i sidecar.wav out.ogg``. ffmpeg's
        ``.ogg`` muxer default codec is Vorbis and a libvorbis-less build silently writes
        Ogg/FLAC — real Ogg, but not Opus, which is what a voice note is expected to carry.
        Requesting an ``.ogg`` path (which this path already did, and which the test above
        asserts) is not the same as producing Opus.
        """
        import shutil
        import subprocess
        import wave

        from tools import tts_tool
        from tools.tts_tool_delivery import _finalize_wav_output, _sniff_ogg_codec, _wav_sidecar_path

        ffmpeg = shutil.which("ffmpeg")
        encoders = subprocess.run([ffmpeg, "-hide_banner", "-h", "encoder=libopus"],
                                  capture_output=True, check=False).stdout if ffmpeg else b""
        if not ffmpeg or b"libopus" not in encoders:
            pytest.skip("ffmpeg without libopus cannot produce voice-bubble audio")

        def fake_piper(text, output_path, tts_config):
            wav_path = _wav_sidecar_path(output_path)
            with wave.open(wav_path, "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(22050)
                fh.writeframes(b"\x00\x00" * 4410)
            return _finalize_wav_output(wav_path, output_path)

        monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "piper"})
        monkeypatch.setattr(tts_tool, "_generate_piper_tts", fake_piper)

        runner = _make_runner()
        adapter = _make_adapter(Platform.TELEGRAM)
        captured: dict = {}

        async def _capture(chat_id=None, audio_path=None, **kwargs):
            # _send_voice_reply unlinks the audio in its finally block, so inspect it in-flight.
            captured["path"] = audio_path
            captured["codec"] = _sniff_ogg_codec(audio_path)

        adapter.send_voice = AsyncMock(side_effect=_capture)
        runner.adapters[Platform.TELEGRAM] = adapter

        await runner._send_voice_reply(_make_event(Platform.TELEGRAM), "hello from auto tts")

        assert captured["path"].endswith(".ogg")
        assert captured["codec"] == "opus"

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

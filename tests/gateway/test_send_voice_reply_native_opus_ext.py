"""Regression test: ``GatewayRunner._send_voice_reply`` must pick an
extension that matches the configured TTS provider's native output.

Providers in the native-Opus set ({openai, elevenlabs, mistral, gemini,
inworld}) honor the supplied output path's extension. Passing ``.mp3`` (the
old hardcoded value) makes them produce MP3 bytes, which downstream
``adapter.send_voice`` then routes through Telegram's ``sendAudio`` (audio-
file card) instead of ``sendVoice`` (waveform bubble). Picking ``.ogg``
for those providers restores native voice-note rendering.

Other providers (edge, neutts, etc.) still need ``.mp3`` / ``.wav`` and get
converted to ``.opus`` downstream by ``_convert_to_opus``.
"""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_event():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="208214988",
        user_id="208214988",
        chat_type="dm",
    )
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )


def _runner_with_adapter(send_voice_mock):
    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(
        send_voice=send_voice_mock,
        is_in_voice_channel=lambda *_a, **_k: False,
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner


def _patch_tts_to_capture_path(monkeypatch, recorder: list):
    """Patch the TTS tool to record the output_path it was handed."""

    def _fake_text_to_speech_tool(*, text, output_path, **_kwargs):
        recorder.append(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(b"\x00" * 32)
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr(
        "tools.tts_tool.text_to_speech_tool",
        _fake_text_to_speech_tool,
    )
    monkeypatch.setattr(
        "tools.tts_tool._strip_markdown_for_tts",
        lambda text: text,
    )


@pytest.mark.parametrize(
    "provider",
    ["openai", "elevenlabs", "mistral", "gemini", "inworld"],
)
@pytest.mark.asyncio
async def test_voice_reply_picks_ogg_for_native_opus_providers(
    monkeypatch, tmp_path, provider
):
    """Native-Opus providers must receive a ``.ogg`` output path so the
    Telegram adapter routes the file through ``sendVoice`` (waveform bubble)
    instead of ``sendAudio`` (audio-file card)."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {"provider": provider})
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda _cfg: provider)
    paths: list = []
    _patch_tts_to_capture_path(monkeypatch, paths)

    send_voice = AsyncMock()
    runner = _runner_with_adapter(send_voice)
    event = _make_event()

    await runner._send_voice_reply(event, "Hello there.")

    assert len(paths) == 1, "TTS tool should have been called exactly once"
    assert paths[0].endswith(".ogg"), (
        f"Expected .ogg path for native-Opus provider {provider!r}, got {paths[0]!r}"
    )


@pytest.mark.parametrize("provider", ["edge", "neutts", "kittentts", "piper", "xai"])
@pytest.mark.asyncio
async def test_voice_reply_keeps_mp3_for_non_native_opus_providers(
    monkeypatch, tmp_path, provider
):
    """Non-native-Opus providers still get ``.mp3`` so the existing
    ``_convert_to_opus`` step (Edge TTS et al.) keeps working."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr("tools.tts_tool._load_tts_config", lambda: {"provider": provider})
    monkeypatch.setattr("tools.tts_tool._get_provider", lambda _cfg: provider)
    paths: list = []
    _patch_tts_to_capture_path(monkeypatch, paths)

    send_voice = AsyncMock()
    runner = _runner_with_adapter(send_voice)
    event = _make_event()

    await runner._send_voice_reply(event, "Hello there.")

    assert len(paths) == 1
    assert paths[0].endswith(".mp3"), (
        f"Expected .mp3 path for non-native-Opus provider {provider!r}, got {paths[0]!r}"
    )

"""
Tests for #24870 — Telegram: audio file attachments must NOT be routed to STT.

Telegram distinguishes three kinds of audio payloads:
  - message.voice  → Opus/OGG voice message  → STT pipeline
  - message.audio  → audio file attachment   → file path note, NOT STT
  - message.document (audio mime) → generic file route

These tests confirm that:
  1. MessageType.VOICE events still flow through the STT pipeline.
  2. MessageType.AUDIO events bypass STT and get a file-path context note instead.
  3. Mixed media lists (voice + audio) split correctly.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import hermes_yaml as yaml

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(stt_enabled: bool = True) -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=stt_enabled)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False
    return runner


def _voice_event(path: str = "/tmp/voice.ogg") -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
        media_urls=[path],
        media_types=["audio/ogg"],
    )


def _audio_event(path: str = "/tmp/song.mp3") -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=MessageType.AUDIO,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm"),
        media_urls=[path],
        media_types=["audio/mpeg"],
    )


# ---------------------------------------------------------------------------
# 1. VOICE still goes through STT
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_voice_message_still_transcribed():
    """MessageType.VOICE must still be sent through _enrich_message_with_transcription."""
    runner = _make_runner(stt_enabled=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm")
    event = _voice_event("/tmp/voice.ogg")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "hello world", "provider": "whisper"},
    ) as mock_transcribe:
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    mock_transcribe.assert_called_once_with("/tmp/voice.ogg", None, "gateway")
    # The transcript passes through as a plain quoted line — no "voice message"
    # meta-commentary in the LLM-visible prompt.
    assert "hello world" in result


# ---------------------------------------------------------------------------
# 2. AUDIO file attachment bypasses STT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_attachment_context_note_format():
    """Context note for audio file attachments should include the file path and guidance."""
    runner = _make_runner(stt_enabled=True)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm")
    event = _audio_event("/tmp/cache_12345_my_song.mp3")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=AssertionError("must not be called"),
    ):
        with patch(
            "tools.credential_files.to_agent_visible_cache_path",
            side_effect=lambda p: p,
        ):
            result = await runner._prepare_inbound_message_text(
                event=event,
                source=source,
                history=[],
            )

    # STT must not run (transcribe_audio raises); the agent sees the file path.
    assert "my_song.mp3" in result


@pytest.mark.asyncio
async def test_opted_in_audio_file_is_transcribed_and_keeps_its_path(tmp_path, monkeypatch):
    """``stt.transcribe_audio_files: true`` sends an audio file to STT; the path note stays (#24870)."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"stt": {"transcribe_audio_files": True}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    runner = _make_runner()
    runner.config = load_gateway_config()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="1", chat_type="dm")
    event = _audio_event("/tmp/cache_12345_forwarded_note.m4a")

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "see you at noon", "provider": "whisper"},
    ) as mock_transcribe, patch(
        "tools.credential_files.to_agent_visible_cache_path",
        side_effect=lambda p: p,
    ):
        result = await runner._prepare_inbound_message_text(event=event, source=source, history=[])

    mock_transcribe.assert_called_once_with("/tmp/cache_12345_forwarded_note.m4a", None, "gateway")
    assert "see you at noon" in result
    assert "forwarded_note.m4a" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_opt_in", [True, False])
async def test_audio_file_opt_in_follows_the_receiving_profile(primary_opt_in):
    """Receiving-profile policy controls actual inbound STT; unloaded profiles fail closed."""
    runner = _make_runner()
    runner.config = GatewayConfig.from_dict({"stt": {"transcribe_audio_files": primary_opt_in}})
    runner._primary_profile_name = "default"
    runner._profile_configs = {
        "family": GatewayConfig.from_dict({"stt": {"transcribe_audio_files": not primary_opt_in}})
    }
    for profile, expected in [(None, primary_opt_in), ("family", not primary_opt_in), ("unloaded", False)]:
        event = _audio_event()
        event.source.profile = profile
        with patch(
            "tools.transcription_tools.transcribe_audio",
            return_value={"success": True, "transcript": "see you at noon", "provider": "whisper"},
        ) as transcribe, patch("tools.credential_files.to_agent_visible_cache_path", side_effect=lambda p: p):
            result = await runner._prepare_inbound_message_text(event=event, source=event.source, history=[])
        assert transcribe.call_count == int(expected)
        assert ("see you at noon" in result) is expected
        assert event.media_urls[0] in result


# ---------------------------------------------------------------------------
# 3. STT disabled still results in no transcription for audio file attachments
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Telegram gateway: msg.audio → MessageType.AUDIO (not VOICE)
# ---------------------------------------------------------------------------


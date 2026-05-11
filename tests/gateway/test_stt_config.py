"""Gateway STT config tests — honor stt.enabled: false from config.yaml."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def test_gateway_config_stt_disabled_from_dict_nested():
    config = GatewayConfig.from_dict({"stt": {"enabled": False}})
    assert config.stt_enabled is False


def test_gateway_config_reads_stt_echo_transcript_options():
    config = GatewayConfig.from_dict(
        {
            "stt": {
                "enabled": True,
                "echo_transcript": True,
                "echo_transcript_prefix": "🎤 Transcription STT",
            }
        }
    )

    assert config.stt_echo_transcript is True
    assert config.stt_echo_transcript_prefix == "🎤 Transcription STT"


def test_load_gateway_config_bridges_stt_enabled_from_config_yaml(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.dump({"stt": {"enabled": False}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    config = load_gateway_config()

    assert config.stt_enabled is False


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_skips_when_stt_disabled():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=False)

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=AssertionError("transcribe_audio should not be called when STT is disabled"),
    ):
        result = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
        )

    assert "transcription is disabled" in result.lower()
    assert "caption" in result


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_avoids_bogus_no_provider_message_for_backend_key_errors():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": False, "error": "VOICE_TOOLS_OPENAI_KEY not set"},
    ):
        result = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
        )

    assert "No STT provider is configured" not in result
    assert "trouble transcribing" in result
    assert "caption" in result


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_transcribes_queued_voice_event():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/queued-voice.ogg"],
        media_types=["audio/ogg"],
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "queued voice transcript",
            "provider": "local_command",
        },
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "queued voice transcript" in result
    assert "voice message" in result.lower()


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_echoes_voice_transcript_when_enabled():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.config.stt_echo_transcript = True
    runner.config.stt_echo_transcript_prefix = "🎤 Transcription"
    adapter = AsyncMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        thread_id="11",
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/queued-voice.ogg"],
        media_types=["audio/ogg"],
        message_id="voice-msg-1",
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "queued voice transcript",
            "provider": "local_command",
        },
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "queued voice transcript" in result
    adapter.send.assert_awaited_once()
    args, kwargs = adapter.send.await_args
    assert args[0] == "123"
    assert "🎤 Transcription" in args[1]
    assert "queued voice transcript" in args[1]
    assert kwargs["metadata"] == {
        "thread_id": "11",
        "telegram_dm_topic_reply_fallback": True,
        "telegram_reply_to_message_id": "voice-msg-1",
    }


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_applies_personal_stt_corrections(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    from tools.stt_corrections import add_correction

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    add_correction("gétoueur du Christart", "GetHooked starter kit")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.config.stt_echo_transcript = True
    runner.config.stt_echo_transcript_prefix = "🎤 Transcription"
    adapter = AsyncMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="group")
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/queued-voice.ogg"],
        media_types=["audio/ogg"],
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "Le gétoueur du Christart est prêt",
            "provider": "local_command",
        },
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "GetHooked starter kit" in result
    assert "gétoueur du Christart" not in result
    args, _kwargs = adapter.send.await_args
    assert "GetHooked starter kit" in args[1]


@pytest.mark.asyncio
async def test_handle_stt_correct_command_adds_personal_correction(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    from tools.stt_corrections import apply_stt_corrections

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    runner = GatewayRunner.__new__(GatewayRunner)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="group")
    event = MessageEvent(
        text="/stt-correct gétoueur du Christart => GetHooked starter kit",
        source=source,
    )

    response = await runner._handle_stt_correct_command(event)

    assert "Correction STT ajoutée" in response
    assert apply_stt_corrections("gétoueur du Christart") == "GetHooked starter kit"

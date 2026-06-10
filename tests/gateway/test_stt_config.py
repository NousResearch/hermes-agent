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
async def test_enrich_message_with_transcription_surfaces_path_when_stt_disabled():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=False)
    runner._has_setup_skill = lambda: True  # Should NOT be consulted in disabled branch.

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=AssertionError("transcribe_audio should not be called when STT is disabled"),
    ), patch(
        "gateway.run._probe_audio_duration",
        new=AsyncMock(return_value="0:12"),
    ):
        result, transcripts = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
        )

    assert "/tmp/voice.ogg" in result
    assert "voice message" in result.lower()
    assert "(duration: 0:12)" in result
    assert "caption" in result
    assert transcripts == []


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_omits_duration_on_probe_failure():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=False)

    with patch(
        "gateway.run._probe_audio_duration",
        new=AsyncMock(return_value=None),
    ):
        result, transcripts = await runner._enrich_message_with_transcription(
            "",
            ["/tmp/voice.ogg"],
        )

    assert "/tmp/voice.ogg" in result
    assert "duration" not in result.lower()
    assert transcripts == []


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_avoids_bogus_no_provider_message_for_backend_key_errors():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": False, "error": "VOICE_TOOLS_OPENAI_KEY not set"},
    ):
        result, transcripts = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
        )

    assert "No STT provider is configured" not in result
    assert "[voice message could not be transcribed]" in result
    # The opaque backend cause must NOT leak into the LLM-visible prompt.
    assert "VOICE_TOOLS_OPENAI_KEY" not in result
    assert "caption" in result
    assert transcripts == []


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_returns_tuple_for_empty_content_placeholder():
    """A successful transcription whose caption is the empty-content placeholder
    must still return the ``(text, transcripts)`` tuple.

    The Discord adapter delivers a captionless voice note as the literal
    ``"(The user sent a message with no text content)"`` placeholder. When STT
    succeeds we strip that redundant placeholder and return just the transcript
    prefix — but the method's contract (and every caller, which unpacks the
    result as ``text, transcripts = ...``) requires a 2-tuple. Returning a bare
    string here raised ``ValueError: too many values to unpack`` and dropped the
    whole voice message on the floor.
    """
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "hello from a captionless voice note",
            "provider": "local_command",
        },
    ):
        result, transcripts = await runner._enrich_message_with_transcription(
            "(The user sent a message with no text content)",
            ["/tmp/voice.ogg"],
        )

    # The redundant placeholder is stripped, leaving only the transcript prefix.
    assert "hello from a captionless voice note" in result
    assert "(The user sent a message with no text content)" not in result
    # Crucially, the transcripts are still surfaced so callers can echo them.
    assert transcripts == ["hello from a captionless voice note"]


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
    # Success path: the transcript passes through as a plain quoted line, with
    # no "voice message" meta-commentary that the LLM would echo back.
    assert "queued voice transcript" in result


@pytest.mark.asyncio
async def test_observed_audio_context_transcribes_on_addressed_followup(tmp_path):
    from gateway.run import GatewayRunner

    audio_path = tmp_path / "audio_123.ogg"
    audio_path.write_bytes(b"fake ogg bytes")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)

    history = [
        {
            "role": "user",
            "observed": True,
            "content": f"[Alice|111]\n[audio 'voice.ogg' saved at: {audio_path}]",
        },
        {"role": "assistant", "content": "previous reply"},
    ]

    with patch("gateway.run._resolve_observed_audio_cache_path", return_value=str(audio_path.resolve())), patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "observed voice transcript",
            "provider": "openai",
        },
    ) as transcribe:
        enriched = await runner._enrich_observed_audio_context(
            history,
            channel_prompt="observed Telegram group context",
            current_message="[Bob|222]\nconsegue acessar esse audio?",
        )

    transcribe.assert_called_once_with(str(audio_path.resolve()))
    assert enriched[0] is not history[0]
    assert "Observed audio transcript" in enriched[0]["content"]
    assert "observed voice transcript" in enriched[0]["content"]
    assert enriched[0]["_observed_audio_transcripts"] == ["observed voice transcript"]
    assert enriched[1] is history[1]


@pytest.mark.asyncio
async def test_observed_audio_context_only_runs_for_telegram_observe_prompt(tmp_path):
    from gateway.run import GatewayRunner

    audio_path = tmp_path / "audio_123.ogg"
    audio_path.write_bytes(b"fake ogg bytes")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    history = [
        {
            "role": "user",
            "observed": True,
            "content": f"[Alice|111]\n[audio 'voice.ogg' saved at: {audio_path}]",
        },
    ]

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=AssertionError("observed audio should not be transcribed"),
    ):
        enriched = await runner._enrich_observed_audio_context(
            history,
            channel_prompt=None,
            current_message="[Bob|222]\nconsegue acessar esse audio?",
        )

    assert enriched is history


@pytest.mark.asyncio
async def test_observed_audio_context_waits_for_audio_request(tmp_path):
    from gateway.run import GatewayRunner

    audio_path = tmp_path / "audio_123.ogg"
    audio_path.write_bytes(b"fake ogg bytes")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    history = [
        {
            "role": "user",
            "observed": True,
            "content": f"[Alice|111]\n[audio 'voice.ogg' saved at: {audio_path}]",
        },
    ]

    with patch(
        "tools.transcription_tools.transcribe_audio",
        side_effect=AssertionError("observed audio should wait for an audio request"),
    ):
        enriched = await runner._enrich_observed_audio_context(
            history,
            channel_prompt="observed Telegram group context",
            current_message="[Bob|222]\nbom dia",
        )

    assert enriched is history


def test_observed_audio_cache_path_resolves_agent_visible_docker_path(tmp_path):
    from gateway.run import _resolve_observed_audio_cache_path

    host_audio_dir = tmp_path / "cache" / "audio"
    host_audio_dir.mkdir(parents=True)
    host_audio = host_audio_dir / "audio_123.ogg"
    host_audio.write_bytes(b"fake ogg bytes")

    with patch(
        "gateway.platforms.base.get_audio_cache_dir",
        return_value=host_audio_dir,
    ), patch(
        "hermes_constants.get_hermes_home",
        return_value=tmp_path,
    ), patch(
        "tools.credential_files.get_cache_directory_mounts",
        return_value=[
            {
                "host_path": str(host_audio_dir),
                "container_path": "/root/.hermes/cache/audio",
            }
        ],
    ):
        resolved = _resolve_observed_audio_cache_path(
            "/root/.hermes/cache/audio/audio_123.ogg"
        )

    assert resolved == str(host_audio.resolve())

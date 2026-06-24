"""Gateway STT config tests — honor stt.enabled: false from config.yaml."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from gateway.config import GatewayConfig, Platform, load_gateway_config
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def test_gateway_config_stt_disabled_from_dict_nested():
    config = GatewayConfig.from_dict({"stt": {"enabled": False}})
    assert config.stt_enabled is False


def test_gateway_config_stt_transcript_echo_defaults_off():
    config = GatewayConfig.from_dict({"stt": {"enabled": True}})
    assert config.stt_send_transcription is False
    assert config.stt_send_transcription_header == ""


def test_gateway_config_stt_transcript_echo_from_dict_nested():
    config = GatewayConfig.from_dict(
        {"stt": {"send_transcription": True, "send_transcription_header": "STT:\n"}}
    )
    assert config.stt_send_transcription is True
    assert config.stt_send_transcription_header == "STT:\n"


def test_load_gateway_config_bridges_stt_enabled_from_config_yaml(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.dump(
            {
                "stt": {
                    "enabled": False,
                    "send_transcription": True,
                    "send_transcription_header": "STT:\n",
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    config = load_gateway_config()

    assert config.stt_enabled is False
    assert config.stt_send_transcription is True
    assert config.stt_send_transcription_header == "STT:\n"


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
async def test_prepare_inbound_message_text_does_not_echo_voice_transcript_by_default():
    """STT transcript echo must be explicit opt-in.

    The agent still receives the transcript wrapper in context, but the gateway
    must not also send a deterministic copy before the model replies. Otherwise
    profiles that already quote voice transcripts at the top of the final answer
    show the user the same transcript twice.
    """
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False
    runner._session_key_for_source = lambda source: "telegram:dm:123"
    runner._consume_pending_native_image_paths = lambda session_key: []
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {}
    runner._reply_anchor_for_event = lambda event: None
    echo_adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.TELEGRAM: echo_adapter}  # type: ignore[dict-item]

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm")
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/voice.ogg"],
        media_types=["audio/ogg"],
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "single visible transcript",
            "provider": "local_command",
        },
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "single visible transcript" in result
    echo_adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_dequeue_pending_voice_does_not_echo_transcript_by_default():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False
    echo_adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.TELEGRAM: echo_adapter}  # type: ignore[dict-item]

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="group",
        thread_id="7",
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/queued-voice.ogg"],
        media_types=["audio/ogg"],
    )
    pending_adapter = SimpleNamespace(get_pending_message=lambda session_key: event)

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "queued transcript once",
            "provider": "local_command",
        },
    ):
        result = await runner._dequeue_pending_with_transcription(
            pending_adapter,
            "telegram:group:123:7",
            source,
        )

    assert result is not None
    assert "queued transcript once" in result
    echo_adapter.send.assert_not_called()


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_echoes_voice_transcript_when_enabled():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        stt_enabled=True,
        stt_send_transcription=True,
        stt_send_transcription_header="STT:\n",
    )
    runner._has_setup_skill = lambda: False
    runner._session_key_for_source = lambda source: "telegram:dm:123"
    runner._consume_pending_native_image_paths = lambda session_key: []
    runner._thread_metadata_for_source = lambda *_args, **_kwargs: {"thread_id": "7"}
    runner._reply_anchor_for_event = lambda event: "42"
    echo_adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.TELEGRAM: echo_adapter}  # type: ignore[dict-item]

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="group",
        thread_id="7",
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/voice.ogg"],
        media_types=["audio/ogg"],
        reply_to_message_id="42",
    )

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "visible because opt in",
            "provider": "local_command",
        },
    ):
        await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    echo_adapter.send.assert_awaited_once_with(
        "123",
        "STT:\n> 🎙 visible because opt in",
        metadata={"thread_id": "7"},
    )

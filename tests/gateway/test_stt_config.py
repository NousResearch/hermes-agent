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
async def test_enrich_message_with_transcription_guards_empty_transcript():
    """success=True with an empty/whitespace transcript must not emit empty
    quotes — it gets a sentinel note and is excluded from transcripts (#41603)."""
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": "   \n\t", "provider": "local_command"},
    ):
        result, transcripts = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
        )

    assert "empty or inaudible" in result
    assert '""' not in result
    assert transcripts == []


@pytest.mark.asyncio
async def test_enrich_message_with_transcription_surfaces_stt_fallback_warning():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner._has_setup_skill = lambda: False

    with patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={
            "success": True,
            "transcript": "fallback transcript",
            "provider": "local",
            "fallback_from": "parakeet",
            "fallback_reason": "command exited 127",
        },
    ):
        result, transcripts = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
        )

    assert "fallback transcript" in result
    assert "STT fallback" not in result
    assert "command exited 127" not in result
    assert transcripts == ["fallback transcript"]
    assert GatewayRunner._format_stt_echo(transcripts[0]) == (
        '🎙️ "fallback transcript"\n\n'
        "⚠️ STT fallback: parakeet failed, so Hermes used "
        "local / faster-whisper."
    )


def test_format_stt_echo_includes_fallback_notice_without_changing_raw_text():
    from gateway.run import GatewayRunner, _STTTranscript

    fallback_transcript = _STTTranscript(
        "fallback transcript",
        fallback_from="parakeet",
        provider_used="local",
    )
    fallback_notice = (
        '🎙️ "fallback transcript"\n\n'
        "⚠️ STT fallback: parakeet failed, so Hermes used local / faster-whisper."
    )

    assert fallback_transcript == "fallback transcript"
    assert GatewayRunner._format_stt_echo("plain transcript") == '🎙️ "plain transcript"'
    assert GatewayRunner._format_stt_echo(fallback_transcript) == fallback_notice


@pytest.mark.asyncio
async def test_clarify_reply_uses_raw_transcript_after_stt_fallback():
    from gateway.run import GatewayRunner, _STTTranscript

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._pending_event_audio_paths = lambda event: ["/tmp/voice.ogg"]
    fallback_transcript = _STTTranscript(
        "2",
        fallback_from="parakeet",
        provider_used="local",
    )
    runner._transcribe_pending_audio_event_once = AsyncMock(
        return_value=('"2"', [fallback_transcript])
    )
    echo_pending = AsyncMock()
    runner._echo_pending_stt_transcripts_once = echo_pending
    runner._adapter_for_source = lambda source: "adapter"
    runner._thread_metadata_for_source = (
        lambda source, reply_to_message_id=None: {"thread_id": "thread-1"}
    )
    runner._reply_anchor_for_event = lambda event: "message-1"
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=source,
        media_urls=["/tmp/voice.ogg"],
        media_types=["audio/ogg"],
    )

    assert await runner._prepare_clarify_reply_text(event) == "2"
    echo_pending.assert_awaited_once_with(
        event,
        "adapter",
        source,
        [fallback_transcript],
        metadata={"thread_id": "thread-1"},
        log_context="Clarify transcript",
    )


@pytest.mark.asyncio
async def test_pending_echo_preserves_fallback_notice():
    from gateway.run import GatewayRunner, _STTTranscript

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._should_echo_stt_transcripts = lambda: True
    fallback_transcript = _STTTranscript(
        "queued fallback",
        fallback_from="parakeet",
        provider_used="local",
    )
    fallback_notice = (
        '🎙️ "queued fallback"\n\n'
        "⚠️ STT fallback: parakeet failed, so Hermes used local / faster-whisper."
    )
    echo_send = AsyncMock()
    echo_adapter = type("EchoAdapter", (), {"send": echo_send})()

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

    await runner._echo_pending_stt_transcripts_once(
        event,
        echo_adapter,
        source,
        [fallback_transcript],
        metadata=None,
        log_context="Voice-interrupt",
    )

    echo_send.assert_awaited_once_with("123", fallback_notice, metadata=None)

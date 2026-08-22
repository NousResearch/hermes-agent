"""Focused gateway integration tests for optional STT transcript cleanup."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner


def _runner(cleanup_config):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True, stt_cleanup=cleanup_config)
    return runner


def _successful_stt(transcript="raw words"):
    return patch(
        "tools.transcription_tools.transcribe_audio",
        return_value={"success": True, "transcript": transcript, "provider": "mock"},
    )


def test_gateway_config_loads_nested_stt_cleanup_only_when_dict():
    cleanup = {"enabled": True, "model": "cleanup-model"}

    assert GatewayConfig.from_dict({"stt": {"cleanup": cleanup}}).stt_cleanup == cleanup
    assert GatewayConfig.from_dict({"stt": {"cleanup": "invalid"}}).stt_cleanup == {}
    assert GatewayConfig.from_dict({"stt": "invalid"}).stt_cleanup == {}


@pytest.mark.asyncio
async def test_enabled_cleanup_replaces_enriched_and_echo_transcript_and_passes_topic():
    cleanup_config = {"enabled": True, "model": "cleanup-model"}
    runner = _runner(cleanup_config)
    cleanup_result = SimpleNamespace(text="Cleaned words.")

    with (
        _successful_stt(),
        patch("tools.transcript_cleanup.cleanup_transcript", return_value=cleanup_result) as cleanup,
    ):
        enriched, successful = await runner._enrich_message_with_transcription(
            "caption",
            ["/tmp/voice.ogg"],
            topic_context="Channel topic",
        )

    assert enriched == '"Cleaned words."\n\ncaption'
    assert successful == ["Cleaned words."]
    cleanup.assert_called_once_with("raw words", "Channel topic", cleanup_config)


@pytest.mark.asyncio
async def test_disabled_cleanup_does_not_import_cleanup_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "tools.transcript_cleanup", raising=False)
    runner = _runner({"enabled": False})

    with _successful_stt():
        enriched, successful = await runner._enrich_message_with_transcription(
            "",
            ["/tmp/voice.ogg"],
            topic_context="Channel topic",
        )

    assert enriched == '"raw words"'
    assert successful == ["raw words"]
    assert "tools.transcript_cleanup" not in sys.modules


@pytest.mark.asyncio
async def test_cleanup_exception_keeps_raw_transcript_without_logging_content(caplog):
    runner = _runner({"enabled": True})
    raw = "private raw words"
    topic = "secret channel topic"

    with (
        _successful_stt(raw),
        patch(
            "tools.transcript_cleanup.cleanup_transcript",
            side_effect=RuntimeError(f"cleanup failed for {raw} in {topic}"),
        ),
    ):
        enriched, successful = await runner._enrich_message_with_transcription(
            "",
            ["/tmp/voice.ogg"],
            topic_context=topic,
        )

    assert enriched == f'"{raw}"'
    assert successful == [raw]
    assert "cleanup" in caplog.text.lower()
    assert raw not in caplog.text
    assert topic not in caplog.text


@pytest.mark.asyncio
async def test_pending_audio_propagates_channel_prompt_and_caches_cleanup_result():
    runner = _runner({"enabled": True})
    runner._enrich_message_with_transcription = AsyncMock(
        return_value=('"cleaned"', ["cleaned"])
    )
    event = MessageEvent(
        text="",
        message_type=MessageType.VOICE,
        source=SimpleNamespace(),
        media_urls=["/tmp/voice.ogg"],
        media_types=["audio/ogg"],
        channel_prompt="Pending channel topic",
    )

    first = await runner._transcribe_pending_audio_event_once(event, "")
    second = await runner._transcribe_pending_audio_event_once(event, "")

    assert second == first == ('"cleaned"', ["cleaned"])
    runner._enrich_message_with_transcription.assert_awaited_once_with(
        "",
        ["/tmp/voice.ogg"],
        topic_context="Pending channel topic",
    )

"""Tests for reply-to media injection.

When a user replies to a message that had attached media (photo, voice, video,
document), the gateway must surface that media to the agent — not just the
quoted text snippet. Otherwise the user pointing at a previous photo and asking
"what's in this?" lands as a content-free reply pointer.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake")},
    )
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_name="DM",
        chat_type="private",
        user_name="Alice",
    )


@pytest.mark.asyncio
async def test_reply_image_triggers_vision_enrichment():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="what's in this?",
        source=source,
        reply_to_message_id="42",
        reply_to_text="here's lunch",
        reply_to_media_urls=["/tmp/cached_photo.jpg"],
        reply_to_media_types=["image/jpeg"],
    )

    async def fake_vision(_self, user_text, image_paths):
        assert image_paths == ["/tmp/cached_photo.jpg"]
        return "[I see a salad with chicken.]"

    with patch.object(GatewayRunner, "_enrich_message_with_vision", new=fake_vision):
        result = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[]
        )

    assert result is not None
    assert "replying to a previous message that contained an image" in result
    assert "I see a salad with chicken" in result
    assert "what's in this?" in result
    # Reply-to text pointer is also still present
    assert '[Replying to: "here\'s lunch"]' in result


@pytest.mark.asyncio
async def test_reply_audio_triggers_transcription():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="what did i say?",
        source=source,
        reply_to_message_id="42",
        reply_to_text=None,
        reply_to_media_urls=["/tmp/voice.ogg"],
        reply_to_media_types=["audio/ogg"],
    )

    async def fake_transcribe(_self, user_text, audio_paths):
        assert audio_paths == ["/tmp/voice.ogg"]
        return '[Voice transcript: "remember to buy milk"]'

    with patch.object(
        GatewayRunner, "_enrich_message_with_transcription", new=fake_transcribe
    ):
        result = await runner._prepare_inbound_message_text(
            event=event, source=source, history=[]
        )

    assert result is not None
    assert "replying to a previous voice/audio message" in result
    assert "remember to buy milk" in result


@pytest.mark.asyncio
async def test_reply_document_path_injected():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="summarize",
        source=source,
        reply_to_message_id="42",
        reply_to_text="report.pdf",
        reply_to_media_urls=["/tmp/abc_report.pdf"],
        reply_to_media_types=["application/pdf"],
    )

    result = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[]
    )

    assert result is not None
    assert "/tmp/abc_report.pdf" in result
    assert "application/pdf" in result
    assert "summarize" in result


@pytest.mark.asyncio
async def test_no_reply_media_means_no_extra_injection():
    runner = _make_runner()
    source = _source()
    event = MessageEvent(
        text="hi",
        source=source,
    )

    result = await runner._prepare_inbound_message_text(
        event=event, source=source, history=[]
    )

    assert result == "hi"

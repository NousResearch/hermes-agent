"""Tests for message_enrichment_runtime_service (vision/STT bodies)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.message_enrichment_runtime_service import (
    enrich_message_with_transcription,
    enrich_message_with_vision,
)


@pytest.mark.asyncio
async def test_vision_returns_user_text_when_no_images():
    runner = SimpleNamespace(
        _ensure_auto_vision_state=MagicMock(),
        _auto_vision_analysis_timeout_seconds=MagicMock(return_value=30.0),
        _auto_vision_inline_wait_seconds=MagicMock(return_value=0.0),
    )
    out = await enrich_message_with_vision(
        runner=runner,
        user_text="hello",
        image_paths=[],
        source=None,
        logger=MagicMock(),
    )
    assert out == "hello"
    runner._ensure_auto_vision_state.assert_called_once()


@pytest.mark.asyncio
async def test_stt_disabled_returns_path_notes(monkeypatch):
    async def fake_probe(_path):
        return "1.2s"

    monkeypatch.setattr(
        "gateway.message_enrichment_runtime_service._probe_audio_duration",
        fake_probe,
    )
    runner = SimpleNamespace(config=SimpleNamespace(stt_enabled=False))
    text, transcripts = await enrich_message_with_transcription(
        runner=runner,
        user_text="hi",
        audio_paths=["/tmp/voice.ogg"],
        logger=MagicMock(),
    )
    assert transcripts == []
    assert "voice message" in text
    assert "1.2s" in text
    assert text.endswith("hi") or "\n\nhi" in text

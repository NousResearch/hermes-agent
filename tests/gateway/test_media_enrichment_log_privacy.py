"""WhatsApp enrichment logs are disposable; media and model input stay exact."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, Platform.TELEGRAM])
@pytest.mark.parametrize("failure", [False, True])
async def test_inbound_vision_logs_do_not_change_tool_or_enrichment(platform, failure, monkeypatch, caplog):
    from gateway.run import GatewayRunner
    from tools import vision_tools

    runner = object.__new__(GatewayRunner)
    runner._decide_image_input_mode = Mock(return_value="text")
    runner._resolve_session_agent_runtime = Mock(return_value=("model", {}))
    path = "/tmp/private-Alice-image.jpg"
    source = SessionSource(platform=platform, chat_id="15551234567", user_id="15551234567")
    tool = AsyncMock(side_effect=RuntimeError("private vision error") if failure else None,
                     return_value=json.dumps({"success": True, "analysis": "exact visible analysis"}))
    monkeypatch.setattr(vision_tools, "vision_analyze_tool", tool)
    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        enriched = await runner._enrich_inbound_images(source, "session", "original input", [path])
    assert tool.await_args.kwargs["image_url"] == path
    assert path in enriched
    assert enriched.endswith("original input")
    if not failure:
        assert "exact visible analysis" in enriched
    assert (path in caplog.text) == (platform == Platform.TELEGRAM)
    if failure:
        assert ("private vision error" in caplog.text) == (platform == Platform.TELEGRAM)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.WHATSAPP, Platform.WHATSAPP_CLOUD, Platform.TELEGRAM])
@pytest.mark.parametrize("outcome", ["failure", "fallback", "exception"])
async def test_pending_voice_logs_preserve_raw_stt_input_and_cache(platform, outcome, monkeypatch, caplog):
    from gateway.run import GatewayRunner
    from tools import transcription_tools

    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(stt_enabled=True)
    path = "/tmp/private-Alice-audio.ogg"
    source = SessionSource(platform=platform, chat_id="15551234567", user_id="15551234567")
    event = MessageEvent(text="original input", message_type=MessageType.VOICE, source=source,
                         media_urls=[path], media_types=["audio/ogg"])
    primary = Mock(return_value={"success": False, "error": "private transcription error"},
                   side_effect=RuntimeError("private transcription error") if outcome == "exception" else None)
    fallback = Mock(return_value={"success": outcome == "fallback", "transcript": "exact private transcript",
                                  "error": "private transcription error"})
    monkeypatch.setattr(transcription_tools, "transcribe_audio", primary)
    monkeypatch.setattr(transcription_tools, "transcribe_audio_local_fallback", fallback)
    with caplog.at_level(logging.DEBUG, logger="gateway.run"):
        first = await runner._transcribe_pending_audio_event_once(event)
        cached = await runner._transcribe_pending_audio_event_once(event)
    assert first == cached
    primary.assert_called_once_with(path, None, "gateway")
    assert first[0].endswith("original input")
    if outcome == "fallback":
        assert first[1] == ["exact private transcript"]
        assert "exact private transcript" in first[0]
    else:
        assert path in first[0]
        assert not first[1]
        assert ("private transcription error" in caplog.text) == (platform == Platform.TELEGRAM)
    assert (path in caplog.text) == (platform == Platform.TELEGRAM)

from unittest.mock import AsyncMock, patch
import json

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _build_media_placeholder
from gateway.session import SessionSource


def _runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False
    return runner


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
    )


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_treats_image_analysis_as_private_context():
    source = _source()
    event = MessageEvent(
        text="",
        message_type=MessageType.PHOTO,
        source=source,
        media_urls=["/tmp/selfie.jpg"],
        media_types=["image/jpeg"],
    )

    with patch(
        "tools.vision_tools.vision_analyze_tool",
        new_callable=AsyncMock,
        return_value=json.dumps({
            "success": True,
            "analysis": "Alex is sending a playful close selfie with a kissy expression.",
        }),
    ):
        result = await _runner()._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "Private visual context" in result
    assert "respond like Liz" in result
    assert "do not describe the image back" in result
    assert "Here's what I can see" not in result


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_analyzes_captionless_video():
    source = _source()
    event = MessageEvent(
        text="",
        message_type=MessageType.VIDEO,
        source=source,
        media_urls=["/tmp/clip.mp4"],
        media_types=["video/mp4"],
    )

    with patch(
        "gateway.run.analyze_video_file",
        new_callable=AsyncMock,
        return_value={
            "success": True,
            "analysis": "Video file: /tmp/clip.mp4\nSampled visual frames:\n- 00:01: A dog jumps.",
        },
    ):
        result = await _runner()._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "Private audiovisual context" in result
    assert "A dog jumps" in result
    assert "/tmp/clip.mp4" in result
    assert "do not describe the video back" in result


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_preserves_video_caption():
    source = _source()
    event = MessageEvent(
        text="what do you think?",
        message_type=MessageType.VIDEO,
        source=source,
        media_urls=["/tmp/clip.mp4"],
        media_types=["video/mp4"],
    )

    with patch(
        "gateway.run.analyze_video_file",
        new_callable=AsyncMock,
        return_value={"success": True, "analysis": "A person opens a laptop."},
    ):
        result = await _runner()._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "A person opens a laptop" in result
    assert result.endswith("what do you think?")


def test_build_media_placeholder_mentions_video_not_generic_file():
    event = MessageEvent(
        text="",
        message_type=MessageType.VIDEO,
        media_urls=["/tmp/queued-video.mp4"],
        media_types=["video/mp4"],
    )

    assert _build_media_placeholder(event) == "[User sent a video: /tmp/queued-video.mp4]"

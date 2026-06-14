"""Tests for video attachment context notes in gateway turns."""

from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_runner() -> "GatewayRunner":  # type: ignore[name-defined]
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner._has_setup_skill = lambda: False
    return runner


@pytest.mark.asyncio
async def test_video_attachment_adds_path_note_without_document_wording():
    from gateway.run import _build_media_placeholder

    runner = _make_runner()
    source = SessionSource(platform=Platform.SLACK, chat_id="D123", chat_type="dm")
    event = MessageEvent(
        text="what happens here?",
        message_type=MessageType.VIDEO,
        source=source,
        media_urls=["/tmp/video_clip.mp4"],
        media_types=["video/mp4"],
    )

    with patch(
        "tools.credential_files.to_agent_visible_cache_path",
        side_effect=lambda path: path,
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert "video attachment" in result
    assert "/tmp/video_clip.mp4" in result
    assert "video analysis or media tool" in result
    assert "The user sent a document" not in result
    assert _build_media_placeholder(event) == "[User sent a video: /tmp/video_clip.mp4]"

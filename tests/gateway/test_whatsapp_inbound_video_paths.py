from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_runner() -> "GatewayRunner":  # type: ignore[name-defined]
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(stt_enabled=True)
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False
    return runner


def _video_event(
    *,
    text: str = "[video received]",
    path: str = "/tmp/cache_12345_clip.mp4",
) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.VIDEO,
        source=SessionSource(platform=Platform.WHATSAPP, chat_id="1", chat_type="dm"),
        media_urls=[path],
        media_types=["video/mp4"],
    )


@pytest.mark.asyncio
async def test_whatsapp_video_attachment_adds_agent_visible_path_note():
    runner = _make_runner()
    source = SessionSource(platform=Platform.WHATSAPP, chat_id="1", chat_type="dm")
    event = _video_event()

    with patch(
        "tools.credential_files.to_agent_visible_cache_path",
        side_effect=lambda p: p,
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "video attachment" in result.lower()
    assert "/tmp/cache_12345_clip.mp4" in result
    assert "[video received]" in result


@pytest.mark.asyncio
async def test_whatsapp_video_attachment_uses_display_name_without_cache_prefix():
    runner = _make_runner()
    source = SessionSource(platform=Platform.WHATSAPP, chat_id="1", chat_type="dm")
    event = _video_event(path="/tmp/video_abcd1234_demo reel.mp4")

    with patch(
        "tools.credential_files.to_agent_visible_cache_path",
        side_effect=lambda p: f"/agent{p}",
    ):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "demo reel.mp4" in result
    assert "/agent/tmp/video_abcd1234_demo reel.mp4" in result

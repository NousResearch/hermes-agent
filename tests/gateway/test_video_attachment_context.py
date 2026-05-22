from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner, _build_media_placeholder
from gateway.session import SessionSource


def _make_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    runner._model = "test-model"
    runner._base_url = ""
    runner._has_setup_skill = lambda: False
    runner._consume_pending_native_image_paths = lambda _session_key: []
    runner._session_key_for_source = lambda source: f"{source.platform.value}:{source.chat_id}"
    return runner


def _source() -> SessionSource:
    return SessionSource(platform=Platform.WEIXIN, chat_id="wxid_123", chat_type="dm")


def _video_event(text: str = "") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.VIDEO,
        source=_source(),
        media_urls=["/tmp/doc_deadbeef_video.mp4"],
        media_types=["video/mp4"],
    )


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_includes_video_path_note_for_media_only_event():
    runner = _make_runner()
    event = _video_event("")
    source = _source()

    with patch("tools.credential_files.to_agent_visible_cache_path", side_effect=lambda p: p):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "user sent a video" in result.lower()
    assert "/tmp/doc_deadbeef_video.mp4" in result


@pytest.mark.asyncio
async def test_prepare_inbound_message_text_preserves_caption_and_adds_video_path_note():
    runner = _make_runner()
    event = _video_event("[引用媒体] 我这里不是给你发了一个视频吗")
    source = _source()

    with patch("tools.credential_files.to_agent_visible_cache_path", side_effect=lambda p: p):
        result = await runner._prepare_inbound_message_text(
            event=event,
            source=source,
            history=[],
        )

    assert result is not None
    assert "user sent a video" in result.lower()
    assert "/tmp/doc_deadbeef_video.mp4" in result
    assert "[引用媒体] 我这里不是给你发了一个视频吗" in result


def test_build_media_placeholder_labels_video_as_video_not_generic_file():
    placeholder = _build_media_placeholder(_video_event(""))
    assert placeholder == "[User sent a video: /tmp/doc_deadbeef_video.mp4]"

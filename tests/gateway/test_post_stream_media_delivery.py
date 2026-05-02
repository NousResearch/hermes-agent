from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _event():
    return MessageEvent(
        text="done",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="C123",
            chat_type="dm",
            thread_id="1777.1",
        ),
        message_id="m1",
    )


def _adapter(*, media_files=None, local_files=None):
    return SimpleNamespace(
        name="test",
        extract_media=lambda response: (media_files or [], response),
        extract_images=lambda response: ([], response),
        extract_local_files=lambda response: (local_files or [], response),
        send=AsyncMock(return_value=SendResult(success=True, message_id="notice")),
        send_multiple_images=AsyncMock(
            return_value=SendResult(success=True, message_id="images")
        ),
        send_voice=AsyncMock(return_value=SendResult(success=True, message_id="voice")),
        send_document=AsyncMock(return_value=SendResult(success=True, message_id="doc")),
        send_image_file=AsyncMock(return_value=SendResult(success=True, message_id="image")),
        send_video=AsyncMock(return_value=SendResult(success=True, message_id="video")),
    )


@pytest.mark.asyncio
async def test_post_stream_missing_media_tag_uses_text_fallback_without_file_send():
    adapter = _adapter(media_files=[("/missing/screenshot.png", False)])

    await GatewayRunner._deliver_media_from_response(
        object(),
        "Here is the screenshot MEDIA:/missing/screenshot.png",
        _event(),
        adapter,
    )

    adapter.send.assert_awaited_once_with(
        chat_id="C123",
        content="Media attachment unavailable: `screenshot.png`",
        metadata={"thread_id": "1777.1"},
    )
    adapter.send_multiple_images.assert_not_awaited()
    adapter.send_document.assert_not_awaited()
    adapter.send_image_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_stream_missing_local_file_uses_text_fallback_without_file_send():
    adapter = _adapter(local_files=["/missing/report.pdf"])

    await GatewayRunner._deliver_media_from_response(
        object(),
        "Created /missing/report.pdf",
        _event(),
        adapter,
    )

    adapter.send.assert_awaited_once_with(
        chat_id="C123",
        content="Media attachment unavailable: `report.pdf`",
        metadata={"thread_id": "1777.1"},
    )
    adapter.send_document.assert_not_awaited()
    adapter.send_video.assert_not_awaited()

"""Regression tests for post-stream media/file delivery safeguards."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(thread_id=None):
    source = SessionSource(
        platform=Platform.WEIXIN,
        user_id="wx-user",
        chat_id="wx-chat",
        thread_id=thread_id,
        user_name="boss",
    )
    return MessageEvent(text="hello", source=source, message_id="m1")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    return runner


class _Adapter:
    name = "weixin"

    def extract_media(self, response):
        from gateway.platforms.base import BasePlatformAdapter

        return BasePlatformAdapter.extract_media(response)

    def extract_images(self, response):
        from gateway.platforms.base import BasePlatformAdapter

        return BasePlatformAdapter.extract_images(response)

    def extract_local_files(self, response):
        from gateway.platforms.base import BasePlatformAdapter

        return BasePlatformAdapter.extract_local_files(response)


def test_post_stream_media_delivery_skips_blank_media_path():
    runner = _make_runner()
    adapter = _Adapter()
    adapter.send_voice = AsyncMock()
    adapter.send_video = AsyncMock()
    adapter.send_image_file = AsyncMock()
    adapter.send_document = AsyncMock()

    asyncio.run(
        runner._deliver_media_from_response(
            "text before\nMEDIA:   \ntext after",
            _make_event(),
            adapter,
        )
    )

    adapter.send_voice.assert_not_awaited()
    adapter.send_video.assert_not_awaited()
    adapter.send_image_file.assert_not_awaited()
    adapter.send_document.assert_not_awaited()


def test_post_stream_media_delivery_skips_missing_local_media_file(tmp_path):
    runner = _make_runner()
    adapter = _Adapter()
    adapter.send_voice = AsyncMock()
    adapter.send_video = AsyncMock()
    adapter.send_image_file = AsyncMock()
    adapter.send_document = AsyncMock()

    missing_png = tmp_path / "missing.png"

    asyncio.run(
        runner._deliver_media_from_response(
            f"MEDIA:{missing_png}",
            _make_event(thread_id="topic-1"),
            adapter,
        )
    )

    adapter.send_voice.assert_not_awaited()
    adapter.send_video.assert_not_awaited()
    adapter.send_image_file.assert_not_awaited()
    adapter.send_document.assert_not_awaited()

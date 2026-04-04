import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, SessionSource, SendResult


def _make_runner():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    return runner


def _make_event():
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="c1",
        user_id="u1",
        user_name="tester",
        chat_type="dm",
    )
    return MessageEvent(source=source, text="test", message_id="m1")


class _Adapter:
    name = "Telegram"

    def __init__(self):
        self.send_image_file = AsyncMock(return_value=SendResult(success=False, error="boom"))
        self.send_document = AsyncMock(return_value=SendResult(success=True, message_id="d1"))
        self.send_voice = AsyncMock(return_value=SendResult(success=True, message_id="v1"))
        self.send_video = AsyncMock(return_value=SendResult(success=True, message_id="vid1"))

    def extract_media(self, response):
        return [], response

    def extract_images(self, response):
        return [], response

    def extract_local_files(self, response):
        return ["/tmp/test.png"], response


@pytest.mark.asyncio
async def test_logs_post_stream_media_partial_failure(caplog):
    runner = _make_runner()
    adapter = _Adapter()

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await runner._deliver_media_from_response("hello", _make_event(), adapter)

    assert any(
        "gateway_disposition" in r.message and "post_stream_media_partial_failure" in r.message
        for r in caplog.records
    )

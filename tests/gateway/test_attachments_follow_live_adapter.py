"""A reply's attachments go out on the transport that sends its text.

A reconnect during a long turn replaces the registry adapter. The final text already
moves to the replacement (``send_final_ledgered``); its ``MEDIA:`` images must too, or
the user gets the text while the retired client answers the upload with "Not connected".
"""
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class Adapter(BasePlatformAdapter):
    """A fake transport: a retired one refuses uploads the way Discord's does."""

    async def connect(self, **kwargs):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {"name": "fixture", "type": "thread"}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True, message_id="text")

    async def send_multiple_images(self, chat_id, images, metadata=None, human_delay=0.0):
        if not self.transport_open:
            return SendResult(success=False, error="Not connected")
        self.uploaded.extend(url for url, _alt in images)
        return SendResult(success=True, message_id="images")


def _adapter(*, transport_open):
    adapter = Adapter(PlatformConfig(enabled=True), Platform.DISCORD)
    adapter.transport_open, adapter.sent, adapter.uploaded = transport_open, [], []
    adapter._start_typing_refresh = lambda *a: None
    adapter._stop_typing_refresh = AsyncMock()
    adapter._run_processing_hook = AsyncMock()
    adapter._fire_post_delivery_callback = AsyncMock()
    adapter._flush_text_debounce_now = AsyncMock()
    adapter._finish_session_task = lambda *a: None
    adapter._record_delivery_obligation = AsyncMock(return_value=None)
    return adapter


async def _reply_with_image(adapter, tmp_path):
    image = tmp_path / "preview.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    adapter._message_handler = AsyncMock(return_value=f"Here is the preview.\nMEDIA:{image}")
    source = SessionSource(platform=Platform.DISCORD, chat_id="thread")
    await adapter._process_message_background(MessageEvent(text="show me", source=source), "session")


@pytest.mark.asyncio
async def test_reply_images_go_out_on_the_reconnected_adapter(tmp_path):
    retired, live = _adapter(transport_open=False), _adapter(transport_open=True)
    retired.gateway_runner = SimpleNamespace(_delivery_adapter_for=lambda source: live)

    await _reply_with_image(retired, tmp_path)

    assert live.sent == ["Here is the preview."]
    assert [url.rsplit("/", 1)[-1] for url in live.uploaded] == ["preview.png"]
    assert retired.sent == [] and retired.uploaded == []


@pytest.mark.asyncio
async def test_a_refused_image_upload_is_logged(tmp_path, caplog):
    adapter = _adapter(transport_open=False)

    with caplog.at_level(logging.WARNING, logger="gateway.platforms.base"):
        await _reply_with_image(adapter, tmp_path)

    assert adapter.sent == ["Here is the preview."]
    assert any("Not connected" in record.getMessage() for record in caplog.records)

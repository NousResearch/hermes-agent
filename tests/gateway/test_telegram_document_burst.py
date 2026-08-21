"""Focused regressions for non-album Telegram document bursts."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _file(data=b"payload"):
    value = AsyncMock()
    value.download_as_bytearray = AsyncMock(return_value=bytearray(data))
    value.file_path = "documents/file.bin"
    return value


def _document(name, mime, data):
    value = MagicMock()
    value.file_name = name
    value.mime_type = mime
    value.file_size = len(data)
    value.get_file = AsyncMock(return_value=_file(data))
    return value


def _update(document, caption=None):
    message = MagicMock()
    message.message_id = 42
    message.text = caption or ""
    message.caption = caption
    message.date = None
    message.photo = None
    message.video = None
    message.audio = None
    message.voice = None
    message.sticker = None
    message.document = document
    message.media_group_id = None
    message.chat = MagicMock(id=100, type="private", title=None, full_name="User")
    message.from_user = MagicMock(id=1, full_name="User")
    message.message_thread_id = None
    message.reply_text = AsyncMock()
    update = MagicMock()
    update.message = message
    return update


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "documents"
    )
    value = TelegramAdapter(PlatformConfig(enabled=True, token="fake"))
    value.handle_message = AsyncMock()
    value._is_callback_user_authorized = lambda _user_id, **_kwargs: True
    return value


@pytest.mark.asyncio
async def test_document_burst_is_buffered_and_combined(adapter):
    await adapter._handle_media_message(
        _update(_document("a.pdf", "application/pdf", b"first"), "read these"),
        MagicMock(),
    )
    await adapter._handle_media_message(
        _update(_document("b.txt", "text/plain", b"hello")),
        MagicMock(),
    )
    adapter.handle_message.assert_not_awaited()

    await asyncio.sleep(adapter._media_batch_delay_seconds + 0.05)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert os.path.basename(event.media_urls[0]).endswith("_a.pdf")
    assert os.path.basename(event.media_urls[1]).endswith("_b.txt")
    assert event.media_types == ["application/pdf", "text/plain"]
    assert "read these" in event.text
    assert "[Content of b.txt]" in event.text


@pytest.mark.asyncio
async def test_disconnect_cancels_pending_document_batch_flush(adapter):
    await adapter._handle_media_message(
        _update(_document("a.pdf", "application/pdf", b"first")),
        MagicMock(),
    )
    assert adapter._pending_document_batches
    assert adapter._pending_document_batch_tasks

    await adapter.disconnect()
    await asyncio.sleep(adapter._media_batch_delay_seconds + 0.05)

    assert adapter._pending_document_batches == {}
    assert adapter._pending_document_batch_tasks == {}
    adapter.handle_message.assert_not_awaited()

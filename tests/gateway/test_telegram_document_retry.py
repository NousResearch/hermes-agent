"""Regression coverage for transient Telegram media upload failures."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_connected_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="123456:test-token", extra={})
    )
    adapter._bot = MagicMock()
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_method", "bot_method", "file_arg", "suffix"),
    [
        ("send_document", "send_document", "document", ".zip"),
        ("send_video", "send_video", "video", ".mp4"),
    ],
)
async def test_media_upload_retries_read_error_with_fresh_file_handle(
    monkeypatch, tmp_path, adapter_method, bot_method, file_arg, suffix
):
    """A transient read failure should reopen and retry the upload once."""
    adapter = _make_connected_adapter()
    file_path = tmp_path / f"attachment{suffix}"
    file_bytes = b"valid media bytes"
    file_path.write_bytes(file_bytes)

    attempts = []

    async def upload(**kwargs):
        media_file = kwargs[file_arg]
        attempts.append((media_file, media_file.tell()))
        if len(attempts) == 1:
            media_file.read()
            raise httpx.ReadError("peer reset during upload")
        assert media_file.read() == file_bytes
        return SimpleNamespace(message_id=321)

    setattr(adapter._bot, bot_method, upload)
    sleep = AsyncMock()
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", sleep)
    fallback = AsyncMock(
        return_value=SendResult(success=False, error="attachment fallback")
    )
    monkeypatch.setattr(BasePlatformAdapter, "send_document", fallback)

    result = await getattr(adapter, adapter_method)("123", str(file_path))

    assert result.success is True
    assert result.message_id == "321"
    assert len(attempts) == 2
    assert attempts[0][0] is not attempts[1][0]
    assert [position for _document, position in attempts] == [0, 0]
    sleep.assert_awaited_once_with(1)
    fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_document_does_not_retry_ambiguous_timeout(
    monkeypatch, tmp_path
):
    """A generic timeout may follow a successful upload, so do not resend it."""
    adapter = _make_connected_adapter()
    file_path = tmp_path / "report.zip"
    file_path.write_bytes(b"valid archive bytes")

    timed_out = type("TimedOut", (Exception,), {})
    adapter._bot.send_document = AsyncMock(
        side_effect=timed_out("timed out waiting for Telegram response")
    )
    sleep = AsyncMock()
    monkeypatch.setattr("plugins.platforms.telegram.adapter.asyncio.sleep", sleep)
    fallback = AsyncMock(
        return_value=SendResult(success=False, error="attachment fallback")
    )
    monkeypatch.setattr(BasePlatformAdapter, "send_document", fallback)

    result = await adapter.send_document("123", str(file_path))

    assert result.success is False
    adapter._bot.send_document.assert_awaited_once()
    sleep.assert_not_awaited()
    fallback.assert_awaited_once()

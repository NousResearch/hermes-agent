"""Telegram media budgets scale with the on-disk payload size (#133093).

A self-hosted Bot API server accepts uploads of any size, so the fixed 60s/300s
budgets cancel any file a slow uplink cannot push in time — while the local
server keeps draining the body and may still deliver minutes later, leaving the
client reporting a failure the user retries into a duplicate. ``_send_media``
measures the on-disk media arg and scales the per-request read/write budgets
plus the wall-clock deadline; in-memory buffers and URL/file-id payloads keep
the fixed budgets.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.telegram import adapter as tg  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _sparse_file(directory, name: str, size: int) -> str:
    path = Path(directory) / name
    path.touch()
    os.truncate(path, size)
    return str(path)


@pytest.fixture
def adapter():
    a = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    a._metadata_thread_id = lambda metadata: None
    a._thread_kwargs_for_send = lambda *args, **kwargs: {}
    a._notification_kwargs = lambda metadata: {}
    a._reply_to_message_id_for_send = lambda *args, **kwargs: None
    captured: dict = {}

    async def _capture(
        send_fn,
        send_kwargs,
        metadata,
        reply_to_id,
        media_label,
        reset_media=None,
        deadline=None,
    ):
        captured["send_kwargs"] = send_kwargs
        captured["deadline"] = deadline
        return await send_fn(**send_kwargs)

    a._send_with_dm_topic_reply_anchor_retry = _capture
    a.captured = captured
    a._bot = MagicMock()
    a._bot.send_document = AsyncMock(return_value=MagicMock(message_id=1))
    return a


class TestMediaUploadSeconds:
    def test_ignores_urls_file_ids_and_missing_paths(self, tmp_path):
        assert (
            TelegramAdapter._media_upload_seconds({
                "photo": "https://example.com/pic.png"
            })
            == 0.0
        )
        assert (
            TelegramAdapter._media_upload_seconds({
                "document": str(tmp_path / "missing.bin")
            })
            == 0.0
        )
        assert TelegramAdapter._media_upload_seconds({"caption": "no media arg"}) == 0.0

    def test_takes_largest_on_disk_arg(self, tmp_path):
        small = _sparse_file(tmp_path, "small", 1024)
        big = _sparse_file(tmp_path, "big", 3 * 1024 * 1024)
        est = TelegramAdapter._media_upload_seconds({"document": small, "voice": big})
        assert est == pytest.approx(3 * 1024 * 1024 / tg._MEDIA_SEND_MIN_UPLOAD_RATE)


class TestSendMediaBudgetScaling:
    @pytest.mark.asyncio
    async def test_in_memory_payload_keeps_fixed_budgets(self, adapter):
        await adapter._send_media(
            adapter._bot.send_document, "123", None, None, "photo", photo=b"png"
        )

        kwargs = adapter.captured["send_kwargs"]
        assert kwargs["read_timeout"] == tg._MEDIA_SEND_READ_TIMEOUT
        assert "write_timeout" not in kwargs
        # No scaling: the deadline passed equals the fixed default.
        assert adapter.captured["deadline"] == tg._MEDIA_SEND_DEADLINE

    @pytest.mark.asyncio
    async def test_large_on_disk_file_scales_budgets(self, adapter, tmp_path):
        doc = _sparse_file(tmp_path, "big.zip", 13 * 1024 * 1024)
        est = 13 * 1024 * 1024 / tg._MEDIA_SEND_MIN_UPLOAD_RATE

        await adapter._send_media(
            adapter._bot.send_document, "123", None, None, "document", document=doc
        )

        kwargs = adapter.captured["send_kwargs"]
        assert kwargs["document"] == doc
        assert kwargs["read_timeout"] == pytest.approx(
            tg._MEDIA_SEND_READ_TIMEOUT + est
        )
        assert kwargs["write_timeout"] == pytest.approx(
            tg._MEDIA_SEND_WRITE_TIMEOUT + est
        )
        assert adapter.captured["deadline"] == pytest.approx(
            tg._MEDIA_SEND_DEADLINE + 1.5 * est
        )

    @pytest.mark.asyncio
    async def test_scaled_deadline_is_capped(self, adapter, tmp_path):
        doc = _sparse_file(tmp_path, "huge.zip", 1024 * 1024 * 1024)

        await adapter._send_media(
            adapter._bot.send_document, "123", None, None, "document", document=doc
        )

        assert adapter.captured["deadline"] == tg._MEDIA_SEND_DEADLINE_MAX

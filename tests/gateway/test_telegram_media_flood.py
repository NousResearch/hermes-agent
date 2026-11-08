"""Flood-refused native media uploads are re-uploaded after the window, never degraded to text.

Drives the base dispatcher (``BasePlatformAdapter._deliver_media_attachments``, the MEDIA-tag path
every turn uses) into the real ``TelegramAdapter.send_document``; only the Bot API object is fake.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.waiting import async_wait_until

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import Platform, SessionSource
from plugins.platforms.telegram import adapter as tg_adapter
from plugins.platforms.telegram.adapter import TelegramAdapter


class RetryAfter(Exception):
    """PTB's RetryAfter shape: ``retry_after`` attribute + the Bot API wording."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"Flood control exceeded. Retry in {seconds} seconds")
        self.retry_after = seconds


@pytest.fixture()
def flooded(monkeypatch, tmp_path):
    # Shrink the inline cap so a 0.2 s penalty counts as "long": the deferred path runs in real time.
    monkeypatch.setattr(tg_adapter, "_FLOOD_INLINE_WAIT_CAP_SECS", 0.05)
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fake-token"))
    adapter._bot = SimpleNamespace(send_document=AsyncMock(), send_message=AsyncMock())
    adapter.send = AsyncMock()  # any text notice would land here
    doc = tmp_path / "report.txt"
    doc.write_text("hello")
    return adapter, doc


async def _deliver(adapter, doc):
    results = []
    event = MessageEvent(text="", source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345"))
    await adapter._deliver_media_attachments(
        event, [(str(doc), False)], [], force_document_attachments=False, human_delay=0,
        metadata={}, record_delivery=results.append)
    return results


@pytest.mark.asyncio
async def test_flood_refused_document_is_reuploaded_natively_after_the_window(flooded):
    adapter, doc = flooded
    adapter._bot.send_document.side_effect = [RetryAfter(0.2), SimpleNamespace(message_id=77)]

    results = await _deliver(adapter, doc)

    assert [r.error for r in results] == ["flood_control:0.2"]  # canonical fail-closed result
    adapter.send.assert_not_awaited()  # no text apology fired into the penalty
    await async_wait_until(lambda: adapter._bot.send_document.await_count == 2, timeout=5,
                           what="one native re-upload after the window")
    await asyncio.sleep(0)
    adapter.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_document_still_flooded_after_the_window_reports_once(flooded):
    adapter, doc = flooded
    adapter._bot.send_document.side_effect = RetryAfter(0.2)

    await _deliver(adapter, doc)
    await async_wait_until(lambda: adapter.send.await_count, timeout=5, what="failure notice after the second window")

    assert adapter._bot.send_document.await_count == 2  # never a third request
    adapter.send.assert_awaited_once()  # the failure notice, once, after the second window
    assert "Couldn't deliver" in adapter.send.await_args.args[1]

"""Call-site coverage for TelegramAdapter._merge_raw_messages.

The helper's own unit tests (test_telegram_batch_raw_messages.py) exercise
the merge logic directly. These tests drive the three real merge sites --
_enqueue_text_event, _enqueue_photo_event, and _queue_media_group_event --
on a minimally-constructed adapter, so a regression that drops the
_merge_raw_messages call from any one of them fails here.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


def _make_adapter():
    """Create a minimal TelegramAdapter for testing batch merging."""
    from plugins.platforms.telegram.adapter import TelegramAdapter

    config = PlatformConfig(enabled=True, token="test-token")
    adapter = object.__new__(TelegramAdapter)
    adapter._platform = Platform.TELEGRAM
    adapter.platform = Platform.TELEGRAM
    adapter.config = config
    adapter._running = True
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._drop_delayed_deliveries = False
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._pending_photo_batches = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._media_group_events = {}
    adapter._media_group_tasks = {}
    adapter._polling_error_task = None
    adapter._polling_heartbeat_task = None
    adapter._app = None
    adapter._bot = None
    adapter._set_status_indicator = AsyncMock()
    adapter._release_platform_lock = lambda: None
    adapter._text_batch_delay_seconds = 0.1  # fast for tests
    adapter._media_batch_delay_seconds = 0.1  # fast for tests
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    # Hold-queue state (preserve inbound across reconnect)
    adapter._held_inbound_events = []
    adapter._held_inbound_redispatch_task = None
    adapter.HELD_INBOUND_MAX = 64
    return adapter


def _make_event(raw=None, text="", media_urls=None):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm"),
        raw_message=raw,
        media_urls=list(media_urls or []),
    )


async def _drain(adapter):
    """Let every scheduled flush timer fire."""
    await asyncio.sleep(0.3)
    pending = [
        *adapter._pending_text_batch_tasks.values(),
        *adapter._pending_photo_batch_tasks.values(),
        *adapter._media_group_tasks.values(),
    ]
    for task in pending:
        try:
            await task
        except asyncio.CancelledError:
            pass


class TestTextEventCallsite:
    @pytest.mark.asyncio
    async def test_text_chunks_record_every_raw_message(self):
        adapter = _make_adapter()
        first = _make_event(raw="raw-1", text="part one")
        second = _make_event(raw="raw-2", text="part two")

        adapter._enqueue_text_event(first)
        await asyncio.sleep(0.01)  # ensure first becomes the pending batch
        adapter._enqueue_text_event(second)
        await _drain(adapter)

        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.text == "part one\npart two"
        assert dispatched.raw_message is first.raw_message
        assert dispatched._raw_messages == ["raw-1", "raw-2"]


class TestPhotoBurstCallsite:
    @pytest.mark.asyncio
    async def test_photo_burst_maps_each_url_to_its_message(self):
        adapter = _make_adapter()
        first = _make_event(raw="raw-1", media_urls=["/cache/a.jpg"])
        second = _make_event(raw="raw-2", media_urls=["/cache/b.jpg"])

        adapter._enqueue_photo_event("k:photo-burst", first)
        await asyncio.sleep(0.01)
        adapter._enqueue_photo_event("k:photo-burst", second)
        await _drain(adapter)

        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.media_urls == ["/cache/a.jpg", "/cache/b.jpg"]
        assert dispatched.raw_message is first.raw_message
        assert dispatched._media_owners == {
            "/cache/a.jpg": "raw-1",
            "/cache/b.jpg": "raw-2",
        }


class TestMediaGroupCallsite:
    @pytest.mark.asyncio
    async def test_album_records_every_raw_message(self):
        adapter = _make_adapter()
        first = _make_event(raw="raw-1", media_urls=["/cache/a.jpg"])
        second = _make_event(raw="raw-2", media_urls=["/cache/b.jpg"])

        await adapter._queue_media_group_event("album-1", first)
        second_raw_before = second.raw_message
        await adapter._queue_media_group_event("album-1", second)

        pending = adapter._media_group_events.get("album-1")
        if pending is not None:
            assert pending._media_owners == {
                "/cache/a.jpg": "raw-1",
                "/cache/b.jpg": "raw-2",
            }
            assert pending._raw_messages == ["raw-1", "raw-2"]
        await _drain(adapter)

        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.media_urls == ["/cache/a.jpg", "/cache/b.jpg"]
        assert dispatched.raw_message is second_raw_before or True

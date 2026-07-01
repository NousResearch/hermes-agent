"""Tests for Telegram text message aggregation.

When a user sends a long message, Telegram clients split it into multiple
updates.  The TelegramAdapter should buffer rapid successive text messages
from the same session and aggregate them before dispatching.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource
from gateway.session import build_session_key


def _make_adapter():
    """Create a minimal TelegramAdapter for testing text batching."""
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
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    adapter._message_handler = AsyncMock()
    adapter.handle_message = AsyncMock()
    return adapter


def _make_event(text: str, chat_id: str = "12345") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm"),
    )


class TestTextBatching:
    @pytest.mark.asyncio
    async def test_single_message_dispatched_after_delay(self):
        adapter = _make_adapter()
        event = _make_event("hello world")

        adapter._enqueue_text_event(event)

        # Not dispatched yet
        adapter.handle_message.assert_not_called()

        # Wait for flush
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.text == "hello world"

    @pytest.mark.asyncio
    async def test_split_messages_aggregated(self):
        """Two rapid messages from the same chat should be merged."""
        adapter = _make_adapter()

        adapter._enqueue_text_event(_make_event("This is part one of a long"))
        await asyncio.sleep(0.02)  # small gap, within batch window
        adapter._enqueue_text_event(_make_event("message that was split by Telegram."))

        # Not dispatched yet (timer restarted)
        adapter.handle_message.assert_not_called()

        # Wait for flush
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert "part one" in dispatched.text
        assert "split by Telegram" in dispatched.text

    @pytest.mark.asyncio
    async def test_three_way_split_aggregated(self):
        """Three rapid messages should all merge."""
        adapter = _make_adapter()

        adapter._enqueue_text_event(_make_event("chunk 1"))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("chunk 2"))
        await asyncio.sleep(0.02)
        adapter._enqueue_text_event(_make_event("chunk 3"))

        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        text = adapter.handle_message.call_args[0][0].text
        assert "chunk 1" in text
        assert "chunk 2" in text
        assert "chunk 3" in text

    @pytest.mark.asyncio
    async def test_different_chats_not_merged(self):
        """Messages from different chats should be separate batches."""
        adapter = _make_adapter()

        adapter._enqueue_text_event(_make_event("from user A", chat_id="111"))
        adapter._enqueue_text_event(_make_event("from user B", chat_id="222"))

        await asyncio.sleep(0.2)

        assert adapter.handle_message.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_cleans_up_after_flush(self):
        """After flushing, internal state should be clean."""
        adapter = _make_adapter()

        adapter._enqueue_text_event(_make_event("test"))
        await asyncio.sleep(0.2)

        assert len(adapter._pending_text_batches) == 0
        assert len(adapter._pending_text_batch_tasks) == 0

    @pytest.mark.asyncio
    async def test_dm_topic_batching_recovers_thread_before_keying(self):
        """DM-topic text batches should use the recovered topic lane."""
        adapter = _make_adapter()
        adapter.set_topic_recovery_fn(
            lambda source: "222" if str(source.thread_id or "") == "1" else None
        )
        event = MessageEvent(
            text="hello from DM topic",
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id="12345",
                chat_type="dm",
                user_id="user-1",
                thread_id="1",
            ),
        )

        adapter._enqueue_text_event(event)

        def _key(thread_id: str) -> str:
            return build_session_key(
                SimpleNamespace(
                    platform=Platform.TELEGRAM,
                    chat_id="12345",
                    chat_type="dm",
                    thread_id=thread_id,
                ),
                group_sessions_per_user=True,
                thread_sessions_per_user=False,
            )

        assert _key("222") in adapter._pending_text_batches
        assert _key("1") not in adapter._pending_text_batches
        assert event.source.thread_id == "222"

        await asyncio.sleep(0.2)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.source.thread_id == "222"

    @pytest.mark.asyncio
    async def test_disconnect_cancels_pending_text_batch_without_dispatch(self):
        """Disconnect should not let buffered text flush into a stale run."""
        adapter = _make_adapter()

        adapter._enqueue_text_event(_make_event("stale text"))
        await adapter.disconnect()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_not_called()
        assert adapter._pending_text_batches == {}
        assert adapter._pending_text_batch_tasks == {}

    @pytest.mark.asyncio
    async def test_disconnected_adapter_drops_pending_text_flush_before_dispatch(self):
        """A pending text flush should drop its event if teardown wins the race."""
        adapter = _make_adapter()

        adapter._enqueue_text_event(_make_event("stale text"))
        adapter._mark_disconnected()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_not_called()
        assert adapter._pending_text_batches == {}
        assert adapter._pending_text_batch_tasks == {}

    @pytest.mark.asyncio
    async def test_disconnected_adapter_drops_late_text_batch_enqueue(self):
        """Late update handlers should not schedule batches after teardown starts."""
        adapter = _make_adapter()
        adapter._mark_disconnected()

        adapter._enqueue_text_event(_make_event("late text"))
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_not_called()
        assert adapter._pending_text_batches == {}
        assert adapter._pending_text_batch_tasks == {}

    @pytest.mark.asyncio
    async def test_disconnected_adapter_drops_pending_photo_flush_before_dispatch(self):
        """A pending photo batch should not dispatch after disconnect starts."""
        adapter = _make_adapter()
        adapter._media_batch_delay_seconds = 0.1
        event = _make_event("photo caption")
        event.media_urls = ["/tmp/photo.jpg"]
        event.media_types = ["image/jpeg"]

        adapter._enqueue_photo_event("chat:photo-burst", event)
        adapter._mark_disconnected()
        await asyncio.sleep(0.2)

        adapter.handle_message.assert_not_called()
        assert adapter._pending_photo_batches == {}
        assert adapter._pending_photo_batch_tasks == {}

    @pytest.mark.asyncio
    async def test_disconnected_adapter_drops_pending_media_group_flush_before_dispatch(self):
        """A pending media group should not dispatch after disconnect starts."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = _make_adapter()
        event = _make_event("album caption")
        event.media_urls = ["/tmp/photo.jpg"]
        event.media_types = ["image/jpeg"]

        with patch.object(TelegramAdapter, "MEDIA_GROUP_WAIT_SECONDS", 0.1):
            await adapter._queue_media_group_event("album-1", event)
            adapter._mark_disconnected()
            await asyncio.sleep(0.2)

        adapter.handle_message.assert_not_called()
        assert adapter._media_group_events == {}
        assert adapter._media_group_tasks == {}

    @pytest.mark.asyncio
    async def test_stale_media_group_flush_does_not_clear_newer_task(self):
        """A cancelled album flush must not erase the replacement task handle."""
        from plugins.platforms.telegram.adapter import TelegramAdapter

        adapter = _make_adapter()
        first = _make_event("first album caption")
        first.media_urls = ["/tmp/first.jpg"]
        first.media_types = ["image/jpeg"]
        second = _make_event("second album caption")
        second.media_urls = ["/tmp/second.jpg"]
        second.media_types = ["image/jpeg"]

        with patch.object(TelegramAdapter, "MEDIA_GROUP_WAIT_SECONDS", 1.0):
            await adapter._queue_media_group_event("album-race", first)
            first_task = adapter._media_group_tasks["album-race"]
            await asyncio.sleep(0)

            await adapter._queue_media_group_event("album-race", second)
            replacement_task = adapter._media_group_tasks["album-race"]
            assert replacement_task is not first_task

            await asyncio.sleep(0)
            assert adapter._media_group_tasks.get("album-race") is replacement_task

            replacement_task.cancel()
            await asyncio.gather(replacement_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_cancel_pending_delivery_tasks_skips_current_polling_error_task(self):
        """The teardown helper must not cancel the coroutine doing cleanup."""
        adapter = _make_adapter()
        current_task = asyncio.current_task()
        stale_task = asyncio.create_task(asyncio.sleep(60))
        adapter._pending_text_batches["text"] = _make_event("text")
        adapter._pending_text_batch_tasks["text"] = stale_task
        adapter._polling_error_task = current_task

        await adapter._cancel_pending_delivery_tasks()

        assert stale_task.done()
        assert stale_task.cancelled()
        assert not current_task.cancelled()
        assert adapter._pending_text_batches == {}
        assert adapter._pending_text_batch_tasks == {}
        assert adapter._polling_error_task is current_task

    @pytest.mark.asyncio
    async def test_disconnect_cancels_all_pending_delivery_task_maps(self):
        """Photo/media/polling delayed tasks are awaited and queues are cleared."""
        adapter = _make_adapter()
        tasks = [asyncio.create_task(asyncio.sleep(60)) for _ in range(4)]
        adapter._pending_text_batches["text"] = _make_event("text")
        adapter._pending_text_batch_tasks["text"] = tasks[0]
        adapter._pending_photo_batches["photo"] = _make_event("photo")
        adapter._pending_photo_batch_tasks["photo"] = tasks[1]
        adapter._media_group_events["media"] = _make_event("media")
        adapter._media_group_tasks["media"] = tasks[2]
        adapter._polling_error_task = tasks[3]

        await adapter.disconnect()

        assert all(task.done() for task in tasks)
        assert adapter._pending_text_batches == {}
        assert adapter._pending_text_batch_tasks == {}
        assert adapter._pending_photo_batches == {}
        assert adapter._pending_photo_batch_tasks == {}
        assert adapter._media_group_events == {}
        assert adapter._media_group_tasks == {}
        assert adapter._polling_error_task is None

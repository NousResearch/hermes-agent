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
    async def test_disconnect_cancels_all_pending_delivery_task_maps(self):
        """Photo/media/polling delayed tasks are awaited and queues are cleared."""
        adapter = _make_adapter()
        tasks = [asyncio.create_task(asyncio.sleep(0.2)) for _ in range(4)]
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


class TestBatchFlushNotCancelledByFollowUp:
    """A follow-up chunk must never cancel a dispatch that is already running.

    ``_enqueue_*`` unconditionally cancels the prior flush task, and the flush
    pops the event out of its pending buffer *before* awaiting
    ``handle_message``.  Without a shield the popped event is in no buffer, was
    never dispatched, and asyncio does not report cancelled tasks — so the
    user's message is lost silently.  The Discord adapter already shields this
    exact path (#12444); these cover the Telegram equivalents.

    ``handle_message`` must really suspend here: the production path awaits
    ``asyncio.to_thread(self._apply_topic_recovery, event)`` before it durably
    queues anything, so the cancellation window is always open.  An
    ``AsyncMock`` never suspends, which is why the existing tests miss this.
    """

    @staticmethod
    def _tracking_handler(entered, completed):
        async def _handle(event):
            label = event.text or f"<{len(event.media_urls)} media>"
            entered.append(label)
            await asyncio.sleep(0.3)
            completed.append(label)
        return _handle

    @pytest.mark.asyncio
    async def test_text_follow_up_does_not_drop_in_flight_message(self):
        adapter = _make_adapter()
        entered, completed = [], []
        adapter.handle_message = self._tracking_handler(entered, completed)

        adapter._enqueue_text_event(_make_event("first message"))
        await asyncio.sleep(0.15)  # flush fired at ~0.1s; dispatch is in flight
        adapter._enqueue_text_event(_make_event("second message"))
        await asyncio.sleep(0.8)

        assert entered == ["first message", "second message"]
        assert completed == ["first message", "second message"], (
            "the in-flight first message was cancelled and silently lost"
        )

    @pytest.mark.asyncio
    async def test_photo_follow_up_does_not_drop_in_flight_batch(self):
        adapter = _make_adapter()
        adapter._media_batch_delay_seconds = 0.1
        entered, completed = [], []
        adapter.handle_message = self._tracking_handler(entered, completed)

        first = _make_event("photo one")
        first.media_urls = ["u1"]
        first.media_types = ["image"]
        adapter._enqueue_photo_event("k", first)
        await asyncio.sleep(0.15)
        second = _make_event("photo two")
        second.media_urls = ["u2"]
        second.media_types = ["image"]
        adapter._enqueue_photo_event("k", second)
        await asyncio.sleep(0.8)

        assert completed == ["photo one", "photo two"], (
            "the in-flight photo batch was cancelled and silently lost"
        )

    @pytest.mark.asyncio
    async def test_media_group_follow_up_does_not_drop_in_flight_album(self):
        adapter = _make_adapter()
        adapter.MEDIA_GROUP_WAIT_SECONDS = 0.1
        entered, completed = [], []
        adapter.handle_message = self._tracking_handler(entered, completed)

        first = _make_event("album caption")
        first.media_urls = ["u1"]
        first.media_types = ["image"]
        adapter._media_group_events["mg1"] = first
        adapter._media_group_tasks["mg1"] = asyncio.create_task(
            adapter._flush_media_group_event("mg1")
        )
        await asyncio.sleep(0.15)
        second = _make_event("")
        second.media_urls = ["u2"]
        second.media_types = ["image"]
        adapter._media_group_events["mg1"] = second
        prior = adapter._media_group_tasks.get("mg1")
        if prior:
            prior.cancel()
        adapter._media_group_tasks["mg1"] = asyncio.create_task(
            adapter._flush_media_group_event("mg1")
        )
        await asyncio.sleep(0.8)

        assert "album caption" in completed, (
            "the in-flight album (carrying the caption) was cancelled and lost"
        )

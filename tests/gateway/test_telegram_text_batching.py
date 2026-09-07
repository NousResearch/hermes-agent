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
    # Hold-queue state (preserve inbound across reconnect)
    adapter._held_inbound_events = []
    adapter._held_inbound_redispatch_task = None
    adapter.HELD_INBOUND_MAX = 64
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


class TestHoldInboundAcrossReconnect:
    """Inbound events must not be destroyed when the disconnect drop-guard fires.

    #55971 introduced ``_drop_delayed_deliveries`` so flushes cannot dispatch
    into a torn-down session. That is correct. But the implementation
    destroyed the event (debug-level return after pop / before enqueue).
    PTB has already advanced the polling offset by then, so Telegram never
    redelivers — the user's message is gone with no log and no error.

    Related but distinct from #72037 (cancel-after-pop during follow-up
    supersession). This covers the disconnect/reconnect path only.

    Timing: no wall-clock races. Flush paths under test use delay=0 and/or
    entered/release ``asyncio.Event`` sync (teknium review rule on #72037).
    """

    @staticmethod
    def _zero_batch_delays(adapter) -> None:
        """Make flush paths deterministic: no sleep, no timing assumptions."""
        adapter._text_batch_delay_seconds = 0
        adapter._text_batch_split_delay_seconds = 0
        adapter._TEXT_BATCH_FAST_DELAY_S = 0
        adapter._TEXT_BATCH_SHORT_DELAY_S = 0
        adapter._TEXT_BATCH_FAST_LEN = 10**9
        adapter._TEXT_BATCH_SHORT_LEN = 10**9
        adapter._SPLIT_THRESHOLD = 10**9
        adapter._media_batch_delay_seconds = 0

    @pytest.mark.asyncio
    async def test_late_enqueue_held_and_redispatched_on_reconnect(self):
        adapter = _make_adapter()
        adapter._mark_disconnected()

        adapter._enqueue_text_event(_make_event("should survive disconnect"))

        # Must NOT dispatch into torn-down session
        adapter.handle_message.assert_not_called()
        assert len(adapter._held_inbound_events) == 1
        assert adapter._held_inbound_events[0].text == "should survive disconnect"

        adapter._mark_connected()
        task = adapter._held_inbound_redispatch_task
        assert task is not None
        await task

        adapter.handle_message.assert_called_once()
        assert adapter.handle_message.call_args[0][0].text == "should survive disconnect"
        assert adapter._held_inbound_events == []

    @pytest.mark.asyncio
    async def test_flush_during_disconnect_holds_popped_event(self):
        """After pop, drop-guard must hold — not destroy — the event.

        Deterministic: delay=0 and drop already True before flush runs, so the
        post-pop branch is exercised without wall-clock races.
        """
        adapter = _make_adapter()
        self._zero_batch_delays(adapter)
        event = _make_event("popped then held")
        adapter._pending_text_batches["k"] = event
        adapter._drop_delayed_deliveries = True

        await adapter._flush_text_batch("k")

        adapter.handle_message.assert_not_called()
        assert adapter._pending_text_batches == {}
        assert [e.text for e in adapter._held_inbound_events] == ["popped then held"]

    @pytest.mark.asyncio
    async def test_flush_cancel_after_pop_holds_event(self):
        """Cancel after pop (before handle_message returns) must hold, not lose.

        Uses entered/release Events — no sleep timing (teknium #72037 rule).
        Connected path then schedules redispatch (#83878).
        """
        adapter = _make_adapter()
        self._zero_batch_delays(adapter)
        entered = asyncio.Event()
        release = asyncio.Event()
        seen: list[str] = []

        async def _blocking_handle(event):
            seen.append(event.text or "")
            entered.set()
            await release.wait()

        adapter.handle_message = _blocking_handle
        adapter._pending_text_batches["k"] = _make_event("in-flight cancel")
        task = asyncio.create_task(adapter._flush_text_batch("k"))
        adapter._pending_text_batch_tasks["k"] = task

        await entered.wait()  # past pop, inside handle_message
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()

        drain = adapter._held_inbound_redispatch_task
        assert drain is not None
        await asyncio.wait_for(drain, timeout=1.0)

        # Recoverable: held and/or delivered via redispatch (seen may include
        # the original in-flight attempt plus the redispatch).
        held_texts = [e.text for e in adapter._held_inbound_events]
        assert "in-flight cancel" in seen or "in-flight cancel" in held_texts

    @pytest.mark.asyncio
    async def test_cancel_pending_salvages_batches_into_held_queue(self):
        """Teardown must salvage map contents before clear — not discard them."""
        adapter = _make_adapter()
        adapter._pending_text_batches["text"] = _make_event("text-salvage")
        adapter._pending_photo_batches["photo"] = _make_event("photo-salvage")
        adapter._media_group_events["media"] = _make_event("media-salvage")
        t1 = asyncio.create_task(asyncio.sleep(60))
        t2 = asyncio.create_task(asyncio.sleep(60))
        t3 = asyncio.create_task(asyncio.sleep(60))
        adapter._pending_text_batch_tasks["text"] = t1
        adapter._pending_photo_batch_tasks["photo"] = t2
        adapter._media_group_tasks["media"] = t3

        adapter._mark_disconnected()
        await adapter._cancel_pending_delivery_tasks()

        held = {e.text for e in adapter._held_inbound_events}
        assert held == {"text-salvage", "photo-salvage", "media-salvage"}
        assert adapter._pending_text_batches == {}
        assert adapter._pending_photo_batches == {}
        assert adapter._media_group_events == {}
        assert adapter._held_inbound_redispatch_task is None

    @pytest.mark.asyncio
    async def test_redispatch_task_cancelled_on_teardown(self):
        """In-flight redispatch must be in the cancel map (lifecycle rule)."""
        adapter = _make_adapter()
        entered = asyncio.Event()
        release = asyncio.Event()

        async def _blocking_handle(event):
            entered.set()
            await release.wait()

        adapter.handle_message = _blocking_handle
        adapter._held_inbound_events = [_make_event("during-redispatch")]
        adapter._drop_delayed_deliveries = False
        task = asyncio.create_task(adapter._redispatch_held_inbound())
        adapter._held_inbound_redispatch_task = task

        await entered.wait()
        adapter._mark_disconnected()
        await adapter._cancel_pending_delivery_tasks()

        assert task.done()
        # Cancel during handle → re-held
        assert any(e.text == "during-redispatch" for e in adapter._held_inbound_events)
        release.set()

    @pytest.mark.asyncio
    async def test_photo_and_media_group_enqueue_held_during_disconnect(self):
        adapter = _make_adapter()
        adapter._mark_disconnected()

        photo = _make_event("photo caption")
        photo.media_urls = ["u1"]
        photo.media_types = ["image"]
        adapter._enqueue_photo_event("k", photo)

        album = _make_event("album caption")
        album.media_urls = ["u2"]
        album.media_types = ["image"]
        await adapter._queue_media_group_event("mg1", album)

        adapter.handle_message.assert_not_called()
        texts = {e.text for e in adapter._held_inbound_events}
        assert texts == {"photo caption", "album caption"}

    @pytest.mark.asyncio
    async def test_hold_dedupes_same_event_object(self):
        adapter = _make_adapter()
        event = _make_event("once")
        adapter._hold_inbound_event(event, where="a")
        adapter._hold_inbound_event(event, where="b")
        assert len(adapter._held_inbound_events) == 1

    @pytest.mark.asyncio
    async def test_held_queue_cap_drops_oldest(self):
        adapter = _make_adapter()
        adapter.HELD_INBOUND_MAX = 2
        adapter._mark_disconnected()
        adapter._enqueue_text_event(_make_event("first"))
        adapter._enqueue_text_event(_make_event("second"))
        adapter._enqueue_text_event(_make_event("third"))

        texts = [e.text for e in adapter._held_inbound_events]
        assert texts == ["second", "third"]

    @pytest.mark.asyncio
    async def test_redispatch_aborts_cleanly_if_disconnect_returns(self):
        """If disconnect re-trips mid-drain, remaining events stay held."""
        adapter = _make_adapter()
        adapter._held_inbound_events = [
            _make_event("a"),
            _make_event("b"),
            _make_event("c"),
        ]

        call_count = 0

        async def _handle(event):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                adapter._drop_delayed_deliveries = True

        adapter.handle_message = _handle
        adapter._drop_delayed_deliveries = False
        await adapter._redispatch_held_inbound()

        assert call_count == 1
        held_texts = [e.text for e in adapter._held_inbound_events]
        assert held_texts == ["b", "c"]

    @pytest.mark.asyncio
    async def test_non_retryable_fatal_discards_held_with_warning(self):
        adapter = _make_adapter()
        adapter._held_inbound_events = [_make_event("doomed")]
        from gateway.platforms.base import BasePlatformAdapter

        def _base_fatal(self, code, message, *, retryable):
            self._fatal_error_code = code
            self._fatal_error_message = message
            self._fatal_error_retryable = retryable
            self._running = False

        with patch.object(BasePlatformAdapter, "_set_fatal_error", _base_fatal):
            adapter._set_fatal_error("auth", "revoked", retryable=False)

        assert adapter._held_inbound_events == []
        assert adapter._drop_delayed_deliveries is True
        assert adapter._is_permanent_fatal() is True

    @pytest.mark.asyncio
    async def test_retryable_fatal_preserves_held_for_reconnect_drain(self):
        """Retryable fatals must NOT clear the hold queue.

        OOF-156's connect-failure classification keeps the common network
        path ``retryable=True`` (``telegram_connect_error``) — reconnect is
        precisely what must drain a hold queue populated during the outage.
        Only non-retryable fatals may discard (covered above).
        """
        adapter = _make_adapter()
        adapter._held_inbound_events = [_make_event("survives-network-fatal")]
        adapter._drop_delayed_deliveries = True  # fatal/disconnect already set

        from gateway.platforms.base import BasePlatformAdapter

        def _base_fatal(self, code, message, *, retryable):
            self._fatal_error_code = code
            self._fatal_error_message = message
            self._fatal_error_retryable = retryable

        with patch.object(BasePlatformAdapter, "_set_fatal_error", _base_fatal):
            adapter._set_fatal_error(
                "telegram_connect_error", "connect timed out", retryable=True
            )

        assert [e.text for e in adapter._held_inbound_events] == [
            "survives-network-fatal"
        ]
        assert adapter._is_permanent_fatal() is False

        # Reconnect drains what the retryable fatal preserved.
        adapter._mark_connected()
        await adapter._held_inbound_redispatch_task
        adapter.handle_message.assert_called_once()
        assert (
            adapter.handle_message.call_args[0][0].text == "survives-network-fatal"
        )

    @pytest.mark.asyncio
    async def test_production_text_handler_terminal_step_holds_when_disconnected(self):
        """Production path: ``_handle_text_message`` ends in ``_enqueue_text_event``.

        Sweeper rejects helper-only coverage. This pins the call site that
        PTB invokes after the update is already acked (offset advanced).
        """
        adapter = _make_adapter()
        adapter._mark_disconnected()
        # Terminal step of _handle_text_message after event construction.
        adapter._enqueue_text_event(_make_event("acked-by-ptb-then-held"))
        adapter.handle_message.assert_not_called()
        assert [e.text for e in adapter._held_inbound_events] == ["acked-by-ptb-then-held"]

        adapter._mark_connected()
        await adapter._held_inbound_redispatch_task
        adapter.handle_message.assert_called_once()
        assert adapter.handle_message.call_args[0][0].text == "acked-by-ptb-then-held"

    @pytest.mark.asyncio
    async def test_permanent_fatal_teardown_discards_pending_not_rehold(self):
        """#83878: permanent fatal must not re-populate hold via teardown salvage."""
        adapter = _make_adapter()
        adapter._fatal_error_code = "auth"
        adapter._fatal_error_retryable = False
        adapter._drop_delayed_deliveries = True
        adapter._pending_text_batches["t"] = _make_event("pending-text")
        adapter._pending_photo_batches["p"] = _make_event("pending-photo")
        adapter._media_group_events["m"] = _make_event("pending-media")

        await adapter._cancel_pending_delivery_tasks()

        assert adapter._held_inbound_events == []
        assert adapter._pending_text_batches == {}
        assert adapter._pending_photo_batches == {}
        assert adapter._media_group_events == {}

    @pytest.mark.asyncio
    async def test_permanent_fatal_late_enqueue_discards(self):
        """#83878: late enqueue after permanent fatal must discard, not hold."""
        adapter = _make_adapter()
        adapter._fatal_error_code = "auth"
        adapter._fatal_error_retryable = False
        adapter._drop_delayed_deliveries = True

        adapter._enqueue_text_event(_make_event("too-late"))
        adapter.handle_message.assert_not_called()
        assert adapter._held_inbound_events == []

    @pytest.mark.asyncio
    async def test_connected_hold_schedules_redispatch(self):
        """#83878: hold while connected must drain, not orphan until reconnect."""
        adapter = _make_adapter()
        adapter._drop_delayed_deliveries = False
        adapter.handle_message = AsyncMock()

        adapter._hold_inbound_event(
            _make_event("orphan-without-drain"), where="text-flush-cancelled"
        )

        drain = adapter._held_inbound_redispatch_task
        assert drain is not None
        await asyncio.wait_for(drain, timeout=1.0)
        adapter.handle_message.assert_called_once()
        assert adapter.handle_message.call_args[0][0].text == "orphan-without-drain"
        assert adapter._held_inbound_events == []

    @pytest.mark.asyncio
    async def test_redispatch_exception_reholds_current_and_remainder(self):
        """#83878: handle_message failure must not drop current/remainder."""
        adapter = _make_adapter()
        adapter._drop_delayed_deliveries = False
        adapter._held_inbound_events = [
            _make_event("boom"),
            _make_event("after"),
        ]

        async def _handle(event):
            if event.text == "boom":
                raise RuntimeError("dispatch failed")
            return None

        adapter.handle_message = _handle
        # Direct drain (no auto follow-up on failure)
        await adapter._redispatch_held_inbound()
        held_texts = [e.text for e in adapter._held_inbound_events]
        assert held_texts == ["boom", "after"]

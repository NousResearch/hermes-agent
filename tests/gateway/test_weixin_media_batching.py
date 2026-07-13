"""Tests for WeixinAdapter media/file burst batching.

When a Weixin user sends multiple files (docx, images, ...) in rapid
succession, each arrives as an independent inbound iLink event. Dispatching
every one immediately hits the gateway's interrupt-recursion guard and the
session locks up.

Media events are routed through the same session-scoped debounce batcher as
text (``_enqueue_text_event`` / ``_flush_text_batch``), so a whole burst — and
any accompanying caption — coalesces into a single ``handle_message`` call.
This module covers the media-specific behaviour, config parsing via
``_coerce_float_extra`` (config.yaml-driven, no env var), shutdown cleanup in
``disconnect()``, and the shield guard that keeps a follow-up file from
cancelling an in-flight dispatch.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SessionSource
from gateway.platforms.weixin import WeixinAdapter


def _make_adapter(media_delay: float = 0.1, text_delay: float = 5.0) -> WeixinAdapter:
    """Minimal WeixinAdapter wired for batching tests.

    ``text_delay`` is deliberately long so that when a media batch is pending
    the test only completes if the media window (``media_delay``) is the one
    actually used by ``_flush_text_batch``.
    """
    adapter = object.__new__(WeixinAdapter)
    adapter.platform = Platform.WEIXIN
    adapter.config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._text_batch_delay_seconds = text_delay
    adapter._text_batch_split_delay_seconds = text_delay
    adapter._media_batch_delay_seconds = media_delay
    adapter.handle_message = AsyncMock()
    return adapter


def _media_event(path: str, chat_id: str = "weixin-dm", text: str = "") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.DOCUMENT,
        source=SessionSource(platform=Platform.WEIXIN, chat_id=chat_id, chat_type="dm"),
        media_urls=[path],
        media_types=["application/octet-stream"],
    )


def _text_event(text: str, chat_id: str = "weixin-dm") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.WEIXIN, chat_id=chat_id, chat_type="dm"),
    )


def _photo_event(path: str, chat_id: str = "weixin-dm", text: str = "") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.PHOTO,
        source=SessionSource(platform=Platform.WEIXIN, chat_id=chat_id, chat_type="dm"),
        media_urls=[path],
        media_types=["image/jpeg"],
    )


def _typed_event(message_type: MessageType) -> MessageEvent:
    return MessageEvent(
        text="",
        message_type=message_type,
        source=SessionSource(platform=Platform.WEIXIN, chat_id="c", chat_type="dm"),
    )


class TestWeixinMediaBurstBatching:
    @pytest.mark.asyncio
    async def test_single_file_dispatches_once_after_window(self):
        adapter = _make_adapter()
        adapter._enqueue_text_event(_media_event("/tmp/a.docx"))

        adapter.handle_message.assert_not_called()  # debounced
        await asyncio.sleep(0.25)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.media_urls == ["/tmp/a.docx"]

    @pytest.mark.asyncio
    async def test_rapid_files_merge_into_single_dispatch(self):
        """8 files sent within the window become one handle_message call."""
        adapter = _make_adapter()

        for i in range(8):
            adapter._enqueue_text_event(_media_event(f"/tmp/file-{i}.docx"))
            await asyncio.sleep(0.01)  # within the 0.1s media window

        adapter.handle_message.assert_not_called()
        await asyncio.sleep(0.25)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert len(dispatched.media_urls) == 8
        assert dispatched.media_urls[0] == "/tmp/file-0.docx"
        assert dispatched.media_urls[-1] == "/tmp/file-7.docx"

    @pytest.mark.asyncio
    async def test_caption_and_files_coalesce_into_one_turn(self):
        """A text caption + a file burst in the same session dispatch together.

        This is the point of routing media through the text batcher: the
        caption is not split off into its own separate turn.
        """
        adapter = _make_adapter()

        adapter._enqueue_text_event(_text_event("here are the files"))
        await asyncio.sleep(0.01)
        for i in range(3):
            adapter._enqueue_text_event(_media_event(f"/tmp/f{i}.docx"))
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.25)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert "here are the files" in dispatched.text
        assert len(dispatched.media_urls) == 3

    @pytest.mark.asyncio
    async def test_media_window_used_not_text_window(self):
        """With a pending media batch, the flush must use the media delay.

        text_delay is 5s; if the flush wrongly used it, this would time out
        rather than flush inside 0.25s.
        """
        adapter = _make_adapter(media_delay=0.1, text_delay=5.0)
        adapter._enqueue_text_event(_media_event("/tmp/a.docx"))

        await asyncio.sleep(0.25)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_different_sessions_dispatch_separately(self):
        adapter = _make_adapter()

        adapter._enqueue_text_event(_media_event("/tmp/a.docx", chat_id="chat-a"))
        adapter._enqueue_text_event(_media_event("/tmp/b.docx", chat_id="chat-b"))

        await asyncio.sleep(0.25)
        assert adapter.handle_message.call_count == 2

    @pytest.mark.asyncio
    async def test_state_cleaned_up_after_flush(self):
        adapter = _make_adapter()
        adapter._enqueue_text_event(_media_event("/tmp/a.docx"))
        await asyncio.sleep(0.25)

        assert adapter._pending_text_batches == {}
        assert adapter._pending_text_batch_tasks == {}


class TestWeixinMediaBatchShutdown:
    """Regression for review comment: pending flush timers must be torn down
    in ``disconnect()`` so none can fire ``handle_message`` after shutdown."""

    @pytest.mark.asyncio
    async def test_disconnect_cancels_pending_batch(self):
        # Long media window so the batch is still pending at disconnect.
        adapter = _make_adapter(media_delay=30.0)
        adapter._token = "test-token"
        adapter._running = True
        adapter._poll_task = None
        adapter._poll_session = None
        adapter._send_session = None
        adapter._release_platform_lock = MagicMock()
        adapter._mark_disconnected = MagicMock()

        adapter._enqueue_text_event(_media_event("/tmp/a.docx"))
        assert adapter._pending_text_batch_tasks  # armed

        await adapter.disconnect()

        adapter.handle_message.assert_not_called()
        assert adapter._pending_text_batches == {}
        assert adapter._pending_text_batch_tasks == {}
        adapter._mark_disconnected.assert_called_once()

    @pytest.mark.asyncio
    async def test_followup_file_does_not_cancel_inflight_dispatch(self):
        """A file arriving while handle_message is mid-flight must not abort it.

        ``_enqueue_text_event`` cancels the prior flush task on every new
        event; without ``asyncio.shield`` around handle_message that cancel
        would propagate into the running agent turn. File turns run long, so
        this window is wide in practice.
        """
        adapter = _make_adapter()

        handle_started = asyncio.Event()
        release_handle = asyncio.Event()
        first_cancelled = asyncio.Event()
        first_completed = asyncio.Event()
        calls = [0]

        async def slow_handle(event):
            calls[0] += 1
            if calls[0] == 1:
                handle_started.set()
                try:
                    await release_handle.wait()
                    first_completed.set()
                except asyncio.CancelledError:
                    first_cancelled.set()
                    raise

        adapter.handle_message = slow_handle

        adapter._enqueue_text_event(_media_event("/tmp/a.docx"))
        await asyncio.wait_for(handle_started.wait(), timeout=1.0)

        # Follow-up file cancels batch 1's flush task, which is awaiting
        # inside handle_message.
        adapter._enqueue_text_event(_media_event("/tmp/b.docx"))
        await asyncio.sleep(0.05)

        assert not first_cancelled.is_set(), (
            "in-flight handle_message was cancelled by a follow-up file — "
            "asyncio.shield is missing"
        )

        release_handle.set()
        await asyncio.wait_for(first_completed.wait(), timeout=1.0)

        for task in list(adapter._pending_text_batch_tasks.values()):
            task.cancel()
        await asyncio.sleep(0.01)


class TestWeixinMediaRoutingDecision:
    """Only file/photo bursts are debounced. Voice/audio/video dispatch
    immediately so their message_type-keyed downstream handling (STT,
    voice-reply routing) is preserved and they incur no debounce latency."""

    @pytest.mark.parametrize(
        "mtype,expected",
        [
            (MessageType.DOCUMENT, True),
            (MessageType.PHOTO, True),
            (MessageType.VOICE, False),
            (MessageType.AUDIO, False),
            (MessageType.VIDEO, False),
            (MessageType.TEXT, False),
        ],
    )
    def test_should_batch_media_by_type(self, mtype, expected):
        adapter = _make_adapter(media_delay=2.0)
        assert adapter._should_batch_media(_typed_event(mtype)) is expected

    def test_zero_delay_never_batches(self):
        adapter = _make_adapter(media_delay=0.0)
        assert adapter._should_batch_media(_typed_event(MessageType.DOCUMENT)) is False


class TestWeixinMediaMergeReclassifies:
    """When media merges into a pending text batch, the merged event's
    message_type must reflect the media so downstream handling runs."""

    @pytest.mark.asyncio
    async def test_photo_merged_into_text_batch_becomes_photo(self):
        adapter = _make_adapter()
        adapter._enqueue_text_event(_text_event("look at this"))
        await asyncio.sleep(0.01)
        adapter._enqueue_text_event(_photo_event("/tmp/p.jpg"))

        await asyncio.sleep(0.25)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.message_type == MessageType.PHOTO
        assert "look at this" in dispatched.text
        assert dispatched.media_urls == ["/tmp/p.jpg"]

    @pytest.mark.asyncio
    async def test_document_merged_into_text_batch_becomes_document(self):
        adapter = _make_adapter()
        adapter._enqueue_text_event(_text_event("here"))
        await asyncio.sleep(0.01)
        adapter._enqueue_text_event(_media_event("/tmp/a.docx"))

        await asyncio.sleep(0.25)

        adapter.handle_message.assert_called_once()
        dispatched = adapter.handle_message.call_args[0][0]
        assert dispatched.message_type == MessageType.DOCUMENT


class TestWeixinMediaBatchConfig:
    """``media_batch_delay_seconds`` is parsed via ``_coerce_float_extra``:
    config.yaml-driven, and invalid/non-finite/negative values fall back to
    the default instead of breaking adapter construction."""

    def test_zero_disables_media_batching(self):
        adapter = WeixinAdapter(
            PlatformConfig(
                enabled=True,
                token="test-token",
                extra={"account_id": "test-account", "media_batch_delay_seconds": 0},
            )
        )
        assert adapter._media_batch_delay_seconds == 0.0

    def test_invalid_value_falls_back_to_default(self):
        adapter = WeixinAdapter(
            PlatformConfig(
                enabled=True,
                token="test-token",
                extra={"account_id": "test-account", "media_batch_delay_seconds": "abc"},
            )
        )
        assert adapter._media_batch_delay_seconds == 2.0

    def test_negative_value_falls_back_to_default(self):
        adapter = WeixinAdapter(
            PlatformConfig(
                enabled=True,
                token="test-token",
                extra={"account_id": "test-account", "media_batch_delay_seconds": -1},
            )
        )
        assert adapter._media_batch_delay_seconds == 2.0

    def test_default_when_unset(self):
        adapter = WeixinAdapter(
            PlatformConfig(
                enabled=True,
                token="test-token",
                extra={"account_id": "test-account"},
            )
        )
        assert adapter._media_batch_delay_seconds == 2.0

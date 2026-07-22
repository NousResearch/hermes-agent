"""Late regressions for Telegram startup media buffering."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from plugins.platforms.telegram.adapter import TelegramAdapter


class _DummyTask:
    def __init__(self):
        self.cancelled = False

    def done(self):
        return False

    def cancel(self):
        self.cancelled = True


def _source(*, profile=None, thread_id=None) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="100",
        chat_type="dm",
        user_id="1",
        user_name="Tester",
        thread_id=thread_id,
        profile=profile,
    )


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter.config = SimpleNamespace(
        extra={
            "group_sessions_per_user": True,
            "thread_sessions_per_user": False,
        }
    )
    adapter._pending_messages = {}
    adapter._pending_photo_batches = {}
    adapter._pending_photo_batch_tasks = {}
    adapter._pending_text_batches = {}
    adapter._pending_text_batch_tasks = {}
    adapter._media_group_events = {}
    adapter._media_group_tasks = {}
    adapter._media_downloads_in_progress_by_session = {}
    adapter._drop_delayed_deliveries = False
    adapter._topic_recovery_fn = None
    adapter.handle_message = AsyncMock()
    return adapter


def _patch_media_event_build(monkeypatch, adapter, source, msg_type):
    monkeypatch.setattr(adapter, "_is_user_authorized_from_message", lambda _msg: True)
    monkeypatch.setattr(adapter, "_should_process_message", lambda _msg: True)
    monkeypatch.setattr(adapter, "_media_message_type", lambda _msg: msg_type)
    monkeypatch.setattr(
        adapter,
        "_build_message_event",
        lambda _msg, actual_type, update_id=None: MessageEvent(
            text="",
            message_type=actual_type,
            source=source,
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_apply_telegram_group_observe_attribution",
        lambda event: event,
    )


def _video_message(video=None, document=None, *, thread_id=None):
    return SimpleNamespace(
        message_id=1,
        text="",
        caption=None,
        photo=None,
        video=video,
        audio=None,
        voice=None,
        sticker=None,
        document=document,
        media_group_id=None,
        chat=SimpleNamespace(id=100, type="private", title=None, full_name="Tester"),
        from_user=SimpleNamespace(id=1, full_name="Tester"),
        message_thread_id=thread_id,
        reply_text=AsyncMock(),
    )


def test_forwarded_text_batch_is_startup_pending():
    adapter = _make_adapter()
    source = _source()
    key = build_session_key(source)
    adapter._pending_text_batches[key] = MessageEvent(
        text="[Forwarded message | From: Alice]\n\nhello",
        message_type=MessageType.TEXT,
        source=source,
    )
    assert adapter.has_startup_media_pending(key) is True


def test_metadata_empty_forward_header_is_startup_pending():
    adapter = _make_adapter()
    source = _source()
    key = build_session_key(source)
    adapter._pending_text_batches[key] = MessageEvent(
        text="[Forwarded message]\n\nhello",
        message_type=MessageType.TEXT,
        source=source,
    )
    assert adapter.has_startup_media_pending(key) is True


def test_pop_consumes_forwarded_debounce_and_cancels_flush():
    adapter = _make_adapter()
    source = _source()
    key = build_session_key(source)
    task = _DummyTask()
    adapter._pending_text_batches[key] = MessageEvent(
        text="[Forwarded message | From: Alice]\n\nhello",
        message_type=MessageType.TEXT,
        source=source,
    )
    adapter._pending_text_batch_tasks[key] = task

    event = adapter.pop_startup_media_event(key)

    assert event is not None
    assert event.text.startswith("[Forwarded message | From: Alice]")
    assert key not in adapter._pending_text_batches
    assert key not in adapter._pending_text_batch_tasks
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_forwarded_debounce_uses_profile_scoped_startup_key(monkeypatch):
    adapter = _make_adapter()
    source = _source(profile="secondary")
    expected = build_session_key(source, profile="secondary")
    monkeypatch.setattr(adapter, "_flush_text_batch", AsyncMock())

    adapter._enqueue_text_event(
        MessageEvent(
            text="hello",
            message_type=MessageType.TEXT,
            source=source,
            forward_origin={"type": "user", "sender_name": "Alice"},
        )
    )
    try:
        assert expected in adapter._pending_text_batches
        assert adapter.has_startup_media_pending(expected) is True
        event = adapter.pop_startup_media_event(expected)
        assert event is not None
        assert event.text.startswith("[Forwarded message | From: Alice]")
        await asyncio.sleep(0)
    finally:
        task = adapter._pending_text_batch_tasks.pop(expected, None)
        if task is not None and not task.done():
            task.cancel()


@pytest.mark.asyncio
async def test_media_counter_uses_recovered_topic_key(monkeypatch):
    adapter = _make_adapter()
    source = _source(thread_id="raw-topic")
    recovered_source = _source(thread_id="recovered-topic")
    recovered_key = build_session_key(recovered_source)
    adapter._topic_recovery_fn = lambda _source: "recovered-topic"
    file_obj = SimpleNamespace(
        file_path="clip.mp4",
        download_as_bytearray=AsyncMock(return_value=bytearray(b"video")),
    )
    video = SimpleNamespace(file_size=128, get_file=AsyncMock(return_value=file_obj))
    msg = _video_message(video=video, thread_id="raw-topic")
    _patch_media_event_build(monkeypatch, adapter, source, MessageType.VIDEO)
    seen = []
    original_track = adapter._track_media_download_start

    def track(event):
        key = original_track(event)
        seen.append(key)
        return key

    monkeypatch.setattr(adapter, "_track_media_download_start", track)
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.cache_video_from_bytes",
        lambda *_args, **_kwargs: "/tmp/recovered-video.mp4",
    )

    await adapter._handle_media_message(SimpleNamespace(message=msg, update_id=10), MagicMock())

    assert seen == [recovered_key]
    delivered_event = adapter.handle_message.await_args.args[0]
    assert delivered_event.source.thread_id == "recovered-topic"


@pytest.mark.asyncio
async def test_video_counter_is_live_until_download_finishes(monkeypatch):
    adapter = _make_adapter()
    source = _source()
    key = build_session_key(source)
    started = asyncio.Event()
    release = asyncio.Event()

    async def download():
        started.set()
        await release.wait()
        return bytearray(b"video")

    file_obj = SimpleNamespace(file_path="clip.mp4", download_as_bytearray=download)
    video = SimpleNamespace(file_size=128, get_file=AsyncMock(return_value=file_obj))
    msg = _video_message(video=video)
    _patch_media_event_build(monkeypatch, adapter, source, MessageType.VIDEO)
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.cache_video_from_bytes",
        lambda *_args, **_kwargs: "/tmp/video.mp4",
    )

    task = asyncio.create_task(
        adapter._handle_media_message(SimpleNamespace(message=msg, update_id=11), MagicMock())
    )
    await started.wait()
    assert adapter._media_downloads_in_progress_by_session[key] == 1
    release.set()
    await task
    assert key not in adapter._media_downloads_in_progress_by_session


@pytest.mark.asyncio
async def test_video_document_counter_is_live_through_delivery(monkeypatch):
    adapter = _make_adapter()
    adapter._max_doc_bytes = 1024 * 1024
    source = _source()
    key = build_session_key(source)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handle(_event):
        entered.set()
        await release.wait()

    adapter.handle_message = AsyncMock(side_effect=handle)
    file_obj = SimpleNamespace(
        file_path="clip.mp4",
        download_as_bytearray=AsyncMock(return_value=bytearray(b"video-doc")),
    )
    document = SimpleNamespace(
        file_name="clip.mp4",
        mime_type="video/mp4",
        file_size=128,
        get_file=AsyncMock(return_value=file_obj),
    )
    msg = _video_message(document=document)
    _patch_media_event_build(monkeypatch, adapter, source, MessageType.DOCUMENT)
    monkeypatch.setattr(
        "plugins.platforms.telegram.adapter.cache_video_from_bytes",
        lambda *_args, **_kwargs: "/tmp/video-document.mp4",
    )

    task = asyncio.create_task(
        adapter._handle_media_message(SimpleNamespace(message=msg, update_id=12), MagicMock())
    )
    await entered.wait()
    assert adapter._media_downloads_in_progress_by_session[key] == 1
    release.set()
    await task
    assert key not in adapter._media_downloads_in_progress_by_session
    adapter.handle_message.assert_awaited_once()

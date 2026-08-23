"""Regression tests for WhatsApp bridge polling."""

import asyncio
import contextlib
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _AsyncResponse:
    status = 200

    def __init__(self, payload=None, entered=None, release=None):
        self._payload = [{"messageId": "media-1"}] if payload is None else payload
        self._entered = entered
        self._release = release

    async def __aenter__(self):
        if self._entered is not None:
            self._entered.set()
        if self._release is not None:
            await self._release.wait()
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._payload


def _message_event(message_type, *, message_id, text=""):
    return MessageEvent(
        text=text,
        message_type=message_type,
        message_id=message_id,
        media_urls=[f"/tmp/{message_id}.jpg"] if message_type == MessageType.PHOTO else [],
        media_types=["image/jpeg"] if message_type == MessageType.PHOTO else [],
        source=SessionSource(
            platform=Platform.WHATSAPP,
            chat_id="chat-1",
            chat_type="dm",
            user_id="user-1",
            user_name="tester",
        ),
    )


async def _cancel_tasks(tasks):
    tasks = [task for task in tasks if task is not None]
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_same_session_photos_start_one_real_handler_and_queue_second():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(
        PlatformConfig(enabled=True, typing_indicator=False, extra={"session_name": "test"})
    )
    first = _message_event(MessageType.PHOTO, message_id="photo-1", text="first")
    second = _message_event(MessageType.PHOTO, message_id="photo-2", text="second")
    adapter._build_message_event = AsyncMock(side_effect=[first, second])
    adapter._send_read_receipt = AsyncMock()
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()

    async def blocked_handler(_event):
        handler_started.set()
        await release_handler.wait()

    handler = AsyncMock(side_effect=blocked_handler)
    adapter.set_message_handler(handler)

    try:
        await asyncio.gather(
            adapter._ingest_message({"messageId": "photo-1"}),
            adapter._ingest_message({"messageId": "photo-2"}),
        )
        await asyncio.wait_for(handler_started.wait(), timeout=1)

        session_key = build_session_key(first.source)
        assert handler.await_count == 1
        assert len(adapter._session_tasks) == 1
        assert not adapter._session_tasks[session_key].done()
        assert adapter._pending_messages[session_key] is second
        assert adapter._pending_messages[session_key].media_urls == ["/tmp/photo-2.jpg"]
    finally:
        release_handler.set()
        await _cancel_tasks(list(adapter._background_tasks))


@pytest.mark.asyncio
async def test_intentional_poll_cancellation_does_not_schedule_replacement():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter._running = True
    adapter._shutting_down = False
    adapter._poll_task = None
    adapter._poll_restart_task = None

    async def blocked_poll():
        await asyncio.Event().wait()

    adapter._poll_messages = blocked_poll
    adapter._start_polling()
    poll_task = adapter._poll_task
    await asyncio.sleep(0)

    adapter._shutting_down = True
    adapter._running = False
    poll_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await poll_task
    await asyncio.sleep(0)

    assert adapter._poll_restart_task is None


def test_poll_task_done_consumes_exception_during_shutdown():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter._running = False
    adapter._shutting_down = True
    task = MagicMock()
    task.cancelled.return_value = False
    task.exception.return_value = RuntimeError("late poll failure")

    adapter._poll_task_done(task)

    task.exception.assert_called_once_with()


@pytest.mark.asyncio
async def test_text_ingestion_uses_batch_enqueue_not_direct_handle_message():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    event = _message_event(MessageType.TEXT, message_id="text-1", text="hello")
    adapter._build_message_event = AsyncMock(return_value=event)
    adapter._send_read_receipt = AsyncMock()
    adapter._enqueue_text_event = MagicMock()
    adapter.handle_message = AsyncMock()

    await adapter._ingest_message({"messageId": "text-1"})
    await asyncio.sleep(0)

    adapter._enqueue_text_event.assert_called_once_with(event)
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_poll_task_crash_is_logged_and_restarted_while_running(caplog):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter._running = True
    adapter._shutting_down = False
    adapter._poll_task = None
    adapter._poll_restart_delay = 0
    replacement_started = asyncio.Event()

    async def poll_messages():
        if not replacement_started.is_set():
            replacement_started.set()
            raise RuntimeError("poll loop crashed")
        await asyncio.Event().wait()

    adapter._poll_messages = poll_messages

    try:
        with caplog.at_level(
            logging.ERROR, logger="plugins.platforms.whatsapp.adapter"
        ):
            adapter._start_polling()
            crashed_task = adapter._poll_task
            await asyncio.wait_for(replacement_started.wait(), timeout=1)

            async def replacement_is_running():
                while adapter._poll_task is crashed_task:
                    await asyncio.sleep(0)
                return adapter._poll_task

            replacement_task = await asyncio.wait_for(
                replacement_is_running(), timeout=1
            )

        assert replacement_task is not crashed_task
        assert not replacement_task.done()
        assert "poll loop crashed" in caplog.text
    finally:
        adapter._running = False
        adapter._shutting_down = True
        poll_task = adapter._poll_task
        if poll_task is not None and not poll_task.done():
            poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poll_task


@pytest.mark.asyncio
async def test_poll_task_unexpected_cancellation_is_logged_and_restarted(caplog):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter._running = True
    adapter._shutting_down = False
    adapter._poll_task = None
    adapter._poll_restart_task = None
    adapter._poll_restart_delay = 0
    starts = 0
    replacement_started = asyncio.Event()

    async def poll_messages():
        nonlocal starts
        starts += 1
        if starts > 1:
            replacement_started.set()
        await asyncio.Event().wait()

    adapter._poll_messages = poll_messages

    try:
        with caplog.at_level(
            logging.ERROR, logger="plugins.platforms.whatsapp.adapter"
        ):
            adapter._start_polling()
            await asyncio.sleep(0)
            adapter._poll_task.cancel()
            await asyncio.wait_for(replacement_started.wait(), timeout=1)

        assert "cancelled unexpectedly" in caplog.text
        assert not adapter._poll_task.done()
    finally:
        adapter._running = False
        adapter._shutting_down = True
        adapter._poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await adapter._poll_task


@pytest.mark.asyncio
async def test_poll_task_missing_http_session_exits_loudly_and_retries(caplog):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter._running = True
    adapter._shutting_down = False
    adapter._poll_task = None
    adapter._poll_restart_task = None
    adapter._poll_restart_delay = 0.01
    adapter._http_session = None
    starts = 0
    original_poll = adapter._poll_messages

    async def counted_poll():
        nonlocal starts
        starts += 1
        await original_poll()

    adapter._poll_messages = counted_poll

    try:
        with caplog.at_level(
            logging.ERROR, logger="plugins.platforms.whatsapp.adapter"
        ):
            adapter._start_polling()

            async def retried_more_than_once():
                while starts < 3:
                    await asyncio.sleep(0)

            await asyncio.wait_for(retried_more_than_once(), timeout=1)

        assert "missing HTTP session" in caplog.text
    finally:
        adapter._running = False
        adapter._shutting_down = True
        restart_task = adapter._poll_restart_task
        if restart_task is not None:
            restart_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await restart_task


@pytest.mark.asyncio
async def test_poll_messages_keeps_polling_while_media_handler_is_blocked():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter._running = True
    adapter._bridge_port = 3000
    adapter._poll_interval = 0
    second_get = asyncio.Event()
    release_second_get = asyncio.Event()
    responses = [
        _AsyncResponse([{"messageId": "media-1"}]),
        _AsyncResponse([], entered=second_get, release=release_second_get),
    ]
    adapter._http_session = MagicMock()
    adapter._http_session.get = MagicMock(side_effect=responses)
    adapter._check_managed_bridge_exit = AsyncMock(return_value=None)
    adapter._send_read_receipt = AsyncMock()
    adapter._build_message_event = AsyncMock(
        return_value=MagicMock(message_type=MessageType.PHOTO)
    )
    never_finishes = asyncio.Event()

    async def blocked_media_handler(_event):
        await never_finishes.wait()

    adapter.handle_message = AsyncMock(side_effect=blocked_media_handler)

    poll_task = asyncio.create_task(adapter._poll_messages())
    try:
        await asyncio.wait_for(second_get.wait(), timeout=1)
        assert adapter._http_session.get.call_count == 2
        adapter._build_message_event.assert_awaited_once_with({"messageId": "media-1"})
        adapter.handle_message.assert_awaited_once()
    finally:
        adapter._running = False
        release_second_get.set()
        poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task
        inbound_tasks = list(adapter._inbound_tasks)
        for task in inbound_tasks:
            task.cancel()
        await asyncio.gather(*inbound_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_inbound_event_construction_timeout_logs_and_retires_task(caplog):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(
        PlatformConfig(enabled=True, typing_indicator=False, extra={"session_name": "test"})
    )
    adapter._inbound_event_timeout = 0.01
    async def hung_build(_data):
        await asyncio.Event().wait()

    adapter._build_message_event = AsyncMock(side_effect=hung_build)

    with caplog.at_level(logging.ERROR, logger="plugins.platforms.whatsapp.adapter"):
        adapter._start_inbound_task({"messageId": "hung-media"})
        task = next(iter(adapter._inbound_tasks))
        await asyncio.wait_for(task, timeout=1)
        await asyncio.sleep(0)

    assert not adapter._inbound_tasks
    assert "timed out" in caplog.text
    adapter._build_message_event.assert_awaited_once_with({"messageId": "hung-media"})


@pytest.mark.asyncio
async def test_disconnect_joins_hung_media_build():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(
        PlatformConfig(enabled=True, typing_indicator=False, extra={"session_name": "test"})
    )
    async def hung_build(_data):
        await asyncio.Event().wait()

    adapter._build_message_event = AsyncMock(side_effect=hung_build)
    adapter._start_inbound_task({"messageId": "hung-media"})
    task = next(iter(adapter._inbound_tasks))
    await asyncio.sleep(0)

    await adapter.disconnect()

    assert task.done()
    assert not adapter._inbound_tasks


@pytest.mark.asyncio
async def test_disconnect_joins_hung_read_receipt_started_by_ingestion():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(
        PlatformConfig(enabled=True, typing_indicator=False, extra={"session_name": "test"})
    )
    event = _message_event(MessageType.TEXT, message_id="text-1", text="hello")
    adapter._build_message_event = AsyncMock(return_value=event)
    adapter._enqueue_text_event = MagicMock()
    receipt_started = asyncio.Event()
    receipt_cancelled = asyncio.Event()
    receipt_task = None

    async def hung_read_receipt(_data):
        nonlocal receipt_task
        receipt_task = asyncio.current_task()
        receipt_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            receipt_cancelled.set()

    adapter._send_read_receipt = hung_read_receipt

    await adapter._ingest_message({"messageId": "text-1"})
    await asyncio.wait_for(receipt_started.wait(), timeout=1)

    await adapter.disconnect()

    assert receipt_task is not None
    assert receipt_task.cancelled()
    assert receipt_cancelled.is_set()
    assert not adapter._inbound_tasks


@pytest.mark.asyncio
async def test_disconnect_cancels_pending_poll_restart_without_replacement():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter(
        PlatformConfig(enabled=True, typing_indicator=False, extra={"session_name": "test"})
    )
    adapter._running = True
    adapter._poll_restart_delay = 60
    adapter._poll_messages = AsyncMock()
    adapter._poll_restart_task = asyncio.create_task(adapter._restart_polling_after_delay())
    restart_task = adapter._poll_restart_task
    await asyncio.sleep(0)

    await adapter.disconnect()
    await asyncio.sleep(0)

    assert restart_task.cancelled()
    assert adapter._poll_restart_task is None
    adapter._poll_messages.assert_not_awaited()

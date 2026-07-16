"""Regression tests for typing-refresh and final-delivery ordering."""

import asyncio
from contextlib import suppress

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, **kwargs):
        return SendResult(success=True, message_id="sent")

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def get_chat_info(self, chat_id):
        return {}


def _make_event() -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="42",
            chat_type="dm",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery_kind", ["text", "media"])
async def test_final_delivery_waits_for_typing_refresh_to_stop(
    delivery_kind, tmp_path,
):
    """Final text and attachment delivery must start only after refresh stops."""
    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="test"),
        Platform.TELEGRAM,
    )
    event = _make_event()
    session_key = build_session_key(event.source)
    typing_stopped = False
    delivery_observations = []

    attachment = tmp_path / "result.pdf"
    attachment.write_bytes(b"test attachment")

    async def handler(_event):
        if delivery_kind == "media":
            return f"MEDIA:{attachment}"
        return "final response"

    async def stop_typing_refresh(chat_id, typing_task=None, **kwargs):
        nonlocal typing_stopped
        if typing_task is not None and not typing_task.done():
            typing_task.cancel()
            with suppress(asyncio.CancelledError):
                await typing_task
        typing_stopped = True

    async def observe_text_delivery(**kwargs):
        delivery_observations.append(typing_stopped)
        return SendResult(success=True, message_id="text")

    async def observe_media_delivery(**kwargs):
        delivery_observations.append(typing_stopped)
        return SendResult(success=True, message_id="media")

    adapter._message_handler = handler
    adapter._stop_typing_refresh = stop_typing_refresh
    adapter._send_with_retry = observe_text_delivery
    adapter.send_document = observe_media_delivery
    adapter._get_human_delay = lambda: 0
    adapter.filter_media_delivery_paths = lambda paths: paths

    task = asyncio.create_task(
        adapter._process_message_background(event, session_key)
    )
    adapter._session_tasks[session_key] = task
    await task

    assert delivery_observations == [True]


@pytest.mark.asyncio
async def test_cancelling_processor_during_typing_stop_suppresses_final_delivery():
    """Caller cancellation while stopping typing must not be mistaken for child cancellation."""
    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="test"),
        Platform.TELEGRAM,
    )
    event = _make_event()
    session_key = build_session_key(event.source)
    typing_cleanup_started = asyncio.Event()
    release_typing_cleanup = asyncio.Event()
    delivered: list[str] = []

    async def handler(_event):
        # Let the refresh coroutine start so cancellation enters its cleanup.
        await asyncio.sleep(0)
        return "stale final response"

    async def slow_typing_refresh(chat_id, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            typing_cleanup_started.set()
            await release_typing_cleanup.wait()
            raise

    async def observe_text_delivery(**kwargs):
        delivered.append(kwargs["content"])
        return SendResult(success=True, message_id="text")

    adapter._message_handler = handler
    adapter._keep_typing = slow_typing_refresh
    adapter._send_with_retry = observe_text_delivery

    task = asyncio.create_task(
        adapter._process_message_background(event, session_key)
    )
    adapter._session_tasks[session_key] = task

    await asyncio.wait_for(typing_cleanup_started.wait(), timeout=1)
    task.cancel()
    release_typing_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert delivered == []


@pytest.mark.asyncio
async def test_cancellation_during_finally_still_releases_session_tracking():
    """Cancellation in typing cleanup must not strand the session guard."""
    adapter = _StubAdapter(
        PlatformConfig(enabled=True, token="test"),
        Platform.TELEGRAM,
    )
    event = _make_event()
    session_key = build_session_key(event.source)
    typing_cleanup_started = asyncio.Event()
    release_typing_cleanup = asyncio.Event()

    async def failing_handler(_event):
        await asyncio.sleep(0)
        raise RuntimeError("expected test failure")

    async def slow_typing_refresh(chat_id, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            typing_cleanup_started.set()
            await release_typing_cleanup.wait()
            raise

    adapter._message_handler = failing_handler
    adapter._keep_typing = slow_typing_refresh

    task = asyncio.create_task(
        adapter._process_message_background(event, session_key)
    )
    adapter._session_tasks[session_key] = task

    await asyncio.wait_for(typing_cleanup_started.wait(), timeout=1)
    task.cancel()
    release_typing_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert session_key not in adapter._session_tasks
    assert session_key not in adapter._active_sessions

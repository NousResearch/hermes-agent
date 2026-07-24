"""Regression tests for duplicate inbound delivery while a session is active."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter():
    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), Platform.BLUEBUBBLES)
    adapter._send_with_retry = AsyncMock(return_value=None)
    return adapter


def _make_event(text="hi", message_id="msg-1"):
    source = SessionSource(
        platform=Platform.BLUEBUBBLES,
        chat_id="any;-;+15550000000",
        chat_type="dm",
        user_id="+15550000000",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
    )


@pytest.mark.asyncio
async def test_duplicate_active_message_id_is_dropped_not_queued():
    """A repeated webhook delivery of the active message must not become a follow-up.

    BlueBubbles can redeliver the same inbound iMessage while the first copy is
    still being processed. If the duplicate is queued as a pending follow-up, the
    gateway can send the first response via the queued-follow-up fallback and
    then send the final response again.
    """
    adapter = _make_adapter()
    first = _make_event(text="first", message_id="same-message")
    duplicate = _make_event(text="first", message_id="same-message")
    session_key = build_session_key(first.source)

    started = asyncio.Event()
    release = asyncio.Event()
    processed = []

    async def handler(event):
        processed.append(event.text)
        started.set()
        await release.wait()
        return "ok"

    adapter._message_handler = handler

    await adapter.handle_message(first)
    await asyncio.wait_for(started.wait(), timeout=1.0)

    await adapter.handle_message(duplicate)

    assert session_key not in adapter._pending_messages
    assert processed == ["first"]

    release.set()
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_duplicate_queued_message_id_is_dropped_while_drain_is_active():
    """The active ID must follow a queued event into its drain task.

    A transport can redeliver the queued follow-up after the first turn hands
    it to a drain task. It must be dropped rather than re-queued as another
    follow-up.
    """
    adapter = _make_adapter()
    # Bypass text debouncing so the test controls the queued-turn handoff.
    adapter._busy_text_mode = "interrupt"
    first = _make_event(text="first", message_id="first-message")
    second = _make_event(text="second", message_id="queued-message")
    duplicate_second = _make_event(text="second", message_id="queued-message")
    session_key = build_session_key(first.source)

    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()
    release_second = asyncio.Event()
    processed = []

    async def handler(event):
        processed.append(event.text)
        if event.message_id == "first-message":
            first_started.set()
            await release_first.wait()
        else:
            second_started.set()
            await release_second.wait()
        return "ok"

    adapter._message_handler = handler

    await adapter.handle_message(first)
    await asyncio.wait_for(first_started.wait(), timeout=1.0)
    await adapter.handle_message(second)
    assert adapter._pending_messages[session_key] is second

    release_first.set()
    await asyncio.wait_for(second_started.wait(), timeout=1.0)
    await adapter.handle_message(duplicate_second)

    assert session_key not in adapter._pending_messages
    assert processed == ["first", "second"]

    release_second.set()
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_duplicate_late_arrival_message_id_is_dropped_while_drain_is_active():
    """The late-arrival cleanup drain must also take ownership of the ID."""
    adapter = _make_adapter()
    adapter._busy_text_mode = "interrupt"
    first = _make_event(text="first", message_id="first-message")
    late = _make_event(text="late", message_id="late-message")
    duplicate_late = _make_event(text="late", message_id="late-message")
    session_key = build_session_key(first.source)

    late_started = asyncio.Event()
    release_late = asyncio.Event()
    injected = False
    processed = []

    async def handler(event):
        processed.append(event.text)
        if event.message_id == "late-message":
            late_started.set()
            await release_late.wait()
        return "ok"

    async def stop_typing_injects_late(*_args, **_kwargs):
        nonlocal injected
        if not injected:
            adapter._pending_messages[session_key] = late
            injected = True

    adapter._message_handler = handler
    adapter.stop_typing = stop_typing_injects_late

    await adapter.handle_message(first)
    await asyncio.wait_for(late_started.wait(), timeout=1.0)
    await adapter.handle_message(duplicate_late)

    assert session_key not in adapter._pending_messages
    assert processed == ["first", "late"]

    release_late.set()
    await adapter.cancel_background_tasks()


@pytest.mark.asyncio
async def test_active_message_id_is_cleared_after_completion():
    adapter = _make_adapter()
    event = _make_event(text="single", message_id="clear-me")
    session_key = build_session_key(event.source)

    async def handler(_event):
        return "ok"

    adapter._message_handler = handler

    await adapter.handle_message(event)
    await asyncio.sleep(0)
    await adapter.cancel_background_tasks()

    assert session_key not in adapter._active_sessions
    assert session_key not in adapter._active_session_message_ids


@pytest.mark.asyncio
async def test_cancel_background_tasks_clears_active_message_ids():
    adapter = _make_adapter()
    adapter._active_session_message_ids["stale-session"] = "stale-message"

    await adapter.cancel_background_tasks()

    assert not adapter._active_session_message_ids

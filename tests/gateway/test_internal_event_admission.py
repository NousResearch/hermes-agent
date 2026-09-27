"""Regression coverage for #121000: pinned internal-event admission must be truthful."""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from gateway.wake import WakeNotAccepted


class _DispatchAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)

    @property
    def name(self):
        return "telegram"

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _route():
    """A real base adapter plus the smallest runner seam needed by session dispatch."""
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
    )
    entry = SessionEntry(
        session_key="agent:main:telegram:dm:chat-1",
        session_id="session-pinned",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = object.__new__(GatewayRunner)
    runner._session_db = None  # one real fail-closed arm: no durable session owner
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        get_or_create_session=AsyncMock(return_value=entry),
    )
    runner._session_key_for_source = lambda _source: entry.session_key
    runner._cache_session_source = lambda *_args: None
    runner._is_telegram_topic_lane = lambda *_args: False

    adapter = _DispatchAdapter()
    resolutions = []
    handled_events = []

    async def handler(event):
        handled_events.append(event)
        resolutions.append(await runner._hmwa_resolve_session(event, event.source))

    adapter.set_message_handler(handler)
    return adapter, runner, source, entry.session_key, resolutions, handled_events


async def _drain(adapter):
    tasks = [task for task in adapter._background_tasks if not task.done()]
    if tasks:
        await asyncio.gather(*tasks)


@pytest.mark.parametrize("failure_mode", ("resolution", "queued_cancel"))
@pytest.mark.asyncio
async def test_non_delegation_dispatch_refusal_reaches_carrier(failure_mode):
    """A non-delegation pin may not turn any post-queue refusal into successful admission."""
    adapter, runner, source, session_key, resolutions, handled_events = _route()
    runner._build_process_event_source = lambda _evt: source
    runner._resolve_injection_adapter = lambda _platform, _source: adapter
    event = {
        "type": "completion",
        "session_key": session_key,
        "parent_session_id": "session-pinned",
    }

    if failure_mode == "queued_cancel":
        adapter._active_sessions[session_key] = asyncio.Event()
        carrier = asyncio.create_task(
            runner._inject_watch_notification(
                "process complete", event, raise_not_accepted=True
            )
        )
        async with asyncio.timeout(1):
            while session_key not in adapter._pending_messages:
                await asyncio.sleep(0)
        queued = adapter._pending_messages[session_key]
        assert queued.metadata.get("gateway_session_strict") is not True
        await adapter.cancel_session_processing(session_key)
        with pytest.raises(WakeNotAccepted, match="not accepted for dispatch"):
            await asyncio.wait_for(carrier, timeout=1)
        assert resolutions == []
        assert handled_events == []
        return

    with pytest.raises(WakeNotAccepted, match="not accepted for dispatch"):
        await runner._inject_watch_notification(
            "process complete", event, raise_not_accepted=True
        )

    await _drain(adapter)
    assert resolutions == [None]
    assert handled_events[0].metadata.get("gateway_session_strict") is not True


@pytest.mark.asyncio
async def test_async_delegation_dispatch_remains_fail_closed():
    """A durable delegation completion may still be dropped after adapter admission."""
    adapter, runner, source, session_key, resolutions, _handled_events = _route()
    runner._build_process_event_source = lambda _evt: source
    runner._resolve_injection_adapter = lambda _platform, _source: adapter
    event = {
        "type": "async_delegation",
        "session_key": "agent:main:telegram:dm:chat-1",
        "parent_session_id": "session-pinned",
    }

    assert await runner._inject_watch_notification("delegation complete", event) is True
    await _drain(adapter)
    assert resolutions == [None]

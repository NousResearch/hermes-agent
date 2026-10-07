"""Regression test: cancel_background_tasks must drain late-arrival tasks.

During gateway shutdown, a message arriving while
cancel_background_tasks is mid-await can spawn a fresh
_process_message_background task via handle_message, which is added
to self._background_tasks.  Without the re-drain loop, the subsequent
_background_tasks.clear() drops the reference; the task runs
untracked against a disconnecting adapter.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key

class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}

def _make_adapter():
    adapter = _StubAdapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    adapter._send_with_retry = AsyncMock(return_value=None)
    return adapter

def _event(text, cid="42"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id=cid, chat_type="dm"),
    )

@pytest.mark.asyncio
async def test_cancel_background_tasks_drains_late_arrivals():
    """A message that arrives during the gather window must be picked
    up by the re-drain loop, not leaked as an untracked task."""
    adapter = _make_adapter()
    sk = build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    )

    m1_started = asyncio.Event()
    m1_cleanup_running = asyncio.Event()
    m2_started = asyncio.Event()
    m2_cancelled = asyncio.Event()

    async def handler(event):
        if event.text == "M1":
            m1_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                m1_cleanup_running.set()
                # Widen the gather window with a shielded cleanup
                # delay so M2 can get injected during it.
                await asyncio.shield(asyncio.sleep(0.2))
                raise
        else:  # M2 — the late arrival
            m2_started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                m2_cancelled.set()
                raise

    adapter._message_handler = handler

    # Spawn M1.
    await adapter.handle_message(_event("M1"))
    await asyncio.wait_for(m1_started.wait(), timeout=1.0)

    # Kick off shutdown.  This will cancel M1 and await its cleanup.
    cancel_task = asyncio.create_task(adapter.cancel_background_tasks())

    # Wait until M1's cleanup is running (inside the shielded sleep).
    # This is the race window: cancel_task is awaiting gather, M1 is
    # shielded in cleanup, the _active_sessions entry has been cleared
    # by M1's own finally.
    await asyncio.wait_for(m1_cleanup_running.wait(), timeout=1.0)

    # Clear the active-session entry (M1's finally hasn't fully run yet,
    # but in production the platform dispatcher would deliver a new
    # message that takes the no-active-session spawn path).  For this
    # repro, make it deterministic.
    adapter._active_sessions.pop(sk, None)

    # Inject late arrival — spawns a fresh _process_message_background
    # task and adds it to _background_tasks while cancel_task is still
    # in gather.
    await adapter.handle_message(_event("M2"))
    await asyncio.wait_for(m2_started.wait(), timeout=1.0)

    # Let cancel_task finish.  Round 1's gather completes when M1's
    # shielded cleanup finishes.  Round 2 should pick up M2.
    await asyncio.wait_for(cancel_task, timeout=5.0)

    # Assert M2 was drained, not leaked.
    assert m2_cancelled.is_set(), (
        "Late-arrival M2 was NOT cancelled by cancel_background_tasks — "
        "the re-drain loop is missing and the task leaked"
    )
    assert adapter._background_tasks == set()






@pytest.mark.asyncio
async def test_deferred_command_runs_before_queued_prompt_with_one_session_owner():
    """A deferred control command runs before queued text, and the turn hands off exactly one owner.

    The in-band handoff popped the ordinary follow-up first; the cleanup path then spawned the
    deferred command too, so both ran concurrently on one session in the wrong order.
    """
    adapter = _make_adapter()
    sk = build_session_key(SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm"))
    release = asyncio.Event()
    started = asyncio.Event()
    order: list[str] = []
    active = 0
    overlap = False

    async def handler(event):
        nonlocal active, overlap
        order.append(event.text)
        active += 1
        overlap = overlap or active > 1
        try:
            if event.text == "turn":
                started.set()
                await release.wait()
            else:
                await asyncio.sleep(0.05)
        finally:
            active -= 1

    adapter._message_handler = handler
    await adapter.handle_message(_event("turn"))
    await asyncio.wait_for(started.wait(), timeout=2.0)
    adapter._pending_messages[sk] = _event("follow-up")
    adapter.defer_command_until_idle(sk, _event("/compress"))
    adapter.defer_command_until_idle(sk, _event("/undo"))
    release.set()
    deadline = asyncio.get_running_loop().time() + 5.0
    while len(order) < 4 and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    while any(not t.done() for t in list(adapter._background_tasks)):
        await asyncio.sleep(0.01)
    assert order == ["turn", "/compress", "/undo", "follow-up"]
    assert not overlap


@pytest.mark.asyncio
async def test_cancel_background_tasks_discards_deferred_command_behind_live_owner():
    """Adapter teardown fences deferred commands before cancelling the owner that would drain them.

    Cancelling the owner does not set its interrupt event, so its cleanup drained the queued
    command and ran a transcript mutation during teardown.
    """
    adapter = _make_adapter()
    sk = build_session_key(SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm"))
    started = asyncio.Event()
    ran: list[str] = []

    async def handler(event):
        ran.append(event.text)
        if event.text == "turn":
            started.set()
            await asyncio.Event().wait()

    adapter._message_handler = handler
    await adapter.handle_message(_event("turn"))
    await asyncio.wait_for(started.wait(), timeout=2.0)
    adapter.defer_command_until_idle(sk, _event("/undo"))
    await adapter.cancel_background_tasks()
    await asyncio.sleep(0.05)
    assert ran == ["turn"]

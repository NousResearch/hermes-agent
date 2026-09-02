"""Regression test: cancel_background_tasks must drain late-arrival tasks.

During gateway shutdown, a message arriving while
cancel_background_tasks is mid-await can spawn a fresh
_process_message_background task via handle_message, which is added
to self._background_tasks.  Without the re-drain loop, the subsequent
_background_tasks.clear() drops the reference; the task runs
untracked against a disconnecting adapter.
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
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


def _redirect_flush_dir(tmp_path, monkeypatch):
    """Point the #72680 pending-message spool at a temp dir."""
    flush_dir = tmp_path / "pending_messages"
    flush_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)
    return flush_dir


def _flushed_payloads(flush_dir):
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(flush_dir.glob("*.json"))
    ]


async def _stall_until_drain(adapter, resume):
    """Add a background task that resists cancellation until ``resume`` is set.

    Returns an Event that fires once the task has entered its cancellation
    handler — i.e. once ``cancel_background_tasks`` is parked in the drain
    ``wait_for``, which is the exact window the teardown budget expires in.
    """
    draining = asyncio.Event()

    async def stubborn():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            draining.set()
            # Shielded, so the drain's gather cannot complete until the
            # test releases us — holding the coroutine inside the loop.
            await asyncio.shield(resume.wait())
            raise

    task = asyncio.create_task(stubborn())
    await asyncio.sleep(0)  # let it reach the sleep
    adapter._background_tasks.add(task)
    return draining, task


@pytest.mark.asyncio
async def test_pending_flush_survives_teardown_budget_cancellation(tmp_path, monkeypatch):
    """The #72680 flush must still run when the teardown budget cancels the drain.

    ``GatewayRunner._bounded_adapter_teardown`` wraps this coroutine in
    ``_await_adapter_cleanup_with_timeout``, which calls ``task.cancel()``
    once ``HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT`` expires.  If the
    settle tail is not in a ``finally``, that cancellation skips the flush
    and the queued follow-up is lost with no on-disk recovery copy.
    """
    flush_dir = _redirect_flush_dir(tmp_path, monkeypatch)
    adapter = _make_adapter()
    sk = build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    )
    adapter._pending_messages[sk] = _event("queued follow-up")

    resume = asyncio.Event()
    draining, stubborn_task = await _stall_until_drain(adapter, resume)

    cancel_task = asyncio.create_task(adapter.cancel_background_tasks())
    await asyncio.wait_for(draining.wait(), timeout=1.0)

    # The teardown budget expires: the runner cancels the cleanup coroutine
    # outright while it is still parked in the drain.
    cancel_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancel_task

    payloads = _flushed_payloads(flush_dir)
    assert len(payloads) == 1, (
        "pending message was NOT flushed to disk when the teardown budget "
        "cancelled the drain — the settle tail is not running in a finally"
    )
    assert payloads[0]["session_key"] == sk
    assert payloads[0]["reason"] == "adapter_shutdown"
    assert payloads[0]["data"]["text"] == "queued follow-up"
    assert adapter._pending_messages == {}

    resume.set()
    await asyncio.gather(stubborn_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancellation_still_propagates_to_caller(tmp_path, monkeypatch):
    """The finally must not swallow CancelledError.

    ``_await_adapter_cleanup_with_timeout`` relies on the cancellation
    completing; a swallowed CancelledError would turn a bounded teardown
    back into an unbounded one.
    """
    _redirect_flush_dir(tmp_path, monkeypatch)
    adapter = _make_adapter()

    resume = asyncio.Event()
    draining, stubborn_task = await _stall_until_drain(adapter, resume)

    cancel_task = asyncio.create_task(adapter.cancel_background_tasks())
    await asyncio.wait_for(draining.wait(), timeout=1.0)
    cancel_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await cancel_task
    assert cancel_task.cancelled()

    resume.set()
    await asyncio.gather(stubborn_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_timely_teardown_flushes_pending_exactly_once(tmp_path, monkeypatch):
    """Moving the tail into a finally must not double-flush the timely path.

    A clean teardown already flushed correctly before this change (#72680);
    guard that it still writes exactly one payload per pending slot.
    """
    flush_dir = _redirect_flush_dir(tmp_path, monkeypatch)
    adapter = _make_adapter()
    sk = build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    )
    adapter._pending_messages[sk] = _event("queued follow-up")

    await adapter.cancel_background_tasks()

    payloads = _flushed_payloads(flush_dir)
    assert len(payloads) == 1
    assert payloads[0]["data"]["text"] == "queued follow-up"
    assert adapter._pending_messages == {}
    assert adapter._active_sessions == {}
    assert adapter._background_tasks == set()


@pytest.mark.asyncio
async def test_second_teardown_pass_after_cancellation_is_safe(tmp_path, monkeypatch):
    """A retried teardown after a cancelled pass must not raise or re-flush.

    ``_bounded_adapter_teardown`` keeps making forward progress after a
    timeout, so the adapter can see a second cleanup call.  The first pass
    already drained the pending slots, so the second must be a no-op.
    """
    flush_dir = _redirect_flush_dir(tmp_path, monkeypatch)
    adapter = _make_adapter()
    sk = build_session_key(
        SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm")
    )
    adapter._pending_messages[sk] = _event("queued follow-up")

    resume = asyncio.Event()
    draining, stubborn_task = await _stall_until_drain(adapter, resume)

    cancel_task = asyncio.create_task(adapter.cancel_background_tasks())
    await asyncio.wait_for(draining.wait(), timeout=1.0)
    cancel_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancel_task

    assert len(_flushed_payloads(flush_dir)) == 1

    resume.set()
    await asyncio.gather(stubborn_task, return_exceptions=True)

    # Second pass: nothing left to flush, and it must complete cleanly.
    await adapter.cancel_background_tasks()
    assert len(_flushed_payloads(flush_dir)) == 1
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_cancel_background_tasks_handles_no_tasks():
    """Regression guard: no tasks, no hang, no error."""
    adapter = _make_adapter()
    await adapter.cancel_background_tasks()
    assert adapter._background_tasks == set()


@pytest.mark.asyncio
async def test_cancel_background_tasks_bounded_rounds():
    """Regression guard: the drain loop is bounded — it does not spin
    forever even if late-arrival tasks keep getting spawned."""
    adapter = _make_adapter()

    # Single well-behaved task that cancels cleanly — baseline check
    # that the loop terminates in one round.
    async def quick():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            raise

    task = asyncio.create_task(quick())
    adapter._background_tasks.add(task)

    await adapter.cancel_background_tasks()
    assert task.done()
    assert adapter._background_tasks == set()

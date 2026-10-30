"""Regression test for #81335 — fatal-error handler must survive cancellation
of the task that awaits ``_notify_fatal_error``.

The Telegram adapter escalates exhausted polling retries from inside its own
``_polling_error_task``. The gateway's fatal handler tears the adapter down
via ``disconnect()``, which cancels that very task. The handler used to be
killed mid-flight by the propagating ``CancelledError``: the adapter was
already popped from the gateway's adapter map, but the platform was never
queued for background reconnection — a zombie gateway.

These tests model that carrier-cancellation race directly against
``BasePlatformAdapter._notify_fatal_error``.
"""

import asyncio

import pytest

from gateway.platforms.base import BasePlatformAdapter


class _FakeAdapter:
    """Minimal stand-in exposing only what ``_notify_fatal_error`` touches."""

    _notify_fatal_error = BasePlatformAdapter._notify_fatal_error

    def __init__(self):
        self._fatal_error_handler = None
        self.handler_completed = False
        self._detached_fatal_tasks = set()


@pytest.mark.asyncio
async def test_handler_survives_carrier_cancellation():
    """Handler must run to completion even when the awaiting task is
    cancelled from inside the handler (the disconnect() self-cancel race)."""
    adapter = _FakeAdapter()
    carrier_task = None

    async def gateway_handler(a):
        # Step 1: teardown — cancels the carrier task (what the real
        # handler does indirectly via adapter.disconnect()).
        carrier_task.cancel()
        # Yield so the cancellation is delivered while we're still running.
        await asyncio.sleep(0.05)
        # Step 2: the part that never ran before the fix — queueing the
        # platform for background reconnection.
        a.handler_completed = True

    adapter._fatal_error_handler = gateway_handler

    async def carrier():
        await adapter._notify_fatal_error()

    carrier_task = asyncio.create_task(carrier())
    with pytest.raises(asyncio.CancelledError):
        await carrier_task

    # Let the detached, shielded handler finish.
    await asyncio.sleep(0.2)

    assert carrier_task.cancelled()
    assert adapter.handler_completed, (
        "fatal-error handler was killed by carrier cancellation — platform "
        "would never be queued for reconnection (zombie gateway, #81335)"
    )


@pytest.mark.asyncio
async def test_carrier_cancellation_still_propagates():
    """The carrier task itself must still observe CancelledError (teardown
    semantics unchanged) — only the handler is shielded."""
    adapter = _FakeAdapter()
    carrier_task = None

    async def gateway_handler(a):
        carrier_task.cancel()
        await asyncio.sleep(0.05)
        a.handler_completed = True

    adapter._fatal_error_handler = gateway_handler

    async def carrier():
        await adapter._notify_fatal_error()

    carrier_task = asyncio.create_task(carrier())
    with pytest.raises(asyncio.CancelledError):
        await carrier_task
    assert carrier_task.cancelled()


@pytest.mark.asyncio
async def test_uncancelled_path_unchanged():
    """Normal path (no cancellation) behaves exactly as before."""
    adapter = _FakeAdapter()

    async def gateway_handler(a):
        a.handler_completed = True

    adapter._fatal_error_handler = gateway_handler
    await adapter._notify_fatal_error()
    assert adapter.handler_completed


@pytest.mark.asyncio
async def test_sync_handler_still_supported():
    """Synchronous handlers (non-coroutine return) keep working."""
    adapter = _FakeAdapter()

    def gateway_handler(a):
        a.handler_completed = True

    adapter._fatal_error_handler = gateway_handler
    await adapter._notify_fatal_error()
    assert adapter.handler_completed


@pytest.mark.asyncio
async def test_no_handler_is_noop():
    adapter = _FakeAdapter()
    await adapter._notify_fatal_error()  # must not raise

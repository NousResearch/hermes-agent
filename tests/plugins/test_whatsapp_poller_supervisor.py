"""Regression tests for the WhatsApp inbound poller supervisor.

Incident 2026-08-15: the poll task died silently (permanent ``break`` on a
transient condition) while the adapter kept reporting healthy — outbound
sends worked, /health said connected, but nobody drained GET /messages for
~20h and 18 inbound messages were lost.

Contract under test: ANY poll-task termination outside an orderly shutdown
must flip the adapter into the fatal-retryable state so the gateway's
reconnect watcher rebuilds it.
"""

import asyncio
from typing import Optional

import pytest

from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def _bare_adapter() -> WhatsAppAdapter:
    a = WhatsAppAdapter.__new__(WhatsAppAdapter)
    a._poll_task = None
    a._shutting_down = False
    a._running = True
    a._fatal_error_code = None
    a._fatal_error_message = None
    a._fatal_error_retryable = True
    a._fatal_error_handler = None

    async def _noop_notify():
        return None

    a._notify_fatal_error = _noop_notify  # type: ignore[method-assign]
    a._write_runtime_status_safe = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return a


@pytest.mark.asyncio
async def test_poller_crash_sets_retryable_fatal_error():
    """A poll task that raises must mark the adapter fatal-retryable."""
    adapter = _bare_adapter()

    async def crashing_poll():
        raise RuntimeError("bridge HTTP session vanished while poller running")

    adapter._poll_messages = crashing_poll  # type: ignore[method-assign]
    adapter._poll_task = adapter._spawn_poll_task()
    with pytest.raises(RuntimeError):
        await adapter._poll_task
    await asyncio.sleep(0)  # let the done-callback run

    assert adapter.has_fatal_error
    assert adapter.fatal_error_code == "whatsapp_poller_died"
    assert adapter.fatal_error_retryable


@pytest.mark.asyncio
async def test_poller_plain_return_sets_retryable_fatal_error():
    """A poll task that just returns (the old silent-break bug) is an anomaly."""
    adapter = _bare_adapter()

    async def returning_poll():
        return  # old behaviour: silent break -> task completes quietly

    adapter._poll_messages = returning_poll  # type: ignore[method-assign]
    adapter._poll_task = adapter._spawn_poll_task()
    await adapter._poll_task
    await asyncio.sleep(0)

    assert adapter.has_fatal_error
    assert adapter.fatal_error_code == "whatsapp_poller_died"
    assert adapter.fatal_error_retryable


@pytest.mark.asyncio
async def test_poller_cancel_during_shutdown_is_orderly():
    """disconnect() cancels the poller with _shutting_down set: no fatal."""
    adapter = _bare_adapter()

    async def long_poll():
        await asyncio.sleep(3600)

    adapter._poll_messages = long_poll  # type: ignore[method-assign]
    adapter._poll_task = adapter._spawn_poll_task()
    await asyncio.sleep(0)
    adapter._shutting_down = True
    adapter._poll_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await adapter._poll_task
    await asyncio.sleep(0)

    assert not adapter.has_fatal_error


@pytest.mark.asyncio
async def test_poller_return_after_bridge_exit_fatal_is_not_double_reported():
    """When _check_managed_bridge_exit already set the fatal error, the
    supervisor must not overwrite its code."""
    adapter = _bare_adapter()

    async def poll_after_bridge_died():
        adapter._set_fatal_error(
            "whatsapp_bridge_exited", "bridge exited (code 1)", retryable=True
        )
        return

    adapter._poll_messages = poll_after_bridge_died  # type: ignore[method-assign]
    adapter._poll_task = adapter._spawn_poll_task()
    await adapter._poll_task
    await asyncio.sleep(0)

    assert adapter.fatal_error_code == "whatsapp_bridge_exited"


@pytest.mark.asyncio
async def test_replaced_poll_task_termination_is_ignored():
    """A superseded poll task (reconnect spawned a new one) must not poison
    the fresh adapter state when the old task finishes."""
    adapter = _bare_adapter()

    async def returning_poll():
        return

    adapter._poll_messages = returning_poll  # type: ignore[method-assign]
    old_task = adapter._spawn_poll_task()
    adapter._poll_task = None  # a "newer" lifecycle already cleared/replaced it
    await old_task
    await asyncio.sleep(0)

    assert not adapter.has_fatal_error

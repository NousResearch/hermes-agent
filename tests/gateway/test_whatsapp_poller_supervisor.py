"""Regression tests for the WhatsApp inbound poller supervisor.

The poll task used to end silently (a bare ``break``, or an exception nobody
retrieved) while the adapter kept reporting connected: outbound sends worked,
but nothing drained the bridge's GET /messages queue, so inbound delivery
stopped until a restart. Contract: any poll-task termination outside an
orderly stop flips the adapter into the retryable fatal state, which the
gateway's reconnect watcher rebuilds.
"""

import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform


def _make_adapter():
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter._bridge_port = 19876
    adapter._session_path = Path("/nonexistent/test-wa-session")
    adapter._bridge_process = None
    adapter._bridge_log_fh = adapter._bridge_log = None
    adapter._poll_task = adapter._http_session = None
    adapter._shutting_down = False
    adapter._running = True
    adapter._fatal_error_code = adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._write_runtime_status_safe = lambda *args, **kwargs: None
    return adapter


async def _start_poller(adapter, poll=None):
    """Start the poll task through the adapter's real creation site."""
    if poll is not None:
        adapter._poll_messages = poll
    adapter._attach_to_bridge(None)
    return adapter._poll_task


async def _finish(adapter, task):
    await asyncio.wait({task})
    for _ in range(3):  # done-callback, then the notify task it schedules
        await asyncio.sleep(0)
    if adapter._http_session is not None and not adapter._http_session.closed:
        await adapter._http_session.close()


@pytest.mark.asyncio
async def test_lost_http_session_marks_adapter_retryable_fatal():
    """The real poll loop losing its session must not stop inbound silently."""
    adapter = _make_adapter()
    task = await _start_poller(adapter)
    session, adapter._http_session = adapter._http_session, None
    await session.close()
    await _finish(adapter, task)

    assert adapter.has_fatal_error
    assert adapter.fatal_error_code == "whatsapp_poller_died"
    assert adapter.fatal_error_retryable


@pytest.mark.asyncio
async def test_poller_crash_marks_fatal_and_notifies_gateway():
    adapter = _make_adapter()
    notified = []
    adapter._fatal_error_handler = notified.append

    async def crashing_poll():
        raise RuntimeError("boom")

    task = await _start_poller(adapter, crashing_poll)
    await _finish(adapter, task)

    assert adapter.fatal_error_code == "whatsapp_poller_died"
    assert adapter.fatal_error_retryable
    assert notified == [adapter]


@pytest.mark.asyncio
async def test_failing_fatal_handler_is_logged_not_left_unretrieved(caplog):
    adapter = _make_adapter()

    def broken_handler(_adapter):
        raise RuntimeError("handler broke")

    adapter._fatal_error_handler = broken_handler

    async def returning_poll():
        return

    task = await _start_poller(adapter, returning_poll)
    with caplog.at_level(logging.ERROR):
        await _finish(adapter, task)

    assert adapter.fatal_error_code == "whatsapp_poller_died"
    assert "handler broke" in caplog.text


@pytest.mark.asyncio
async def test_disconnect_cancel_is_orderly():
    adapter = _make_adapter()
    never = asyncio.Event()

    async def long_poll():
        await never.wait()

    task = await _start_poller(adapter, long_poll)
    await asyncio.sleep(0)
    adapter._shutting_down = True
    task.cancel()
    await _finish(adapter, task)

    assert not adapter.has_fatal_error


@pytest.mark.asyncio
async def test_loop_ending_because_adapter_stopped_is_orderly():
    adapter = _make_adapter()

    async def poll_until_stopped():
        adapter._running = False

    task = await _start_poller(adapter, poll_until_stopped)
    await _finish(adapter, task)

    assert not adapter.has_fatal_error


@pytest.mark.asyncio
async def test_bridge_exit_fatal_code_is_kept():
    adapter = _make_adapter()

    async def poll_after_bridge_died():
        adapter._set_fatal_error("whatsapp_bridge_exited", "bridge exited (code 1)", retryable=True)

    task = await _start_poller(adapter, poll_after_bridge_died)
    await _finish(adapter, task)

    assert adapter.fatal_error_code == "whatsapp_bridge_exited"


@pytest.mark.asyncio
async def test_superseded_poll_task_is_ignored():
    adapter = _make_adapter()

    async def returning_poll():
        return

    old_task = await _start_poller(adapter, returning_poll)
    adapter._poll_task = None  # a newer lifecycle already replaced it
    await _finish(adapter, old_task)

    assert not adapter.has_fatal_error

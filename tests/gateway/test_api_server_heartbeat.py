"""Heartbeat + runtime-metrics publishing for the API server adapter (#52323).

With no inbound requests nothing else touches gateway.status for this platform, so an
idle-but-healthy API server reads as frozen on dashboards keyed off ``updated_at``. The
adapter must republish on an interval while running. Bare-``__new__`` fixtures, matching
the house style pinned in test_api_server_run_idempotency.
"""

import asyncio

import pytest

from gateway.platforms.api_server import APIServerAdapter


def _bare_adapter(interval: float = 0.01) -> APIServerAdapter:
    adapter = object.__new__(APIServerAdapter)
    adapter._running = True
    adapter._fatal_error_message = None
    adapter._HEARTBEAT_INTERVAL_S = interval
    adapter._host = "127.0.0.1"
    adapter._port = 8642
    adapter._requests_total = 0
    adapter._requests_in_flight = 0
    adapter._last_request_at = None
    from gateway.platforms.base import Platform
    adapter.platform = Platform.API_SERVER
    adapter.published = []
    adapter._write_runtime_status_safe = (
        lambda context, **kwargs: adapter.published.append((context, kwargs)))
    return adapter


@pytest.mark.asyncio
async def test_heartbeat_loop_publishes_while_running():
    adapter = _bare_adapter()

    async def stop_soon():
        await asyncio.sleep(0.05)
        adapter._running = False

    stopper = asyncio.ensure_future(stop_soon())
    await adapter._heartbeat_loop()
    await stopper
    assert adapter.published, "heartbeat loop published nothing while running"
    for context, kwargs in adapter.published:
        assert context == "heartbeat"
        assert kwargs.get("platform_state") == "connected"


@pytest.mark.asyncio
async def test_heartbeat_loop_stops_when_disconnected_or_fatal():
    adapter = _bare_adapter()
    adapter._running = False
    await asyncio.wait_for(adapter._heartbeat_loop(), timeout=1)
    assert adapter.published == []

    adapter = _bare_adapter()
    adapter._fatal_error_message = "boom"
    await asyncio.wait_for(adapter._heartbeat_loop(), timeout=1)
    assert adapter.published == []


def test_request_metrics_counters():
    adapter = _bare_adapter()
    adapter._record_request_start()
    adapter._record_request_start()
    assert (adapter._requests_total, adapter._requests_in_flight) == (2, 2)
    assert adapter._last_request_at is not None
    adapter._record_request_end()
    adapter._record_request_end()
    adapter._record_request_end()  # must never go negative
    assert (adapter._requests_total, adapter._requests_in_flight) == (2, 0)

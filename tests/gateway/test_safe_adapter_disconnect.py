"""Regression tests: failed-connect path must call adapter.disconnect().

When adapter.connect() returns False or raises, the adapter may have
allocated resources (aiohttp.ClientSession, poll tasks, child
subprocesses) before giving up. Without a defensive disconnect() call
these leak and surface as "Unclosed client session" warnings at
process exit (seen on the 2026-04-18 18:08:16 gateway restart).

The fix: gateway/run.py wraps each adapter connect() with a safety-net
call to _safe_adapter_disconnect() in the failure branches.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


@pytest.fixture
def bare_runner():
    """A GatewayRunner shell that only needs to support _safe_adapter_disconnect."""
    return object.__new__(GatewayRunner)


@pytest.mark.asyncio
async def test_safe_disconnect_times_out_and_continues(bare_runner, monkeypatch, caplog):
    """A wedged adapter disconnect must not block gateway shutdown."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.001")
    adapter = MagicMock()

    async def hang():
        await asyncio.sleep(0.2)

    adapter.disconnect = AsyncMock(side_effect=hang)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await bare_runner._safe_adapter_disconnect(adapter, Platform.MATTERMOST)

    adapter.disconnect.assert_awaited_once()
    assert "Timed out after 0.0s while disconnecting mattermost adapter" in caplog.text


@pytest.mark.asyncio
async def test_safe_disconnect_detaches_cancellation_swallowing_disconnect(
    bare_runner, monkeypatch, caplog
):
    """A disconnect that catches cancellation cannot block fatal recovery.

    ``asyncio.wait_for`` cancels its child at the deadline but then waits for
    it to finish.  A half-closed transport can catch that cancellation while
    unwinding, so the runner must detach the old close task and continue to the
    reconnect queue instead of waiting indefinitely.
    """
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    adapter = MagicMock()
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def swallow_cancellation():
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue
        finished.set()

    adapter.disconnect = AsyncMock(side_effect=swallow_cancellation)
    operation = asyncio.create_task(
        bare_runner._safe_adapter_disconnect(adapter, Platform.MATTERMOST)
    )
    await started.wait()
    done, _pending = await asyncio.wait({operation}, timeout=0.2)
    try:
        assert operation in done
        assert "Timed out after 0.0s while disconnecting mattermost adapter" in caplog.text
    finally:
        # The implementation must detach rather than abandon the old task.
        # Release it here so this test leaves no cancellation-swallowing task
        # behind when it runs against the pre-fix implementation.
        release.set()
        await asyncio.wait({operation}, timeout=0.2)
        await asyncio.wait_for(finished.wait(), timeout=0.2)

"""Regression tests: the shutdown teardown loop must not hang on a wedged adapter.

`GatewayRunner._stop_impl()` tears down every adapter by awaiting
`cancel_background_tasks()` then `disconnect()`. Both calls can block
indefinitely when a platform's network state is half-dead (e.g. a wedged
Feishu/Lark WebSocket thread waiting on I/O). An unbounded await stalls the
whole shutdown past systemd's TimeoutStopSec; the resulting SIGKILL skips
atexit PID-file cleanup, so the next start dies with "PID file race lost"
(#14128).

The fix routes both teardown loops through `_bounded_adapter_teardown`,
which wraps each await in the existing per-adapter timeout budget
(HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT) and always returns.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


@pytest.fixture
def bare_runner():
    """A GatewayRunner shell that only needs _bounded_adapter_teardown."""
    return object.__new__(GatewayRunner)


@pytest.mark.asyncio
async def test_teardown_bounds_hanging_cancel(bare_runner, monkeypatch, caplog):
    """A wedged cancel_background_tasks() must time out, then disconnect runs."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.01")
    adapter = MagicMock()

    async def hang():
        await asyncio.sleep(0.2)

    adapter.cancel_background_tasks = AsyncMock(side_effect=hang)
    adapter.disconnect = AsyncMock(return_value=None)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await asyncio.wait_for(
            bare_runner._bounded_adapter_teardown(adapter, Platform.FEISHU),
            timeout=5.0,
        )

    assert "feishu background-task cancel timed out" in caplog.text
    # disconnect still attempted after the cancel timeout — forward progress.
    adapter.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_teardown_continues_after_cancellation_swallowing_background_cancel(
    bare_runner, monkeypatch, caplog
):
    """A stuck cancellation handler cannot prevent adapter disconnect.

    This models a platform task that catches ``CancelledError`` while it is
    unwinding.  The teardown deadline must release runner ownership promptly,
    then proceed to disconnect instead of waiting for that old task forever.
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

    adapter.cancel_background_tasks = AsyncMock(side_effect=swallow_cancellation)
    adapter.disconnect = AsyncMock(return_value=None)
    operation = asyncio.create_task(
        bare_runner._bounded_adapter_teardown(adapter, Platform.FEISHU)
    )
    await started.wait()
    done, _pending = await asyncio.wait({operation}, timeout=0.2)
    try:
        assert operation in done
        adapter.disconnect.assert_awaited_once()
        assert "feishu background-task cancel timed out" in caplog.text
    finally:
        release.set()
        await asyncio.wait({operation}, timeout=0.2)
        await asyncio.wait_for(finished.wait(), timeout=0.2)



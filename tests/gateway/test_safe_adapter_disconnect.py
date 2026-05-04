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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


@pytest.fixture
def bare_runner():
    """A GatewayRunner shell that only needs to support _safe_adapter_disconnect."""
    return object.__new__(GatewayRunner)


@pytest.mark.asyncio
async def test_safe_disconnect_calls_adapter_disconnect(bare_runner):
    """The helper forwards to adapter.disconnect()."""
    adapter = MagicMock()
    adapter.disconnect = AsyncMock(return_value=None)

    await bare_runner._safe_adapter_disconnect(adapter, Platform.TELEGRAM)

    adapter.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_disconnect_swallows_exceptions(bare_runner):
    """An exception in adapter.disconnect() must not propagate — the
    caller is already on an error path."""
    adapter = MagicMock()
    adapter.disconnect = AsyncMock(side_effect=RuntimeError("partial init"))

    # Must NOT raise
    await bare_runner._safe_adapter_disconnect(adapter, Platform.TELEGRAM)

    adapter.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_disconnect_handles_none_platform(bare_runner):
    """Logging path must tolerate platform=None."""
    adapter = MagicMock()
    adapter.disconnect = AsyncMock(side_effect=ValueError("nope"))

    await bare_runner._safe_adapter_disconnect(adapter, None)

    adapter.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_disconnect_abandons_on_timeout(bare_runner, monkeypatch):
    """A wedged adapter disconnect must be abandoned after the per-adapter
    timeout instead of blocking the entire shutdown (#19937)."""

    async def _hang_forever():
        await asyncio.sleep(3600)  # simulate a stuck websocket

    adapter = MagicMock()
    adapter.disconnect = _hang_forever

    # Patch the timeout to 0.05s so the test is fast
    monkeypatch.setattr(
        "gateway.run._ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT", 0.05
    )

    # Must NOT raise and must complete quickly
    await bare_runner._safe_adapter_disconnect(adapter, Platform.FEISHU)

    # If we get here, the timeout worked — no exception propagated


@pytest.mark.asyncio
async def test_stop_disconnect_abandons_wedged_adapter(monkeypatch):
    """In the stop() method, a wedged adapter must be abandoned after the
    per-adapter timeout so other adapters still get disconnected (#19937)."""
    from tests.gateway.restart_test_helpers import make_restart_runner

    runner, telegram_adapter = make_restart_runner()
    runner._restart_drain_timeout = 0.01  # force quick drain timeout

    call_order = []
    disconnect_timeout = 0.05

    # Override telegram adapter disconnect to track call
    async def _normal_disconnect():
        call_order.append("telegram_disconnect")

    telegram_adapter.disconnect = _normal_disconnect

    # Create a second (feishu) adapter that hangs
    async def _hang_forever():
        call_order.append("feishu_disconnect_start")
        await asyncio.sleep(3600)

    feishu_adapter = AsyncMock()
    feishu_adapter.disconnect = _hang_forever
    feishu_adapter.cancel_background_tasks = AsyncMock()

    runner.adapters[Platform.FEISHU] = feishu_adapter

    # Patch the timeout constant
    monkeypatch.setattr(
        "gateway.run._ADAPTER_DISCONNECT_TIMEOUT_SECS_DEFAULT", disconnect_timeout
    )

    with (patch("gateway.status.remove_pid_file"),
          patch("gateway.status.write_runtime_status"),
          patch("agent.auxiliary_client.shutdown_cached_clients")):
        await runner.stop()

    # The wedged feishu adapter's disconnect should have started
    assert "feishu_disconnect_start" in call_order
    # The normal telegram adapter should have completed
    assert "telegram_disconnect" in call_order

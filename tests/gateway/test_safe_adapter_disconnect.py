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
async def test_safe_disconnect_times_out_and_continues(bare_runner, monkeypatch, caplog):
    """A wedged adapter disconnect must not block gateway shutdown."""
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.001")
    adapter = MagicMock()

    async def hang():
        await asyncio.sleep(60)

    adapter.disconnect = AsyncMock(side_effect=hang)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        await bare_runner._safe_adapter_disconnect(adapter, Platform.FEISHU)

    adapter.disconnect.assert_awaited_once()
    assert "Timed out after 0.0s while disconnecting feishu adapter" in caplog.text

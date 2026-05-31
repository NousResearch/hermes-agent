"""A wedged adapter must never stall the gateway's shutdown path.

Adapter network ops reachable from ``_stop_impl`` are bounded at the call
site so one stuck socket can't hang shutdown into a SIGKILL that leaves
stale gateway.pid / gateway.lock. This suite covers each bounded op as
its commit lands (teardown, notify-sends, fatal-error disconnect).
"""

import asyncio
import types

import pytest

from gateway.run import GatewayRunner


def _runner():
    return object.__new__(GatewayRunner)


def _platform(name="telegram"):
    return types.SimpleNamespace(value=name, platform=name)


class _HangingDisconnect:
    """cancel_background_tasks() returns; disconnect() never completes."""

    async def cancel_background_tasks(self):
        return None

    async def disconnect(self):
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_hung_disconnect_does_not_stall_teardown(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT", "0.1")
    await asyncio.wait_for(
        _runner()._safe_adapter_teardown(_HangingDisconnect(), _platform()),
        timeout=2.0,
    )

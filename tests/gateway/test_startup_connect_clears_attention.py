"""A successful startup connect clears a stale ``needs_attention`` flag.

The reconnect loop's escalation is persisted in the runtime status file, which survives a gateway
restart. Only ``_install_reconnected_adapter`` cleared it, so a platform that recovered through a
restart (rather than an in-process reconnect) stayed "connected" with ``needs_attention: true`` and
an old ``retrying_since`` indefinitely.
"""
import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner


class _Adapter:
    send_path_degraded = False
    DEGRADED_STATUS_MESSAGE = "degraded"


@pytest.mark.asyncio
async def test_startup_connect_clears_needs_attention(monkeypatch):
    runner = object.__new__(GatewayRunner)
    writes = []
    monkeypatch.setattr(runner, "_update_platform_runtime_status", lambda key, **kw: writes.append((key, kw)))
    monkeypatch.setattr(runner, "_publish_primary_adapter", lambda platform, adapter: None)

    raw = [(Platform.WHATSAPP, _Adapter(), None, "ok", None)]
    connected = await runner._start_aggregate_connect_results(raw, [], [])

    assert connected == 1
    key, kw = writes[-1]
    assert key == Platform.WHATSAPP.value
    assert kw["platform_state"] == "connected"
    assert kw["needs_attention"] is False
    assert kw["retrying_since"] is None

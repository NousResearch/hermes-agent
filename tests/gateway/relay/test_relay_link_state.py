"""The relay reports its link to the connector: connected while it is up, not connected during an
outage, connected again once the transport's own re-dial succeeds.

``gateway_state.json`` (read by ``hermes gateway status``, the dashboard and the Desktop health pill)
and ``adapter.is_connected`` (the platform-actions gate) both follow the link. The adapter is built
through the real relay registration path over the real ``WebSocketRelayTransport``; only the
connector is fake, at its network boundary (``fake_connector.FakeConnector``).
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from gateway import relay
from gateway.config import PlatformConfig
from gateway.platform_registry import platform_registry
from gateway.status import publish_runtime_status, read_runtime_status
from tests.gateway.relay.fake_connector import FakeConnector
from tests.gateway.relay.test_relay_transient_recovery import TOKEN
from tests.waiting import async_wait_until


@pytest.fixture(autouse=True)
def _relay_env(monkeypatch):
    for k in ("GATEWAY_RELAY_URL", "GATEWAY_RELAY_ID", "GATEWAY_RELAY_SECRET", "GATEWAY_RELAY_DELIVERY_KEY",
              "GATEWAY_RELAY_ENDPOINT", "GATEWAY_RELAY_ROUTE_KEYS", "GATEWAY_RELAY_INSTANCE_ID",
              "GATEWAY_RELAY_WAKE_URL", "GATEWAY_RELAY_BOT_IDS", "GATEWAY_RELAY_IDP_TOKEN_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "slack")
    monkeypatch.setattr("gateway.run._load_gateway_config", dict, raising=False)
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda: TOKEN)
    platform_registry.unregister("relay")
    yield
    platform_registry.unregister("relay")


@pytest_asyncio.fixture
async def connector(monkeypatch):
    conn = FakeConnector()
    await conn.start()
    conn.install_http(monkeypatch)
    monkeypatch.setenv("GATEWAY_RELAY_URL", conn.url)
    yield conn
    await conn.stop()


def _relay_state() -> str:
    return ((read_runtime_status() or {}).get("platforms", {}).get("relay") or {}).get("state", "")


async def _connected_adapter():
    assert relay.self_provision_relay() is True
    assert relay.register_relay_adapter()
    adapter = platform_registry.get("relay").adapter_factory(PlatformConfig(enabled=True))
    assert await adapter.connect() is True
    publish_runtime_status(platform="relay", platform_state="connected")  # the runner's stamp after connect()
    await async_wait_until(lambda: _relay_state() == "connected", timeout=5.0, what="runner stamp written")
    return adapter


@pytest.mark.asyncio
async def test_connected_relay_reports_connected(connector):
    adapter = await _connected_adapter()
    try:
        assert adapter.is_connected is True
    finally:
        await adapter.disconnect()
    assert adapter.is_connected is False
    await async_wait_until(lambda: _relay_state() == "disconnected", timeout=5.0, what="relay reported disconnected")


@pytest.mark.asyncio
async def test_connector_outage_is_reported_until_the_link_returns(connector):
    adapter = await _connected_adapter()
    try:
        await connector.stop()  # the connector goes away; the transport's supervisor starts re-dialling
        await async_wait_until(lambda: adapter._transport._reader.done(), timeout=10.0, what="relay reader exited")

        assert adapter.is_connected is False
        await async_wait_until(lambda: _relay_state() != "connected", timeout=5.0,
                               what="relay with no live connector link reported as not connected")

        await connector.start()
        await async_wait_until(lambda: adapter.is_connected, timeout=10.0, what="relay re-dialled the connector")
        await async_wait_until(lambda: _relay_state() == "connected", timeout=5.0, what="relay reported connected")
    finally:
        await adapter.disconnect()

"""A relay without credentials must not keep dialing a connector that requires them.

Field report: hosted agents whose boot provision was refused (a revoked gatewayId, a
binding owned by another tenant, a logged-out Nous Portal session) kept dialing the
relay WebSocket with no bearer. The connector closed every upgrade 4401, so each agent
produced hundreds of rejected upgrades an hour, never recovered on its own, and its
status said nothing more specific than "unauthorized".

Production code end to end; only the connector is fake, at its network boundary
(``tests/gateway/relay/fake_connector.py``: a real WebSocket server for ``/relay`` and
an ``urlopen`` stub for ``/relay/provision``).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time

import pytest
import pytest_asyncio
import websockets

from gateway import relay
from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import platform_registry
from gateway.relay import ws_transport
from gateway.run_startup import GatewayStartupMixin
from tests.gateway.relay.fake_connector import FakeConnector, relay_refusal

TOKEN = "nas-identity-token-must-never-be-logged"
REVOKED = relay_refusal(409, "gatewayId was revoked and cannot be re-provisioned; choose a new gatewayId")


@pytest.fixture(autouse=True)
def _relay_env(monkeypatch):
    for k in ("GATEWAY_RELAY_URL", "GATEWAY_RELAY_ID", "GATEWAY_RELAY_SECRET", "GATEWAY_RELAY_DELIVERY_KEY",
              "GATEWAY_RELAY_ENDPOINT", "GATEWAY_RELAY_ROUTE_KEYS", "GATEWAY_RELAY_INSTANCE_ID",
              "GATEWAY_RELAY_WAKE_URL", "GATEWAY_RELAY_BOT_IDS", "GATEWAY_RELAY_IDP_TOKEN_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "slack")
    monkeypatch.setattr("gateway.run._load_gateway_config", dict, raising=False)
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda: TOKEN)
    monkeypatch.setattr(ws_transport, "_CREDENTIALS_REQUIRED_URLS", set(), raising=False)
    monkeypatch.setattr(relay, "_last_provision_problem", None, raising=False)
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


def _build_adapter():
    """What the reconnect watcher does on every pass: a fresh adapter from the registered factory."""
    return platform_registry.get("relay").adapter_factory(PlatformConfig(enabled=True))


def _boot_with_refused_provision(connector) -> None:
    connector.provision_script = [REVOKED]
    assert relay.self_provision_relay() is False
    assert relay.relay_connection_auth() == (None, None)
    assert relay.register_relay_adapter()


@pytest.mark.asyncio
async def test_refused_provision_dials_the_connector_once_not_forever(connector):
    _boot_with_refused_provision(connector)
    first = _build_adapter()
    try:
        assert await first.connect() is False
        # No dialer may be armed: neither the backoff supervisor nor a fresh-token retry.
        assert not first._transport._dialer_running(), "the transport armed an unauthenticated re-dial"
    finally:
        await first.disconnect()
    assert connector.ws_auth == ["rejected"]

    # The reconnect watcher's later passes build fresh adapters; none of them dials again.
    for _ in range(3):
        again = _build_adapter()
        try:
            assert await again.connect() is False
        finally:
            await again.disconnect()
    assert connector.ws_auth == ["rejected"]


@pytest.mark.asyncio
async def test_status_names_the_connector_refusal_and_stays_retryable(connector):
    _boot_with_refused_provision(connector)
    adapter = _build_adapter()
    try:
        assert await adapter.connect() is False
        assert adapter.fatal_error_code == "relay_credentials_missing"
        assert adapter.fatal_error_retryable is True
        assert "gatewayId was revoked" in adapter.fatal_error_message
        assert TOKEN not in adapter.fatal_error_message
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_credentials_from_the_provision_retry_reconnect_without_restart(connector, monkeypatch):
    monkeypatch.setattr(relay, "_PROVISION_REFUSAL_RETRY_S", 0.01, raising=False)
    monkeypatch.setattr(relay, "_PROVISION_RETRY_DELAYS_S", (0.01,), raising=False)
    _boot_with_refused_provision(connector)

    refused = _build_adapter()
    try:
        assert await refused.connect() is False
    finally:
        await refused.disconnect()

    # The operator fixes the binding; the background retry's next provision succeeds.
    runner = object.__new__(GatewayStartupMixin)
    runner._running = True
    runner._background_tasks = set()
    runner._failed_platforms = {Platform.RELAY: {"attempts": 4, "next_retry": time.monotonic() + 300}}
    runner._retain_background_task = lambda task: (runner._background_tasks.add(task), task)[1]
    GatewayStartupMixin._start_relay_provision_retry(runner)
    await asyncio.wait_for(asyncio.gather(*runner._background_tasks), timeout=10)
    assert relay.relay_connection_auth() == (connector.gateway_id, connector.secret)
    assert relay.relay_provision_problem() is None
    # The reconnect watcher's queued relay retry is pulled forward, not left on its backoff.
    assert runner._failed_platforms[Platform.RELAY]["next_retry"] == 0.0
    assert runner._failed_platforms[Platform.RELAY]["retry_now"] is True

    adapter = _build_adapter()
    try:
        assert await adapter.connect() is True
        assert connector.ws_auth == ["rejected", "ok"]
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_re_registering_the_relay_adapter_tries_an_uncredentialed_dial_again(connector):
    """The refusal is remembered per process, not forever: a re-registration (config
    reload, restart) dials once more, so a connector switched to accept uncredentialed
    upgrades is picked up."""
    _boot_with_refused_provision(connector)
    refused = _build_adapter()
    try:
        assert await refused.connect() is False
    finally:
        await refused.disconnect()
    assert connector.ws_auth == ["rejected"]

    assert relay.register_relay_adapter()
    again = _build_adapter()
    try:
        assert await again.connect() is False
    finally:
        await again.disconnect()
    assert connector.ws_auth == ["rejected", "rejected"]


@pytest.mark.asyncio
async def test_a_connector_that_accepts_uncredentialed_upgrades_still_connects(monkeypatch):
    """Dev connectors and harnesses run with no per-gateway secret: nothing changes for them."""
    async def open_connector(ws):
        async for raw in ws:
            for line in str(raw).split("\n"):
                if line.strip():
                    await ws.send(
                        '{"type":"descriptor","descriptor":{"contract_version":1,"platform":"slack",'
                        '"label":"Slack","max_message_length":4000,"supports_draft_streaming":false,'
                        '"supports_edit":true,"supports_threads":true,"markdown_dialect":"slack_mrkdwn",'
                        '"len_unit":"chars","supported_ops":["send"]}}\n'
                    )

    server = await websockets.serve(open_connector, "127.0.0.1", 0)
    port = next(iter(server.sockets)).getsockname()[1]
    monkeypatch.setenv("GATEWAY_RELAY_URL", f"ws://127.0.0.1:{port}/relay")
    try:
        assert relay.register_relay_adapter()
        adapter = _build_adapter()
        try:
            assert await adapter.connect() is True
        finally:
            await adapter.disconnect()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_credentialed_4401_keeps_the_existing_revocation_path(connector, caplog):
    """A 4401 on an upgrade that DID carry credentials is not this guard's business."""
    assert relay.self_provision_relay() is True
    assert relay.register_relay_adapter()
    connector.secret = "rotated-elsewhere" * 4  # the connector no longer accepts our secret
    caplog.set_level(logging.WARNING)
    adapter = _build_adapter()
    try:
        with contextlib.suppress(Exception):  # how a bad secret surfaces is not this test's subject
            await adapter.connect()
        assert adapter.fatal_error_code != "relay_credentials_missing"
    finally:
        await adapter.disconnect()
    assert ws_transport._CREDENTIALS_REQUIRED_URLS == set()

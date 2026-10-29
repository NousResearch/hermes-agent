"""One transient relay failure must not become a lasting one.

Field report: an egress proxy between the gateway and the relay connector briefly
answered 403 (its own DNS lookup timed out). The same request would have worked
seconds later, but on main:

1. boot provisioning (``POST /relay/provision``) is attempted once; afterwards the
   gateway dials the relay with no credentials and is closed 4401 every 5 minutes
   until someone restarts it;
2. a media upload (``POST /relay/media``) is attempted once;
3. the dropped attachment is reported as a successful send;
4. the logs cannot tell the proxy's 403 from a relay 403.

Every test drives production code end to end. Only the connector is fake, at its
network boundary (``tests/gateway/relay/fake_connector.py``): a real WebSocket
server for ``/relay`` and an ``urlopen`` stub for the two HTTP routes.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest
import pytest_asyncio

from gateway import relay
from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import platform_registry
from gateway.run_startup import GatewayStartupMixin
from tests.gateway.relay.fake_connector import (
    FakeConnector,
    proxy_403,
    relay_refusal,
    store_unavailable,
)

TOKEN = "nas-identity-token-must-never-be-logged"


@pytest.fixture(autouse=True)
def _relay_env(monkeypatch):
    for k in ("GATEWAY_RELAY_URL", "GATEWAY_RELAY_ID", "GATEWAY_RELAY_SECRET", "GATEWAY_RELAY_DELIVERY_KEY",
              "GATEWAY_RELAY_ENDPOINT", "GATEWAY_RELAY_ROUTE_KEYS", "GATEWAY_RELAY_INSTANCE_ID",
              "GATEWAY_RELAY_WAKE_URL", "GATEWAY_RELAY_BOT_IDS", "GATEWAY_RELAY_IDP_TOKEN_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "slack")
    monkeypatch.setattr("gateway.run._load_gateway_config", dict, raising=False)
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda: TOKEN)
    # Upload retry pacing is a production constant; collapse it so tests run in
    # milliseconds. raising=False: on a tree without the retry the constant is absent
    # and the tests must fail on behaviour, not on this fixture.
    monkeypatch.setattr("gateway.relay.media._UPLOAD_RETRY_DELAYS_S", (0.0, 0.0), raising=False)
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


def _runner_with_relay_queued():
    """The runner state after relay failed to connect at boot: queued for a retry 5 minutes out."""
    runner = object.__new__(GatewayStartupMixin)
    runner._running = True
    runner._background_tasks = set()
    runner._failed_platforms = {Platform.RELAY: {"attempts": 4, "next_retry": time.monotonic() + 300}}
    runner._retain_background_task = lambda task: (runner._background_tasks.add(task), task)[1]
    return runner


async def _provisioned_adapter(connector):
    assert relay.self_provision_relay() is True
    assert relay.register_relay_adapter()
    adapter = _build_adapter()
    assert await adapter.connect() is True
    return adapter


# ─────────────────────────── 1. provisioning ───────────────────────────


@pytest.mark.asyncio
async def test_boot_provision_proxy_403_recovers_without_restart(connector, caplog, monkeypatch):
    monkeypatch.setattr(relay, "_PROVISION_RETRY_DELAYS_S", (0.01,), raising=False)
    connector.provision_script = [proxy_403()]
    caplog.set_level(logging.INFO)

    # Boot: the proxy answers the one provision attempt with its own 403 page.
    assert relay.self_provision_relay() is False
    assert relay.relay_connection_auth() == (None, None)
    assert relay.register_relay_adapter()

    runner = _runner_with_relay_queued()
    GatewayStartupMixin._start_relay_provision_retry(runner)
    assert runner._background_tasks, "a relay configured without credentials must keep provisioning"
    await asyncio.wait_for(asyncio.gather(*runner._background_tasks), timeout=10)

    assert connector.provision_calls == 2
    assert relay.relay_connection_auth() == (connector.gateway_id, connector.secret)
    # The reconnect watcher is told to dial now rather than at its 5-minute backoff.
    assert runner._failed_platforms[Platform.RELAY]["next_retry"] <= time.monotonic()

    # The watcher's next pass builds a fresh adapter; it now dials WITH credentials.
    adapter = _build_adapter()
    try:
        assert await adapter.connect() is True
        assert connector.ws_auth[-1] == "ok"
        assert adapter.descriptor.platform == "slack"
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_provision_retry_paces_transient_failures_and_throttles_refusals(connector):
    connector.provision_script = [
        proxy_403(),                                                  # proxy, no relay reason: transient
        relay_refusal(403, "enterprise identity lifecycle is not confirmed"),  # relay's own reason
        relay_refusal(503, "upstream unavailable"),                   # 5xx: transient
    ]
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    provisioned = []
    assert await relay.retry_self_provision(lambda: provisioned.append(True), sleep=fake_sleep) is True

    delays, refusal = relay._PROVISION_RETRY_DELAYS_S, relay._PROVISION_REFUSAL_RETRY_S
    # before #1, after the proxy 403, after the refusal (throttled), after the 503
    assert slept == [delays[0], delays[1], refusal, delays[-1]]
    assert refusal > max(delays), "a reasoned refusal must be retried less often than a blip"
    assert connector.provision_calls == 4 and provisioned == [True]


@pytest.mark.asyncio
async def test_no_retry_when_relay_unconfigured_or_credentials_pinned(connector, monkeypatch):
    runner = _runner_with_relay_queued()
    monkeypatch.setenv("GATEWAY_RELAY_ID", "gw-pinned")
    monkeypatch.setenv("GATEWAY_RELAY_SECRET", "pinned-secret")
    GatewayStartupMixin._start_relay_provision_retry(runner)
    monkeypatch.delenv("GATEWAY_RELAY_SECRET")
    monkeypatch.delenv("GATEWAY_RELAY_URL")
    GatewayStartupMixin._start_relay_provision_retry(runner)
    assert runner._background_tasks == set()
    assert connector.provision_calls == 0


@pytest.mark.asyncio
async def test_provision_failure_log_names_the_proxy_and_never_the_token(connector, caplog):
    connector.provision_script = [proxy_403()]
    caplog.set_level(logging.INFO)
    assert relay.self_provision_relay() is False
    text = caplog.text
    assert "server=squid/6.10" in text
    assert "x-squid-error=ERR_DNS_FAIL 0" in text
    assert "ERROR: The requested URL could not be retrieved" in text  # body excerpt
    assert "connector returned HTTP 403" not in text  # the connector never saw the request
    assert TOKEN not in text and connector.bearers_seen and all(b not in text for b in connector.bearers_seen)


# ─────────────────────────── 2-4. media ───────────────────────────


@pytest.mark.asyncio
async def test_upload_403_once_then_succeeds(connector, tmp_path):
    adapter = await _provisioned_adapter(connector)
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 report")
    connector.media_script = [proxy_403()]
    try:
        result = await adapter.send_document("C0REPORTS", str(pdf), file_name="report.pdf")
    finally:
        await adapter.disconnect()
    assert result.success is True
    assert connector.media_calls == 2
    [media] = connector.ops("send_media")
    assert media["source_url"].endswith("/relay/media/m-2")
    assert connector.ops("send") == []  # no fallback notice: the file went through


@pytest.mark.parametrize("status, reason", [(401, "unauthenticated"), (413, "too large"), (400, "bad request")])
@pytest.mark.asyncio
async def test_upload_does_not_retry_final_relay_answers(connector, tmp_path, status, reason):
    adapter = await _provisioned_adapter(connector)
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 report")
    connector.media_script = [relay_refusal(status, reason)] * 3
    try:
        result = await adapter.send_document("C0REPORTS", str(pdf))
    finally:
        await adapter.disconnect()
    assert connector.media_calls == 1
    assert result.success is False


@pytest.mark.asyncio
async def test_upload_failing_every_retry_reports_failure_with_one_notice(connector, tmp_path, caplog):
    adapter = await _provisioned_adapter(connector)
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 report")
    connector.media_script = [proxy_403(), store_unavailable(), proxy_403()]
    caplog.set_level(logging.INFO)
    try:
        result = await adapter.send_document("C0REPORTS", str(pdf), caption="Q3 report attached")
    finally:
        await adapter.disconnect()

    assert connector.media_calls == 3
    assert result.success is False
    assert "403" in result.error
    # Same UX as before: the caption and a notice still reach the channel, once.
    [notice] = connector.ops("send")
    assert "Q3 report attached" in notice["content"]
    assert connector.ops("send_media") == []
    # Upload logs carry the proxy's identity and a body excerpt, never the bearer or secret.
    assert "server=squid/6.10" in caplog.text and "ERROR: The requested URL" in caplog.text
    assert connector.secret not in caplog.text
    assert all(b not in caplog.text for b in connector.bearers_seen)


@pytest.mark.asyncio
async def test_turn_dispatch_posts_one_notice_for_a_dropped_attachment(connector, tmp_path):
    """The interactive path posts its own failure notice for a failed media result.
    The relay fallback already posted one, so the result must say so."""
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    adapter = await _provisioned_adapter(connector)
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 report")
    connector.media_script = [proxy_403()] * 3
    recorded = []
    event = MessageEvent(text="", source=SessionSource(platform=Platform.SLACK, chat_id="C0REPORTS"))
    try:
        await adapter._deliver_media_attachments(
            event, [(str(pdf), False)], [], force_document_attachments=False, human_delay=0,
            metadata={}, record_delivery=recorded.append)
    finally:
        await adapter.disconnect()
    assert [r.success for r in recorded] == [False]
    assert len(connector.ops("send")) == 1

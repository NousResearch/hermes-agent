"""A cron run whose MEDIA attachment never reaches the relay is not ``ok``.

Field report: a scheduled job posts a PDF report through a relay-fronted Slack.
An egress proxy refused the attachment upload once (its own 403). The text
arrived with a "couldn't send file" notice where the PDF should have been, and
the run recorded ``last_status: ok``, so nobody noticed the missing report.

Composed through the gateway's real cron delivery path: the delivery queue
drain -> ``_deliver_result`` -> the live ``RelayAdapter`` -> a real
``WebSocketRelayTransport`` and ``RelayMediaClient``. Only the connector is
fake, at its network boundary.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest

from gateway import relay
from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import platform_registry
from tests.gateway.relay.fake_connector import FakeConnector, proxy_403

CHAT = "C0REPORTS"


@pytest.fixture
def live_relay(monkeypatch):
    """A provisioned, connected relay adapter on a gateway loop thread."""
    for k in ("GATEWAY_RELAY_ID", "GATEWAY_RELAY_SECRET", "GATEWAY_RELAY_DELIVERY_KEY", "GATEWAY_RELAY_ENDPOINT",
              "GATEWAY_RELAY_ROUTE_KEYS", "GATEWAY_RELAY_INSTANCE_ID", "GATEWAY_RELAY_WAKE_URL",
              "GATEWAY_RELAY_BOT_IDS", "GATEWAY_RELAY_IDP_TOKEN_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GATEWAY_RELAY_PLATFORMS", "slack")
    monkeypatch.setattr("gateway.run._load_gateway_config", dict, raising=False)
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda: "nas-token")
    monkeypatch.setattr("gateway.relay.media._UPLOAD_RETRY_DELAYS_S", (0.0, 0.0), raising=False)

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    def run(coro):
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=20)

    connector = FakeConnector()
    run(connector.start())
    connector.install_http(monkeypatch)
    monkeypatch.setenv("GATEWAY_RELAY_URL", connector.url)
    platform_registry.unregister("relay")
    assert relay.self_provision_relay() is True
    assert relay.register_relay_adapter()
    adapter = platform_registry.get("relay").adapter_factory(PlatformConfig(enabled=True))
    assert run(adapter.connect()) is True
    try:
        yield connector, adapter, loop
    finally:
        run(adapter.disconnect())
        run(connector.stop())
        platform_registry.unregister("relay")
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def _gateway_config():
    config = MagicMock()
    config.platforms = {Platform.RELAY: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    return config


def _drain(job, content, adapter, loop, execution_id):
    from cron import scheduler
    from cron.delivery_queue import enqueue, get_status

    enqueue(execution_id, job, content)
    with patch("gateway.config.load_gateway_config", return_value=_gateway_config()), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("cron.scheduler_delivery._maybe_mirror_cron_delivery", lambda *a, **k: None):
        assert scheduler.drain_delivery_queue({Platform.RELAY: adapter}, loop) == 1
    return get_status(execution_id)


def _job():
    return {"id": "job-q3", "name": "Q3 report", "deliver": "origin",
            "origin": {"platform": "slack", "chat_id": CHAT}}


def test_dropped_attachment_fails_the_queued_cron_delivery(live_relay, tmp_path):
    connector, adapter, loop = live_relay
    pdf = tmp_path / "q3-report.pdf"
    pdf.write_bytes(b"%PDF-1.4 q3")
    connector.media_script = [proxy_403()] * 3

    row = _drain(_job(), f"Q3 numbers are in.\n\nMEDIA:{pdf}", adapter, loop, "exec-q3-1")

    assert connector.media_calls == 3                      # the upload was retried, then gave up
    sends = connector.ops("send")
    assert sends and sends[0]["content"].startswith("Q3 numbers are in.")  # the text still went out
    assert connector.ops("send_media") == []
    assert row["status"] == "failed", row
    assert "q3-report.pdf" in row["error"] and "403" in row["error"]


def test_transient_upload_403_still_delivers_the_report(live_relay, tmp_path):
    connector, adapter, loop = live_relay
    pdf = tmp_path / "q3-report.pdf"
    pdf.write_bytes(b"%PDF-1.4 q3")
    connector.media_script = [proxy_403()]

    row = _drain(_job(), f"Q3 numbers are in.\n\nMEDIA:{pdf}", adapter, loop, "exec-q3-2")

    assert connector.media_calls == 2
    assert len(connector.ops("send_media")) == 1
    assert row["status"] == "delivered", row


def test_inline_run_records_delivery_failed_not_ok(live_relay, tmp_path):
    """The inline lane (no queue) feeds the same error into the job record."""
    from cron.jobs import create_job, get_job, mark_job_run
    from cron.scheduler import _deliver_result

    connector, adapter, loop = live_relay
    pdf = tmp_path / "q3-report.pdf"
    pdf.write_bytes(b"%PDF-1.4 q3")
    connector.media_script = [proxy_403()] * 3
    job = create_job("Post the Q3 report", "every 1d", name="Q3 report", deliver="origin",
                     origin={"platform": "slack", "chat_id": CHAT})
    with patch("gateway.config.load_gateway_config", return_value=_gateway_config()), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("cron.scheduler_delivery._maybe_mirror_cron_delivery", lambda *a, **k: None):
        error = _deliver_result(job, f"Q3 numbers are in.\n\nMEDIA:{pdf}",
                                adapters={Platform.RELAY: adapter}, loop=loop)
    mark_job_run(job["id"], True, delivery_error=error)
    record = get_job(job["id"])
    assert record["last_status"] == "delivery_failed"
    assert "403" in record["last_delivery_error"]

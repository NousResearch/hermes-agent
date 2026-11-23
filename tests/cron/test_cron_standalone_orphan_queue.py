"""A cron payload orphaned by transport is kept for redelivery when both lanes fail.

Follow-up to the #125363 / #126156 gates: those queue a payload the live lane rejected retryably.
When no live adapter is ready at all and the standalone send raises before the request leaves this
host (connection refused, DNS, TLS handshake), the payload is in the same position — nothing was
sent anywhere — yet nothing queued it, so the run only reported the error. A platform verdict on
the live lane ("chat not found") still queues nothing: the chat itself refused.
"""

import asyncio
import threading
import time

import pytest

import cron.scheduler_delivery as sd
from gateway import delivery_ledger as dl
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import SendResult
from persistence_assembly import profile_scope


@pytest.fixture(autouse=True)
def _fresh_ledger(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_scope", lambda: profile_scope(home=home))
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)
    monkeypatch.setattr(sd, "_maybe_mirror_cron_delivery", lambda *a, **k: None)


@pytest.fixture
def gateway_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)


def _target(*, live_error=None, loop=None):
    class Transport:
        adapter = type("Adapter", (), {"_owner_profile": "primary"})()
        is_relay = False

        async def send(self, platform, chat_id, content, metadata=None):
            return SendResult(success=False, error=live_error)

    fields = {name: None for name in sd._TargetDelivery.__dataclass_fields__}
    fields.update(job={"id": "job-1"}, platform=Platform.TELEGRAM, platform_name="telegram",
                  chat_id="-100", thread_id="42", transport=Transport(), config=GatewayConfig(),
                  loop=loop, target_adapters={}, mirror_text="", origin={},
                  pconfig=type("PConfig", (), {"token": "tok"})())
    return sd._TargetDelivery(**fields)


def _both_lanes(t, monkeypatch, standalone):
    import gateway.outbound.send as smt
    monkeypatch.setattr(smt, "_send_to_platform", standalone)
    target_errors, delivery_errors = [], []
    assert not sd._deliver_via_live_adapter(
        t, "the report", [], target_errors=target_errors, delivery_errors=delivery_errors,
        unverified_targets=[])
    sd._deliver_standalone(t, "the report", [], target_errors, delivery_errors)
    return delivery_errors


def _refused(*args, **kwargs):
    raise ConnectionRefusedError("connection refused")


def _claim():
    return [row["content"] for row in dl.sweep_failed_for_runtime("telegram", now=time.time() + 600, profile="primary")]


def test_no_live_adapter_and_a_transport_failure_queue_the_payload(monkeypatch):
    t = _target()  # live_adapter_ready=False: the live lane never ran, so there is no live_error
    errors = _both_lanes(t, monkeypatch, _refused)

    assert t.standalone_never_left and not t.standalone_in_flight
    assert any("queued text" in e and "never left this host" in e for e in errors), errors
    assert _claim() == ["the report"]


def test_a_platform_verdict_on_the_live_lane_still_queues_nothing(monkeypatch, gateway_loop):
    t = _target(live_error="chat not found", loop=gateway_loop)
    errors = _both_lanes(t, monkeypatch, _refused)

    assert t.live_error == "chat not found"
    assert not any("queued" in e for e in errors), errors
    assert _claim() == []


def test_a_result_dict_failure_is_not_a_transport_orphan(monkeypatch):
    async def _answered(platform, pconfig, chat_id, message, **kwargs):
        return {"error": "You must pass the token from BotFather"}

    t = _target()
    monkeypatch.setattr("gateway.outbound.send._send_to_platform", _answered)
    sd._standalone_send(t, "the report", [])

    assert not t.standalone_in_flight and not t.standalone_never_left

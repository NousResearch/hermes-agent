"""A cron payload the live lane rejected as reconnect-only is not lost when standalone fails (#125363).

``send_path_degraded`` means the adapter that owns the connection will deliver after it reconnects.
The router raises that rejection (``DeliveryRouter._deliver_to_platform`` -> RuntimeError), so it
reaches the cron lane on the EXCEPTION arm of ``_deliver_via_live_adapter``. A satellite profile's
cron worker has no platform token, so the standalone fallback fails; the payload must then wait in
the delivery ledger for that adapter's post-reconnect sweep. Any other failure still falls through
to standalone exactly as before and queues nothing.
"""

import asyncio
import threading

import pytest

import cron.scheduler_delivery as sd
import cron.scheduler_delivery_continuation as continuation
from gateway import delivery_ledger as dl
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import SendResult


@pytest.fixture(autouse=True)
def _fresh_ledger(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)
    monkeypatch.setattr(continuation, "_maybe_mirror_cron_delivery", lambda *a, **k: None)


@pytest.fixture
def gateway_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)


def _deliver_through_router(monkeypatch, loop, *, live_error: str):
    """Run the live lane against a transport whose send is rejected with ``live_error`` (the router
    raises it), then the standalone lane on a token-less worker. Returns (standalone calls, errors)."""
    class Transport:
        adapter = type("Adapter", (), {"_owner_profile": "satellite"})()
        is_relay = False

        async def send(self, platform, chat_id, content, metadata=None):
            return SendResult(success=False, error=live_error, retryable=live_error == "send_path_degraded")

    fields = {name: None for name in sd._TargetDelivery.__dataclass_fields__}
    fields.update(job={"id": "job-1"}, platform=Platform.TELEGRAM, platform_name="telegram", chat_id="-100",
                  thread_id="42", transport=Transport(), config=GatewayConfig(), loop=loop,
                  target_adapters={}, mirror_text="", origin={})
    t = sd._TargetDelivery(**fields)
    standalone_calls = []
    monkeypatch.setattr(
        sd, "_standalone_send",
        lambda t, content, media: standalone_calls.append(content) or (None, "You must pass the token from BotFather"))
    target_errors, delivery_errors = [], []
    assert not sd._deliver_via_live_adapter(
        t, "the report", [], target_errors=target_errors, delivery_errors=delivery_errors, unverified_targets=[])
    sd._deliver_standalone(t, "the report", [], target_errors, delivery_errors)
    return standalone_calls, delivery_errors


def test_reconnect_only_rejection_survives_a_failed_standalone_for_the_sweep(monkeypatch, gateway_loop):
    standalone_calls, errors = _deliver_through_router(monkeypatch, gateway_loop, live_error="send_path_degraded")
    assert standalone_calls == ["the report"]  # standalone still gets its chance first
    assert any("queued text for telegram:-100:42" in e for e in errors)
    claimed = dl.sweep_failed_for_runtime("telegram", profile="satellite")
    assert [(row["chat_id"], row["thread_id"], row["content"]) for row in claimed] == [("-100", "42", "the report")]


def test_other_live_failures_still_fall_to_standalone_and_queue_nothing(monkeypatch, gateway_loop):
    standalone_calls, errors = _deliver_through_router(monkeypatch, gateway_loop, live_error="chat not found")
    assert standalone_calls == ["the report"]
    assert any("BotFather" in e for e in errors) and not any("queued" in e for e in errors)
    assert dl.sweep_failed_for_runtime("telegram", profile="satellite") == []

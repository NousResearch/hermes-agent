"""A cron payload the live lane rejected as reconnect-only is not lost when standalone fails (#125363).

``send_path_degraded`` means the adapter that owns the connection will deliver after it reconnects.
A satellite profile's cron worker has no platform token, so the standalone fallback fails; the payload
must then wait in the delivery ledger for that adapter's post-reconnect sweep. A profile whose
standalone send works keeps delivering immediately, attachments included.
"""

import pytest

import cron.scheduler_delivery as sd
from gateway import delivery_ledger as dl


@pytest.fixture(autouse=True)
def _fresh_ledger(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)
    monkeypatch.setattr(sd, "_maybe_mirror_cron_delivery", lambda *a, **k: None)


def _target(*, live_error="send_path_degraded", relay=False, thread_id="42", profile="satellite"):
    adapter = type("Adapter", (), {"_owner_profile": profile})()
    transport = type("Transport", (), {"adapter": adapter, "is_relay": relay})()
    fields = {name: None for name in sd._TargetDelivery.__dataclass_fields__}
    fields.update(job={"id": "job-1"}, platform_name="telegram", chat_id="-100", thread_id=thread_id,
                  transport=transport, mirror_text="", origin={}, live_error=live_error)
    return sd._TargetDelivery(**fields)


def _standalone(monkeypatch, result, err):
    sent = []
    monkeypatch.setattr(sd, "_standalone_send", lambda t, content, media: sent.append(media) or (result, err))
    return sent


def _deliver(t, media=()):
    target_errors, delivery_errors = [], []
    sd._deliver_standalone(t, "the report", list(media), target_errors, delivery_errors)
    return delivery_errors


def test_failed_standalone_queues_the_payload_for_the_reconnect_sweep(monkeypatch):
    _standalone(monkeypatch, None, "You must pass the token from BotFather")
    errors = _deliver(_target())
    assert any("queued text for telegram:-100:42" in e for e in errors)
    claimed = dl.sweep_failed_for_runtime("telegram", profile="satellite")
    assert [(row["chat_id"], row["thread_id"], row["content"]) for row in claimed] == [("-100", "42", "the report")]


def test_working_standalone_still_delivers_now_and_queues_nothing(monkeypatch):
    sent = _standalone(monkeypatch, {"success": True}, None)
    assert _deliver(_target(), media=["chart.png"]) == []
    assert sent == [["chart.png"]]
    assert dl.sweep_failed_for_runtime("telegram", profile="satellite") == []


@pytest.mark.parametrize("target", [_target(live_error="chat not found"), _target(live_error=None),
                                    _target(relay=True)])
def test_other_failures_are_not_queued(monkeypatch, target):
    _standalone(monkeypatch, None, "boom")
    errors = _deliver(target)
    assert not any("queued" in e for e in errors)
    assert dl.sweep_failed_for_runtime("telegram", profile="satellite") == []


def test_dropped_attachments_are_reported(monkeypatch):
    _standalone(monkeypatch, None, "You must pass the token from BotFather")
    errors = _deliver(_target(), media=["a.png", "b.pdf"])
    assert any("2 attachment(s) not queued" in e for e in errors)


def test_failure_reports_name_the_thread():
    assert _target(thread_id="42").where == "telegram:-100:42"
    assert _target(thread_id=None).where == "telegram:-100"


def test_live_rejection_raised_by_the_router_is_queued(monkeypatch):
    import asyncio
    import threading

    from gateway.config import GatewayConfig, Platform
    from gateway.platforms.base import SendResult

    class Transport:
        adapter = type("Adapter", (), {"_owner_profile": "satellite"})()
        is_relay = False

        async def send(self, platform, chat_id, content, metadata=None):
            return SendResult(success=False, error="send_path_degraded", retryable=True)

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    try:
        t = _target()
        t.live_error = None
        t.platform, t.transport, t.config, t.loop, t.target_adapters = (
            Platform.TELEGRAM, Transport(), GatewayConfig(), loop, {})
        _standalone(monkeypatch, None, "You must pass the token from BotFather")
        target_errors, delivery_errors = [], []
        assert not sd._deliver_via_live_adapter(
            t, "the report", [], target_errors=target_errors, delivery_errors=delivery_errors,
            unverified_targets=[])
        sd._deliver_standalone(t, "the report", [], target_errors, delivery_errors)
    finally:
        loop.call_soon_threadsafe(loop.stop)
    claimed = dl.sweep_failed_for_runtime("telegram", profile="satellite")
    assert [(row["chat_id"], row["thread_id"], row["content"]) for row in claimed] == [("-100", "42", "the report")]

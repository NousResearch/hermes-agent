"""Cluster gates on cron's standalone fallback, follow-up to the #125363 merge (#126156):

1. ANY retryable live-lane rejection — not just the reconnect-only ``send_path_degraded``
   string — keeps the payload for redelivery when standalone definitively fails. The adapter's
   ``SendResult.retryable`` verdict rides ``DeliverySendError.send_retryable`` because the
   string alone is ambiguous: Telegram's "Not connected" is retryable while the bot client
   rebuilds and permanent when the fatal is not.
2. A standalone TIMEOUT is never queued: the dispatch shield keeps a timed-out send
   un-cancelled ("the send may still be in flight"), so a post-reconnect replay would
   duplicate it. Possible loss is surfaced loudly instead of certain duplication.
3. A credential-less (satellite) worker skips the guaranteed-to-fail-closed standalone attempt
   and goes straight to the queue gate — the check reads the exact value the dispatch passes
   (``pconfig.token`` for Telegram), never a re-derived one.
"""

import asyncio
import threading
import time

import pytest

import cron.scheduler_delivery as sd
from gateway import delivery_ledger as dl
from gateway.config import GatewayConfig, Platform
from gateway.delivery import DeliveryRouter, DeliverySendError, DeliveryTarget
from gateway.platforms.base import SendResult


@pytest.fixture(autouse=True)
def _fresh_ledger(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)
    monkeypatch.setattr(sd, "_maybe_mirror_cron_delivery", lambda *a, **k: None)


@pytest.fixture
def gateway_loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)


def _target(loop, *, token="tok"):
    class Transport:
        adapter = type("Adapter", (), {"_owner_profile": "primary"})()
        is_relay = False

        async def send(self, platform, chat_id, content, metadata=None):
            return SendResult(success=False, error="Not connected", retryable=True)

    fields = {name: None for name in sd._TargetDelivery.__dataclass_fields__}
    fields.update(job={"id": "job-1"}, platform=Platform.TELEGRAM, platform_name="telegram",
                  chat_id="-100", thread_id="42", transport=Transport(), config=GatewayConfig(),
                  loop=loop, target_adapters={}, mirror_text="", origin={},
                  pconfig=type("PConfig", (), {"token": token})())
    return sd._TargetDelivery(**fields)


def _run_lanes(t, loop, monkeypatch, *, standalone_outcome):
    """Live lane (transport rejects "Not connected", retryable) then the standalone lane."""
    standalone_calls = []
    monkeypatch.setattr(
        sd, "_standalone_send",
        lambda t, content, media: standalone_calls.append(content) or standalone_outcome)
    target_errors, delivery_errors = [], []
    assert not sd._deliver_via_live_adapter(
        t, "the report", [], target_errors=target_errors, delivery_errors=delivery_errors,
        unverified_targets=[])
    sd._deliver_standalone(t, "the report", [], target_errors, delivery_errors)
    return standalone_calls, delivery_errors


# --- gate 1: any adapter-flagged retryable rejection queues, by verdict not by string --------

def test_retryable_not_connected_rejection_is_queued_for_redelivery(monkeypatch, gateway_loop):
    t = _target(gateway_loop)
    standalone_calls, errors = _run_lanes(
        t, gateway_loop, monkeypatch, standalone_outcome=(None, "You must pass the token from BotFather", False))
    assert standalone_calls == ["the report"]  # token-holding worker attempts standalone first
    assert t.live_error == "Not connected" and t.live_retryable is True
    assert any("queued text for telegram:-100:42" in e for e in errors)
    # Not reconnect-only: the row waits one backoff tier rather than hammering the adapter.
    assert dl.sweep_failed_for_runtime("telegram", profile="primary") == []
    claimed = dl.sweep_failed_for_runtime("telegram", now=time.time() + 120, profile="primary")
    assert [(row["chat_id"], row["content"]) for row in claimed] == [("-100", "the report")]


def test_non_retryable_same_wording_queues_nothing(monkeypatch, gateway_loop):
    """The SAME "Not connected" string with retryable=False (permanent fatal) must not queue —
    the verdict is the adapter's flag, never the text."""
    t = _target(gateway_loop)

    class FatalTransport:
        adapter = type("Adapter", (), {"_owner_profile": "primary"})()
        is_relay = False

        async def send(self, platform, chat_id, content, metadata=None):
            return SendResult(success=False, error="Not connected", retryable=False)

    t.transport = FatalTransport()
    standalone_calls, errors = _run_lanes(
        t, gateway_loop, monkeypatch, standalone_outcome=(None, "definitive failure", False))
    assert standalone_calls == ["the report"]
    assert not any("queued" in e for e in errors)
    assert dl.sweep_failed_for_runtime("telegram", profile="primary") == []


def test_delivery_send_error_carries_the_retryable_verdict(gateway_loop):
    """The router's raise transports SendResult.retryable; a bare RuntimeError would lose it."""
    class Transport:
        adapter = None
        is_relay = False

        async def send(self, platform, chat_id, content, metadata=None):
            return SendResult(success=False, error="Not connected", retryable=True)

    router = DeliveryRouter(GatewayConfig(), {})
    target = DeliveryTarget(platform=Platform.TELEGRAM, chat_id="-100", is_explicit=True)
    coro = router._deliver_to_platform(target, "hi", {}, transport=Transport())
    with pytest.raises(DeliverySendError) as excinfo:
        asyncio.run_coroutine_threadsafe(coro, gateway_loop).result(timeout=10)
    assert excinfo.value.send_retryable is True
    assert isinstance(excinfo.value, RuntimeError)  # existing `except RuntimeError` stays valid


# --- gate 2: a standalone timeout is not queued (the send may be in flight) -------------------

def test_standalone_timeout_is_not_queued(monkeypatch, gateway_loop):
    t = _target(gateway_loop)
    standalone_calls, errors = _run_lanes(
        t, gateway_loop, monkeypatch,
        standalone_outcome=(None, "standalone send timed out after 45s (the send may still be in flight)", True))
    assert standalone_calls == ["the report"]
    assert not any("queued" in e for e in errors)
    assert dl.sweep_failed_for_runtime("telegram", profile="primary") == []


def test_standalone_send_marks_timeouts_in_flight(monkeypatch):
    """_standalone_send's own timeout arm reports in_flight=True (not a definitive failure)."""
    import tools.send_message_tool as smt

    async def _hang(platform, pconfig, chat_id, message, **kwargs):
        await asyncio.sleep(30)

    monkeypatch.setattr(smt, "_send_to_platform", _hang)
    monkeypatch.setattr(sd, "_get_standalone_send_timeout", lambda: 0.05)
    t = _target(None)
    result, err, in_flight = sd._standalone_send(t, "the report", [])
    assert result is None and "may still be in flight" in err and in_flight is True


def test_standalone_send_definitive_failure_is_not_in_flight(monkeypatch):
    async def _refused(platform, pconfig, chat_id, message, **kwargs):
        return {"error": "You must pass the token from BotFather"}

    import tools.send_message_tool as smt
    monkeypatch.setattr(smt, "_send_to_platform", _refused)
    t = _target(None)
    result, err, in_flight = sd._standalone_send(t, "the report", [])
    # A result-dict error surfaces at the caller; the send itself completed (definitive).
    assert in_flight is False


# --- gate 3: credential-less (satellite) workers skip the doomed standalone attempt -----------

def test_satellite_skips_standalone_and_queues_straight(monkeypatch, gateway_loop):
    t = _target(gateway_loop, token="")
    standalone_calls, errors = _run_lanes(
        t, gateway_loop, monkeypatch, standalone_outcome=(None, "must not be reached", False))
    assert standalone_calls == []  # the guaranteed-to-fail attempt never ran
    assert any("no telegram credential resolved" in e for e in errors)
    assert any("queued text for telegram:-100:42" in e for e in errors)
    claimed = dl.sweep_failed_for_runtime("telegram", now=time.time() + 120, profile="primary")
    assert [(row["chat_id"], row["content"]) for row in claimed] == [("-100", "the report")]


def test_credential_check_reads_the_exact_dispatch_value():
    """Telegram's dispatch is `_send_telegram(pconfig.token, ...)`; the skip keys on that exact
    attribute (whitespace-only counts as missing; a present token attempts standalone)."""
    t = _target(None, token="   ")
    assert sd._standalone_credential_missing(t) is True
    t = _target(None, token="tok")
    assert sd._standalone_credential_missing(t) is False
    t.platform_name = "discord"  # consumption not provable from config: attempt as before
    t.pconfig = type("PConfig", (), {"token": ""})()
    assert sd._standalone_credential_missing(t) is False

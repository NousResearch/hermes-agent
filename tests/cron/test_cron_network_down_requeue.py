"""A cron payload sent while the whole network is down waits for the adapter's reconnect.

A laptop that sleeps on battery runs due jobs in short maintenance wakes with no network. The
platform adapter is already down and reconnecting, so the live lane is skipped (nothing rejects the
payload, ``live_error`` stays None), and the standalone request dies in DNS before it leaves the
host. Both lanes lost to the same outage and nothing was sent, so the payload is ledgered as the
reconnect-only ``network_unreachable`` and the post-reconnect sweep delivers it. A refusal, an
ambiguous attempt, or a never-left failure while the live adapter is up queues nothing, as before.
"""

import sys

import httpx
import pytest

import cron.scheduler_delivery as sd
from gateway import delivery_ledger as dl
from gateway.config import GatewayConfig, Platform
from gateway.platforms._shared import NONE, PARTIAL, DeliveryOutcome, delivery_state, standalone_never_left

NEVER_LEFT = {"error": "Telegram send failed: httpx.ConnectError: [Errno 8] nodename nor servname provided",
              "delivery": NONE, "retryable": True, "never_left": True}
REFUSED = {"error": "Telegram send failed: Bad Request: chat not found", "delivery": NONE, "retryable": True}
LOST = {"error": "Telegram send failed: Timed out", "delivery": "uncertain"}


@pytest.fixture(autouse=True)
def _fresh_ledger(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)
    monkeypatch.setattr(sd, "_maybe_mirror_cron_delivery", lambda *a, **k: None)
    monkeypatch.setattr(sd, "record_cron_delivery", lambda *a, **k: None)


def _target(**overrides):
    fields = {name: None for name in sd._TargetDelivery.__dataclass_fields__}
    fields.update(job={"id": "job-net"}, platform=Platform.TELEGRAM, platform_name="telegram", chat_id="12345",
                  pconfig=type("P", (), {"token": "tok", "extra": {}})(), config=GatewayConfig(),
                  target_adapters={}, mirror_text="", origin={}, live_adapter_ready=False,
                  force_document=False, standalone_in_flight=False, standalone_never_left=False)
    fields.update(overrides)
    return sd._TargetDelivery(**fields)


def _standalone_only(monkeypatch, result, **overrides):
    """The adapter is reconnecting (not live-ready), so only the standalone lane runs."""
    monkeypatch.setattr(sd, "_standalone_send", lambda t, content, media: (dict(result), None))
    errors = []
    sd._deliver_standalone(_target(**overrides), "morning brief", [], [], errors)
    return errors


def test_never_left_while_the_adapter_is_down_waits_for_the_reconnect_sweep(monkeypatch):
    errors = _standalone_only(monkeypatch, NEVER_LEFT)
    assert any("queued text for telegram:12345" in e for e in errors)
    assert dl.pending_retries() == []  # reconnect-only: no timer retries into a dead network
    claimed = dl.sweep_failed_for_runtime("telegram")  # what a primary reconnect runs
    assert [(row["chat_id"], row["content"]) for row in claimed] == [("12345", "morning brief")]


@pytest.mark.parametrize("result", [REFUSED, LOST], ids=["refused", "answer_lost"])
def test_a_refusal_or_an_ambiguous_attempt_is_not_parked(monkeypatch, result):
    errors = _standalone_only(monkeypatch, result)
    assert not any("queued" in e for e in errors)
    assert dl.sweep_failed_for_runtime("telegram") == []


def test_never_left_with_a_live_adapter_up_is_not_parked_for_a_reconnect_that_may_never_come(monkeypatch):
    errors = _standalone_only(monkeypatch, NEVER_LEFT, live_adapter_ready=True)
    assert not any("queued" in e for e in errors)


def test_outcome_stamps_never_left_only_on_a_proven_nothing_sent():
    assert standalone_never_left(DeliveryOutcome().failure({"error": "dns"}, never_left=True))
    refused = DeliveryOutcome().failure({"error": "chat not found"}, never_sent=True)
    assert delivery_state(refused) == NONE and not standalone_never_left(refused)
    partial = DeliveryOutcome()
    partial.landed(7)  # an earlier chunk is on screen: the later connect failure proves nothing
    stamped = partial.failure({"error": "dns"}, never_left=True)
    assert delivery_state(stamped) == PARTIAL and "never_left" not in stamped


def _real_telegram_sender(monkeypatch, bot):
    from tools.send_message_senders import _plugin_standalone_sender
    sender, _ = _plugin_standalone_sender("telegram")  # the module the registry really calls
    mod = sys.modules[sender.__module__]
    monkeypatch.setattr(mod, "_telegram_bot", lambda token, extra=None: bot)
    monkeypatch.setattr(mod, "_telegram_retry_delay", lambda exc, attempt: None)


def _dns_failure():
    from telegram.error import NetworkError
    err = NetworkError("httpx.ConnectError: [Errno 8] nodename nor servname provided, or not known")
    err.__cause__ = httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")
    return err


def test_dead_dns_through_the_real_telegram_sender_is_queued_then_claimed(monkeypatch):
    """End to end on the registered standalone sender: the resolver fails as macOS reports it."""
    pytest.importorskip("telegram")

    class Bot:
        async def send_message(self, **kw):
            raise _dns_failure()

    _real_telegram_sender(monkeypatch, Bot())
    errors = []
    sd._deliver_standalone(_target(), "morning brief", [], [], errors)
    assert any("queued text for telegram:12345" in e for e in errors), errors
    assert dl.pending_retries() == []
    assert [row["content"] for row in dl.sweep_failed_for_runtime("telegram")] == ["morning brief"]


def test_a_refusal_through_the_real_telegram_sender_is_not_parked(monkeypatch):
    """Telegram answered: nothing was sent either, but no reconnect will change its mind."""
    pytest.importorskip("telegram")
    from telegram.error import BadRequest

    class Bot:
        async def send_message(self, **kw):
            raise BadRequest("Chat not found")

    _real_telegram_sender(monkeypatch, Bot())
    errors = []
    sd._deliver_standalone(_target(), "morning brief", [], [], errors)
    assert errors and not any("queued" in e for e in errors)
    assert dl.sweep_failed_for_runtime("telegram") == []

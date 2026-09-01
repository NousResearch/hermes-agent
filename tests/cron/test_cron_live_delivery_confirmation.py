"""Live-adapter delivery confirmation for cron (#77763).

A ``no_agent`` job fired, the scheduler logged
``delivered to telegram:<chat> via live adapter``, and the user received
nothing — no message row, no delivery obligation. The log line was not
evidence of a send:

* the silence-narration filter returns ``{"success": True, "delivered": False}``
  (a successful *drop*), and the normalization block read only ``success``;
* an empty payload skipped the send entirely and still fell into the
  "delivered" branch;
* the log line named the chat but not the lane, so a wrong-thread delivery and
  a phantom one look identical after the fact.

These tests pin the confirmation contract: positive evidence, honest logging,
and fail-closed on nothing-to-send.
"""

import asyncio
import logging
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from cron import scheduler as sched
from cron.scheduler import _confirm_adapter_delivery, _deliver_result
from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# _confirm_adapter_delivery: the contract in isolation
# ---------------------------------------------------------------------------

class _SendResult:
    """Minimal stand-in for an adapter SendResult."""

    def __init__(self, success=True, message_id=None, raw_response=None, **extra):
        self.success = success
        self.message_id = message_id
        self.raw_response = raw_response
        for key, value in extra.items():
            setattr(self, key, value)


class TestConfirmAdapterDelivery:
    def test_none_is_not_delivered(self):
        assert _confirm_adapter_delivery(None, "j1") is False

    def test_missing_success_is_not_delivered(self):
        assert _confirm_adapter_delivery(object(), "j1") is False
        assert _confirm_adapter_delivery({"message_id": 7}, "j1") is False

    def test_explicit_failure_is_not_delivered(self):
        assert _confirm_adapter_delivery(_SendResult(success=False), "j1") is False
        assert _confirm_adapter_delivery({"success": False}, "j1") is False

    def test_filtered_dict_is_not_delivered(self):
        """The exact silence-filter shape: a successful DROP is not a delivery."""
        filtered = {"success": True, "filtered": "silence_narration", "delivered": False}
        assert _confirm_adapter_delivery(filtered, "j1") is False

    def test_delivered_false_on_an_object_is_not_delivered(self):
        result = _SendResult(success=True, message_id=42, delivered=False)
        assert _confirm_adapter_delivery(result, "j1") is False

    def test_positive_evidence_is_delivered_without_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            assert _confirm_adapter_delivery(_SendResult(message_id=1234), "j1") is True
        assert "UNVERIFIED" not in caplog.text

    def test_raw_response_alone_counts_as_evidence(self, caplog):
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            result = _SendResult(raw_response={"ok": True})
            assert _confirm_adapter_delivery(result, "j1") is True
        assert "UNVERIFIED" not in caplog.text

    def test_evidence_free_success_is_accepted_but_warned(self, caplog):
        """Not proof of failure either — accept it, but say so in the log."""
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            assert _confirm_adapter_delivery(_SendResult(), "92e639af907f") is True
        assert "UNVERIFIED" in caplog.text
        assert "92e639af907f" in caplog.text

    def test_evidence_free_success_dict_is_accepted_but_warned(self, caplog):
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            assert _confirm_adapter_delivery({"success": True}, "j1") is True
        assert "UNVERIFIED" in caplog.text


# ---------------------------------------------------------------------------
# _deliver_result: the live lane end to end
# ---------------------------------------------------------------------------

CHAT_ID = "-1001234567890"


def _job(thread_id=None):
    origin = {"platform": "telegram", "chat_id": CHAT_ID}
    if thread_id is not None:
        origin["thread_id"] = thread_id
    return {
        "id": "92e639af907f",
        "name": "Ghost Delivery",
        "deliver": "origin",
        "origin": origin,
    }


def _gateway_config(relay=False):
    config = MagicMock()
    platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    if relay:
        platforms[Platform.RELAY] = PlatformConfig(enabled=True)
    config.platforms = platforms
    config.get_home_channel = lambda p: None
    return config


def _adapters(relay=False):
    adapter = MagicMock()
    if relay:
        adapter.fronts_platform = lambda p: p == Platform.TELEGRAM
        return {Platform.RELAY: adapter}
    return {Platform.TELEGRAM: adapter}


def _run(job, content, send_result, relay=False, standalone_result=None):
    """Drive ``_deliver_result`` over the live lane with a stubbed router.

    Returns ``(error, router_calls, standalone_calls)``.
    """
    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as e:  # noqa: BLE001
            future.set_exception(e)
        return future

    router_calls = []
    standalone_calls = []

    router = MagicMock()

    async def _deliver_to_platform(target, text, metadata):
        router_calls.append({"target": target, "text": text, "metadata": metadata})
        return send_result

    router._deliver_to_platform = _deliver_to_platform

    async def _fake_send_to_platform(platform, pconfig, chat_id, text, **kwargs):
        standalone_calls.append({"chat_id": chat_id, "text": text, "kwargs": kwargs})
        return standalone_result if standalone_result is not None else {}

    with patch("gateway.config.load_gateway_config", return_value=_gateway_config(relay)), \
         patch("cron.scheduler.load_config",
               return_value={"cron": {"wrap_response": False}}), \
         patch("gateway.delivery.DeliveryRouter", return_value=router), \
         patch("tools.send_message_tool._send_to_platform", _fake_send_to_platform), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        error = _deliver_result(job, content, adapters=_adapters(relay), loop=loop)
    return error, router_calls, standalone_calls


class TestFilteredResultIsNotDelivered:
    FILTERED = {"success": True, "filtered": "silence_narration", "delivered": False}

    def test_filtered_dict_does_not_log_a_live_delivery(self, caplog):
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            _, router_calls, standalone_calls = _run(_job(), "...", self.FILTERED)

        assert len(router_calls) == 1                     # the live send was attempted
        assert "via live adapter" not in caplog.text      # but never claimed as delivered
        assert len(standalone_calls) == 1                 # fell back instead of lying

    def test_filtered_dict_fails_closed_on_the_relay_lane(self):
        """Relay owns the destination, so there is no fallback — report it."""
        error, _, standalone_calls = _run(_job(), "...", self.FILTERED, relay=True)

        assert error is not None
        assert "unconfirmed result" in error
        assert "silence_narration" in error  # names the filter, not "unknown"
        assert standalone_calls == []

    def test_confirmed_send_result_still_delivers(self, caplog):
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            error, router_calls, standalone_calls = _run(
                _job(), "Nightly report.", _SendResult(message_id=1234),
            )

        assert error is None
        assert len(router_calls) == 1
        assert standalone_calls == []
        assert "via live adapter" in caplog.text


class TestEmptyPayloadFailsClosed:
    def test_empty_payload_never_reaches_the_adapter(self, caplog):
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            _, router_calls, _ = _run(_job(), "   ", _SendResult(message_id=1))

        assert router_calls == []                     # nothing was sent
        assert "via live adapter" not in caplog.text  # and nothing was claimed
        assert "empty text and no media" in caplog.text

    def test_empty_payload_never_reaches_the_standalone_sender(self, caplog):
        """The native fallback must not re-open the hole the live lane closed.

        Telegram's adapter returns ``SendResult(success=True)`` for empty
        content without an API call, so an unguarded fallback would log a
        standalone "delivered" for the same phantom payload (#77763).
        """
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            error, router_calls, standalone_calls = _run(
                _job(), "   ", _SendResult(message_id=1),
            )

        assert router_calls == []
        assert standalone_calls == []  # _send_to_platform never called
        assert error is not None
        assert "standalone send skipped (empty text and no media)" in error
        assert "delivered to" not in caplog.text

    def test_empty_payload_is_reported_on_the_relay_lane(self):
        error, router_calls, _ = _run(_job(), "", _SendResult(message_id=1), relay=True)

        assert router_calls == []
        assert error is not None
        assert "live adapter send skipped (empty text and no media)" in error


class TestDeliveredLogNamesTheLane:
    def test_log_includes_thread_and_message_id(self, caplog):
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            error, _, _ = _run(
                _job(thread_id="99"), "Nightly report.", _SendResult(message_id=1234),
            )

        assert error is None
        assert "via live adapter thread=99 message_id=1234" in caplog.text

    def test_log_uses_a_dash_when_the_lane_is_unknown(self, caplog):
        """No thread and an evidence-free result must still be attributable."""
        with caplog.at_level(logging.INFO, logger="cron.scheduler"):
            error, _, _ = _run(_job(), "Nightly report.", _SendResult())

        assert error is None
        assert "via live adapter thread=- message_id=-" in caplog.text
        assert "UNVERIFIED" in caplog.text


def test_scheduler_module_exposes_the_confirmation_helper():
    """Guard the import surface the delivery block depends on."""
    assert callable(sched._confirm_adapter_delivery)

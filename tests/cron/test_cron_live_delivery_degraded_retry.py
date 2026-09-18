"""Cron live-lane delivery must retry the adapter's pre-dispatch ``send_path_degraded`` refusal.

A cron job's result was dropped when its delivery landed inside a brief network window. The
Telegram adapter refused the send from a guard that runs **before any API call**:

    if getattr(self, "_send_path_degraded", False):
        return SendResult(success=False, error="send_path_degraded", retryable=True)

so nothing reached the wire and the adapter marks the result ``retryable=True`` — it is saying
"the transport is mid-reconnect, try again". The cron live lane ignored that flag and fell
straight through to the standalone lane, which shares the same broken network path and fails too,
so the result was never delivered even though the adapter recovered seconds later.

These tests pin the contract:

* a retryable pre-dispatch refusal is retried until it succeeds (**must fail on the base
  revision** — base makes exactly one live attempt and moves on);
* the retry is bounded: a transport that never recovers must still fail closed to the standalone
  lane rather than spin;
* a healthy send is never retried (the guard that keeps the added retry from duplicating sends).
"""

import asyncio
import logging
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from cron import scheduler_delivery as sched_delivery
from cron.scheduler import _deliver_result
from gateway.config import Platform, PlatformConfig

CHAT_ID = "-1001234567890"


class _SendResult:
    """Minimal stand-in for an adapter SendResult."""

    def __init__(self, success=True, message_id=None, raw_response=None, error=None,
                 retryable=False):
        self.success = success
        self.message_id = message_id
        self.raw_response = raw_response
        self.error = error
        self.retryable = retryable


def _degraded():
    """Exactly what the adapter returns from its pre-dispatch guard."""
    return _SendResult(success=False, error="send_path_degraded", retryable=True)


def _delivered():
    return _SendResult(success=True, message_id=1234)


def _job():
    return {
        "id": "92e639af907f",
        "name": "Degraded Transport",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": CHAT_ID},
    }


def _gateway_config():
    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    return config


def _run(send_results):
    """Drive ``_deliver_result`` over the live lane with a stubbed router.

    ``send_results`` is the sequence of SendResults the live lane returns, one per attempt
    (the last one repeats). Returns ``(error, router_attempts, standalone_calls)``.
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

    attempts = []
    call_count = 0

    def next_result():
        nonlocal call_count
        idx = min(call_count, len(send_results) - 1)
        call_count += 1
        return send_results[idx]

    router = MagicMock()

    async def _deliver_to_platform(target, text, metadata):
        attempts.append(target)
        return next_result()

    router._deliver_to_platform = _deliver_to_platform

    standalone_calls = []

    async def _fake_send_to_platform(platform, pconfig, chat_id, text, **kwargs):
        standalone_calls.append({"chat_id": chat_id, "text": text})
        return {}

    with patch("gateway.config.load_gateway_config", return_value=_gateway_config()), \
         patch("cron.scheduler.load_config",
               return_value={"cron": {"wrap_response": False}}), \
         patch("cron.scheduler_delivery._record_delivery_verification"), \
         patch("gateway.delivery.DeliveryRouter", return_value=router), \
         patch("tools.send_message_tool._send_to_platform", _fake_send_to_platform), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro), \
         patch("time.sleep", return_value=None):
        error = _deliver_result(_job(), "Nightly report.", adapters={Platform.TELEGRAM: MagicMock()},
                                loop=loop)
    return error, attempts, standalone_calls


class TestDegradedRefusalIsRecognised:
    def test_matches_the_adapter_refusal(self):
        assert sched_delivery._is_degraded_refusal(_degraded()) is True
        assert sched_delivery._is_degraded_refusal(
            {"success": False, "error": "send_path_degraded"}) is True

    def test_does_not_match_other_failures(self):
        assert sched_delivery._is_degraded_refusal(
            _SendResult(success=False, error="Not connected")) is False
        assert sched_delivery._is_degraded_refusal(
            {"success": True, "filtered": "silence_narration", "delivered": False}) is False
        assert sched_delivery._is_degraded_refusal(None) is False


class TestRetryableRefusalIsRetried:
    def test_delivers_on_the_live_lane_once_the_transport_recovers(self, caplog):
        """The behavioural fix: refuse once (transport reconnecting), then succeed."""
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            error, attempts, standalone_calls = _run([_degraded(), _delivered()])

        assert error is None
        assert len(attempts) == 2, "the degraded refusal must be re-attempted on the live lane"
        assert standalone_calls == [], "recovery on the live lane must not reach the standalone lane"
        assert "retrying" in caplog.text

    def test_refusal_does_not_hang_when_the_transport_never_recovers(self, monkeypatch, caplog):
        """Bounded: an adapter that stays down fails closed to standalone instead of spinning."""
        monkeypatch.setattr(sched_delivery, "_LIVE_DEGRADED_RETRY_BUDGET_SECS", 0.0, raising=False)
        with caplog.at_level(logging.WARNING, logger="cron.scheduler"):
            _, attempts, standalone_calls = _run([_degraded()])

        assert len(attempts) == 1, "an exhausted budget must not keep re-attempting"
        assert len(standalone_calls) == 1, "an unresolved refusal must still fall back"
        assert "still refused (send_path_degraded)" in caplog.text

    def test_healthy_send_is_not_retried(self):
        """Guard: the retry must key on the refusal only, or a good send could be duplicated."""
        error, attempts, standalone_calls = _run([_delivered()])

        assert error is None
        assert len(attempts) == 1
        assert standalone_calls == []

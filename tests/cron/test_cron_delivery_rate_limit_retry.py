"""Throttled-send retry for the live-adapter lane.

The WeChat/iLink adapter counts sends in a sliding window; once enough land inside it, it opens an
adapter-wide breaker and rejects **every** send for the cooldown window
(``plugins/platforms/weixin/adapter.py`` raises
``RuntimeError("iLink sendmessage rate limited; cooldown active for 30.0s")``).

``_live_send_text`` re-raised that error, so the scheduler fell through to the standalone lane —
which hits the same breaker — and the run was reported as a delivery failure. A ``no_agent`` cron
job has no next run: the push is simply lost. The 2026-09-20 11:50 建行时光好礼 reminder died this
way while a job 7 minutes later — after the cooldown — delivered fine.

These tests pin the contract:

* a throttle-shaped failure is waited out and retried;
* a **genuine** failure (bad token, unknown chat) keeps failing fast — retrying it would burn the
  delivery window on a hopeless send;
* the retry budget is bounded, and the eventual failure reports one error, not one per attempt.
"""

import sys
import types

import pytest

from cron.scheduler_delivery import (
    _RATE_LIMIT_SEND_MAX_RETRIES,
    _RATE_LIMIT_SEND_MAX_WAIT_SECONDS,
    _live_send_text,
    _rate_limit_retry_wait,
)


# ---------------------------------------------------------------------------
# _rate_limit_retry_wait: the classifier, in isolation
# ---------------------------------------------------------------------------

class TestRateLimitRetryWait:
    def test_cooldown_message_is_waited_out_with_pad(self):
        exc = RuntimeError("iLink sendmessage rate limited; cooldown active for 30.0s")
        assert _rate_limit_retry_wait(exc) == pytest.approx(32.0)

    def test_short_cooldown_is_padded_by_the_pad_constant(self):
        assert _rate_limit_retry_wait(RuntimeError("cooldown active for 0.5s")) == pytest.approx(2.5)

    def test_long_cooldown_is_capped(self):
        # A cooldown longer than the cap must not park the scheduler for ten minutes: the cap keeps
        # the retry inside one delivery window, after which the run reports the real failure.
        assert _rate_limit_retry_wait(RuntimeError("cooldown active for 600.0s")) == \
            _RATE_LIMIT_SEND_MAX_WAIT_SECONDS

    def test_generic_rate_limited_falls_back_to_a_fixed_wait(self):
        assert _rate_limit_retry_wait(RuntimeError("429 rate limited")) == pytest.approx(5.0)
        assert _rate_limit_retry_wait(RuntimeError("RATE LIMITED")) == pytest.approx(5.0)

    @pytest.mark.parametrize("message", [
        "401 unauthorized: bad token",
        "chat not found",
        "cooldown remaining",        # no "for Ns" group -> not throttle-shaped
        "cooldown active for ?s",    # non-numeric group -> not throttle-shaped
        "",
    ])
    def test_a_genuine_error_is_not_retried(self, message):
        assert _rate_limit_retry_wait(RuntimeError(message)) is None

    def test_any_strable_exception_is_handled(self):
        # This runs inside the send error path, where str(exc) must never raise.
        assert _rate_limit_retry_wait(ValueError("cooldown active for 3.0s")) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _live_send_text: the retry at the send boundary
# ---------------------------------------------------------------------------

_THROTTLE = RuntimeError("iLink sendmessage rate limited; cooldown active for 30.0s")
_SUCCESS = ({"success": True, "message_id": "m1"}, None)
_THROTTLED = (None, _THROTTLE)


def _fake_target():
    """A minimal _TargetDelivery carrying only the fields the send path reads."""
    from cron.scheduler_delivery import _TargetDelivery

    return _TargetDelivery(
        job={"id": "job-1"}, platform=None, platform_name="weixin", chat_id="chat-1",
        thread_id=None, transport=None, pconfig=None, runtime_adapter=None,
        target_adapters={}, config={}, loop=None, notify_delivery=False, origin={},
        origin_target=False, origin_user_id=None, is_dm_target=True, mirror_text="",
        mirror_this_target=False, in_channel_surface=False, inchannel_continuable=False,
        opened_thread_id=None, live_adapter_ready=True,
    )


class _Future:
    """Stand-in for a run_coroutine_threadsafe future."""

    def __init__(self, value=None, exc=None):
        self._value, self._exc = value, exc

    def result(self, timeout=None):
        if self._exc is not None:
            raise self._exc
        return self._value

    def cancel(self):
        return False


class _SendPlan:
    """Hands out one future per send attempt, repeating the last outcome.

    Repeating the final entry means an unbounded retry loop still terminates the test instead of
    running off the end of the list.
    """

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.attempts = 0

    def __call__(self, coro, loop):
        coro.close()            # never awaited: the fake future replaces the real send
        self.attempts += 1
        value, exc = self._outcomes[min(self.attempts - 1, len(self._outcomes) - 1)]
        return _Future(value=value, exc=exc)


def _drive(monkeypatch, plan, **kwargs):
    """Run ``_live_send_text`` against ``plan``; return (outcome, state).

    ``outcome`` is the returned tuple, or the raised exception. ``state`` carries the error lists
    and the recorded sleeps, which stay readable even when the send raises.
    """

    class _FakeRouter:
        def __init__(self, config, target_adapters):
            pass

        async def _deliver_to_platform(self, *args, **kwargs):
            raise AssertionError("the fake future should short-circuit the coroutine")

    fake_delivery = types.ModuleType("gateway.delivery")
    fake_delivery.DeliveryRouter = _FakeRouter
    fake_delivery.DeliveryTarget = lambda **kw: kw
    fake_delivery.PartialDeliveryError = type("PartialDeliveryError", (Exception,), {})

    fake_async_utils = types.ModuleType("agent.async_utils")
    fake_async_utils.safe_schedule_threadsafe = plan

    monkeypatch.setitem(sys.modules, "gateway.delivery", fake_delivery)
    monkeypatch.setitem(sys.modules, "agent.async_utils", fake_async_utils)

    sleeps = []
    monkeypatch.setattr("cron.scheduler_delivery.time.sleep", sleeps.append, raising=False)

    target = _fake_target()
    state = {
        "target": target,
        "target_errors": [],
        "delivery_errors": [],
        "unverified_targets": [],
        "sleeps": sleeps,
        "plan": plan,
    }

    try:
        outcome = _live_send_text(
            target, "hello", None, {},
            target_errors=state["target_errors"], delivery_errors=state["delivery_errors"],
            unverified_targets=state["unverified_targets"], **kwargs)
    except Exception as exc:
        return exc, state
    return outcome, state


class TestLiveSendRetry:
    def test_throttled_once_then_delivered(self, monkeypatch):
        """The breaker opens mid-burst; waiting it out delivers a push that was otherwise lost."""
        outcome, state = _drive(monkeypatch, _SendPlan([_THROTTLED, _SUCCESS]))

        assert outcome == (True, False, "m1")
        assert state["plan"].attempts == 2
        assert state["sleeps"] == [pytest.approx(32.0)]
        # The throttled attempt delivered nothing, so it must not be recorded as a delivery error.
        assert state["target_errors"] == []
        assert state["target"].live_error is None

    def test_persistent_throttling_gives_up_after_the_budget(self, monkeypatch):
        """A cooldown that outlasts the budget must give up, not spin forever."""
        plan = _SendPlan([_THROTTLED])
        outcome, state = _drive(monkeypatch, plan)

        assert isinstance(outcome, RuntimeError)
        assert state["plan"].attempts == _RATE_LIMIT_SEND_MAX_RETRIES + 1
        assert len(state["sleeps"]) == _RATE_LIMIT_SEND_MAX_RETRIES

    def test_bounded_failure_records_exactly_one_error(self, monkeypatch):
        """Retries must not multiply the reported failure: one error, however many attempts."""
        outcome, state = _drive(monkeypatch, _SendPlan([_THROTTLED]))

        assert isinstance(outcome, RuntimeError)
        assert len(state["target_errors"]) == 1
        assert "live adapter send failed" in state["target_errors"][0]
        assert state["target"].live_error is not None

    def test_genuine_error_fails_fast_without_sleeping(self, monkeypatch):
        """A bad token is not a throttle: one attempt, no wait, straight to the standalone lane."""
        plan = _SendPlan([(None, RuntimeError("401 unauthorized: bad token"))])
        outcome, state = _drive(monkeypatch, plan)

        assert isinstance(outcome, RuntimeError)
        assert state["plan"].attempts == 1
        assert state["sleeps"] == []

    def test_retry_budget_can_be_exhausted_by_the_caller(self, monkeypatch):
        """The retry counter is threaded through, so a caller can opt out with 0."""
        plan = _SendPlan([_THROTTLED])
        outcome, state = _drive(monkeypatch, plan, _rate_limit_retries=0)

        assert isinstance(outcome, RuntimeError)
        assert state["plan"].attempts == 1
        assert state["sleeps"] == []

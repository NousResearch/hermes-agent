"""A short pause the provider named is waited out, not failed over.

Regression for: a 429 that carries ``Retry-After: 7`` (header or body) left the
primary provider on attempt 1 of 3, because the eager rate-limit fallback runs
before ``compute_error_backoff`` ever gets to honour the pause.
"""

from __future__ import annotations

import pytest

from agent.error_classifier import FailoverReason
from agent.retry_utils import RETRY_AFTER_CAP_SECONDS, named_retry_after_seconds
from agent.turn_recovery import (
    DEFAULT_SHORT_NAMED_PAUSE_CEILING_SECONDS,
    SHORT_NAMED_PAUSE_MAX_DEFERRALS,
    compute_error_backoff,
    short_named_pause,
    short_named_pause_ceiling,
)


class _Response:
    def __init__(self, headers=None):
        self.headers = dict(headers or {})


class _ApiError(Exception):
    def __init__(self, *, header=None, body=None, text="Error code: 429"):
        super().__init__(text)
        if body is not None:
            self.body = body
        self.response = _Response({"Retry-After": str(header)} if header is not None else {})


class _Classified:
    def __init__(self, reason=FailoverReason.rate_limit):
        self.reason = reason
        self.error_context = {}
        self.is_auth = False
        self.billing_unverified = False
        self.retryable = True


def _rate_limited(**kwargs):
    return short_named_pause(
        _Agent(), _Classified(), _ApiError(**kwargs), retry_count=1, max_retries=3,
    )


# --- the named pause, and where it may come from --------------------------


def test_short_pause_in_the_header_defers_the_fallback():
    assert _rate_limited(header=7) == 7.0


def test_short_pause_in_the_body_defers_the_fallback():
    assert _rate_limited(body={"detail": {"retry_after": 37}}) is None, (
        "only the shapes compute_error_backoff reads count as named"
    )
    assert _rate_limited(body={"retry_after": 37}) == 37.0
    assert _rate_limited(body={"error": {"retry_after": 37}}) == 37.0


def test_the_header_wins_over_the_body_just_as_the_wait_does():
    error = _ApiError(header=9, body={"retry_after": 31})
    assert short_named_pause(_Agent(), _Classified(), error, retry_count=1, max_retries=3) == 9.0
    assert named_retry_after_seconds(error) == 9.0


def test_the_decision_and_the_wait_read_the_same_number(monkeypatch):
    """The whole point: never cancel a fallback for a pause we will not sleep.

    Deciding from a source ``compute_error_backoff`` does not read would replace a
    failover with a 2-4s generic backoff into the same wall — strictly worse than
    the behaviour being fixed.
    """
    error = _ApiError(body={"retry_after": 31})
    agent = _Agent()
    wait = compute_error_backoff(
        agent, error, retry_count=1, max_retries=3, is_rate_limited=True,
        is_zai_coding_overload=False, base_url="https://relay.example/v1", model="m",
    )
    assert wait == short_named_pause(_Agent(), _Classified(), error, retry_count=1, max_retries=3) == 31.0


def test_an_unnamed_pause_changes_nothing():
    assert _rate_limited() is None
    assert _rate_limited(body={"message": "slow down"}) is None


def test_a_zero_or_expired_pause_is_not_an_instruction():
    assert _rate_limited(header=0) is None
    assert _rate_limited(header="Wed, 21 Oct 2015 07:28:00 GMT") is None


def test_the_cap_applies_before_the_ceiling_comparison():
    assert named_retry_after_seconds(_ApiError(header=99999)) == RETRY_AFTER_CAP_SECONDS
    assert _rate_limited(header=99999) is None


# --- the four ways it must decline ----------------------------------------


def test_a_long_pause_is_a_wall_and_still_fails_over():
    assert _rate_limited(header=DEFAULT_SHORT_NAMED_PAUSE_CEILING_SECONDS + 1) is None
    assert _rate_limited(header=DEFAULT_SHORT_NAMED_PAUSE_CEILING_SECONDS) is not None


@pytest.mark.parametrize(
    "reason",
    [
        FailoverReason.billing,
        FailoverReason.upstream_rate_limit,
        FailoverReason.overloaded,
        FailoverReason.timeout,
    ],
)
def test_only_a_plain_rate_limit_is_deferred(reason):
    assert short_named_pause(_Agent(), _Classified(reason), _ApiError(header=7), retry_count=1, max_retries=3) is None


def test_the_third_refusal_in_a_turn_fails_over():
    error = _ApiError(header=7)
    for attempt in range(1, SHORT_NAMED_PAUSE_MAX_DEFERRALS + 1):
        assert short_named_pause(_Agent(), _Classified(), error, retry_count=attempt, max_retries=9) == 7.0
    assert short_named_pause(
        _Agent(), _Classified(), error,
        retry_count=SHORT_NAMED_PAUSE_MAX_DEFERRALS + 1, max_retries=9,
    ) is None


def test_the_ceiling_is_configurable(monkeypatch):
    monkeypatch.setenv("HERMES_SHORT_NAMED_PAUSE_MAX_SECONDS", "10")
    assert short_named_pause_ceiling() == 10.0
    assert _rate_limited(header=7) == 7.0
    assert _rate_limited(header=30) is None


# --- the site that has to act on it ---------------------------------------


class _Agent:
    """Only what the eager-fallback site touches."""

    def __init__(self, chain=1):
        self._fallback_chain = [{"provider": "anthropic", "model": "m"}] * chain
        self._fallback_index = 0
        self._credential_pool = None
        self.provider = "relay"
        self.model = "m"
        self.base_url = "https://relay.example/v1"
        self.statuses = []
        self.activated = []

    def _buffer_diagnostic_status(self, text):
        self.statuses.append(text)

    def _emit_diagnostic_status(self, text):
        self.statuses.append(text)

    def _emit_diagnostic_wait(self, text):
        self.statuses.append(text)

    def _client_log_context(self):
        return ""

    def _try_activate_fallback(self, reason=None):
        self.activated.append(reason)
        self._fallback_index += 1
        return True


def _route(agent, error, retry_count):
    from agent import turn_recovery
    from agent.turn_retry_state import TurnRetryState

    return turn_recovery.route_classified_error(
        agent,
        error,
        _Classified(),
        TurnRetryState(),
        error_msg=str(error).lower(),
        error_context={},
        messages=[],
        api_messages=[],
        conversation_history=[],
        system_message=None,
        active_system_prompt=None,
        api_call_count=1,
        effective_task_id="t",
        retry_count=retry_count,
        max_retries=3,
        compression_attempts=0,
        max_compression_attempts=2,
        recovered_with_pool=False,
        base_url=agent.base_url,
        model=agent.model,
    )


def test_the_first_named_short_pause_does_not_switch_providers(monkeypatch):
    agent = _Agent()
    monkeypatch.setattr(
        "agent.conversation_loop._ra",
        lambda: type("R", (), {"_pool_may_recover_from_rate_limit": staticmethod(lambda _p: False)})(),
    )
    _route(agent, _ApiError(header=7), retry_count=1)
    assert agent.activated == [], "the provider asked for 7s; we left anyway"
    assert any("7s" in s for s in agent.statuses)


def test_the_pause_budget_runs_out_and_the_chain_is_used(monkeypatch):
    agent = _Agent()
    monkeypatch.setattr(
        "agent.conversation_loop._ra",
        lambda: type("R", (), {"_pool_may_recover_from_rate_limit": staticmethod(lambda _p: False)})(),
    )
    _route(agent, _ApiError(header=7), retry_count=SHORT_NAMED_PAUSE_MAX_DEFERRALS + 1)
    assert agent.activated == [FailoverReason.rate_limit]


def test_a_provider_with_its_own_divert_is_never_deferred():
    """``nous`` re-enters the loop from its own guard, below this decision and above
    the backoff, so cancelling the fallback would promise a wait that never happens —
    and would let a cross-session breaker file be written where it previously was not."""
    agent = _Agent()
    agent.provider = "nous"
    assert short_named_pause(
        agent, _Classified(), _ApiError(header=7), retry_count=1, max_retries=3
    ) is None
    agent.provider = "relay"
    assert short_named_pause(
        agent, _Classified(), _ApiError(header=7), retry_count=1, max_retries=3
    ) == 7.0


def test_the_last_attempt_is_never_deferred():
    """``retry_count >= max_retries`` routes to the exhausted-retries branch, which
    activates the chain without a reason — so the rate-limit cooldown is never armed and
    the next turn returns to the limited provider. With ``api_max_retries: 1`` the very
    first 429 would hit this."""
    error = _ApiError(header=7)
    assert short_named_pause(
        _Agent(), _Classified(), error, retry_count=1, max_retries=1) is None
    assert short_named_pause(
        _Agent(), _Classified(), error, retry_count=2, max_retries=2) is None
    assert short_named_pause(
        _Agent(), _Classified(), error, retry_count=1, max_retries=2) == 7.0


@pytest.mark.parametrize("junk", ["", "soon", "-1", "0", "nan", "inf", "-inf"])
def test_a_non_finite_or_useless_ceiling_falls_back_to_the_default(monkeypatch, junk):
    monkeypatch.setenv("HERMES_SHORT_NAMED_PAUSE_MAX_SECONDS", junk)
    assert short_named_pause_ceiling() == DEFAULT_SHORT_NAMED_PAUSE_CEILING_SECONDS
    assert _rate_limited(header=600) is None
    assert _rate_limited(header=7) == 7.0


def test_a_local_validation_shaped_error_is_never_deferred():
    """``is_client_error`` also sits between this decision and the backoff.

    Its first disjunct is "this is a ValueError/TypeError", and the classifier decides
    ``rate_limit`` from the error *text*, so a third-party shim raising a 429 as a
    ValueError would be deferred into an abort that never waits.
    """
    payload = {"retry_after": 7}
    text = 'Error code: 429 - {"retry_after": 7}'

    class _ShimValueError(ValueError):
        body = payload

    assert short_named_pause(
        _Agent(), _Classified(), _ShimValueError(text), retry_count=1, max_retries=3
    ) is None

    class _ShapeMismatch(TypeError):
        body = payload

    # ...but the documented exceptions stay ordinary provider failures.
    assert short_named_pause(
        _Agent(), _Classified(), _ShapeMismatch("NoneType object is not iterable"),
        retry_count=1, max_retries=3,
    ) == 7.0
    assert short_named_pause(
        _Agent(), _Classified(), _ApiError(header=7), retry_count=1, max_retries=3
    ) == 7.0

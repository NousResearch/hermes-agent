"""#38929 / #68771: 503/529 "upstream capacity limits" overloads retry over a
bounded backoff window on the primary BEFORE the fallback chain activates.

The Z.AI overload path already gets a long adaptive backoff; this generalises
that policy to provider-agnostic 503/529 overloads so a capacity outage gets
a few retries over ~75s instead of one quick retry then fallback (or give-up
when no fallback chain is configured).
"""

from __future__ import annotations

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.retry_utils import (
    capacity_overload_backoff,
    capacity_overload_retry_ceiling,
    is_capacity_overload_error,
)


class _Err(Exception):
    def __init__(self, status_code, message="Service Unavailable"):
        super().__init__(message)
        self.status_code = status_code
        self.response = None


class _ErrWithStatusAttr(Exception):
    """Some SDKs put the code on .status instead of .status_code."""

    def __init__(self, status, message="Service Unavailable"):
        super().__init__(message)
        self.status = status
        self.response = None


class _ErrWithCause(Exception):
    """httpx/SDK wrappers sometimes nest the real error on __cause__."""

    def __init__(self, cause):
        super().__init__(str(cause))
        self.__cause__ = cause
        self.response = None


def test_503_and_529_are_capacity_overloads():
    assert is_capacity_overload_error(_Err(503)) is True
    assert is_capacity_overload_error(_Err(529)) is True


def test_other_5xx_are_not_capacity_overloads():
    assert is_capacity_overload_error(_Err(500)) is False
    assert is_capacity_overload_error(_Err(502)) is False


def test_4xx_and_2xx_are_not_capacity_overloads():
    assert is_capacity_overload_error(_Err(429)) is False
    assert is_capacity_overload_error(_Err(400)) is False
    assert is_capacity_overload_error(_Err(200)) is False
    assert is_capacity_overload_error(ValueError("no status")) is False


def test_status_attr_also_works():
    """Some SDKs use .status instead of .status_code; we should catch both."""
    assert is_capacity_overload_error(_ErrWithStatusAttr(503)) is True
    assert is_capacity_overload_error(_ErrWithStatusAttr(529)) is True
    assert is_capacity_overload_error(_ErrWithStatusAttr(500)) is False


def test_cause_chain_is_walked():
    """httpx wraps the real error on __cause__; we should walk the chain."""
    inner = _Err(503)
    outer = _ErrWithCause(inner)
    assert is_capacity_overload_error(outer) is True

    # Even on context (not cause) chain.
    try:
        try:
            raise _Err(503)
        except _Err as inner_exc:
            raise ValueError("wrapper") from None  # context, not cause
    except ValueError as exc:
        assert is_capacity_overload_error(exc) is True


def test_cycle_in_cause_chain_does_not_loop():
    """Defensive: a cycle in the cause chain must not hang."""
    a = _Err(500)
    b = _Err(500)
    a.__cause__ = b
    b.__cause__ = a
    assert is_capacity_overload_error(a) is False  # both 500, not 503/529


def test_httpx_response_status_code_works():
    """httpx.HTTPStatusError puts the code on error.response.status_code."""
    response = type("R", (), {"status_code": 503})()
    err = type("E", (Exception,), {"response": response})()
    assert is_capacity_overload_error(err) is True


def test_requests_http_error_response_status_code_works():
    """requests.exceptions.HTTPError also uses response.status_code."""
    response = type("R", (), {"status_code": 529})()
    err = type("E", (Exception,), {"response": response})()
    assert is_capacity_overload_error(err) is True


def test_urllib_style_code_attribute_works():
    """urllib.error.HTTPError and gRPC wrappers expose status as .code."""
    err = type("E", (Exception,), {"code": 503})()
    assert is_capacity_overload_error(err) is True


def test_both_cause_and_context_are_walked():
    """When an exception has both __cause__ and __context__, we walk both."""
    target = _Err(503)

    class WithBoth(Exception):
        pass

    err = WithBoth("outer")
    err.__cause__ = ValueError("unrelated")  # 0, not a 503
    err.__context__ = target  # 503 — must be reached
    assert is_capacity_overload_error(err) is True


def test_self_referential_cause_does_not_loop():
    """An exception whose __cause__ is itself must terminate."""
    err = _Err(500)
    err.__cause__ = err
    assert is_capacity_overload_error(err) is False


def test_503_classified_as_retryable_overloaded():
    result = classify_api_error(_Err(503), provider="nous")
    assert result.reason == FailoverReason.overloaded
    assert result.retryable is True


def test_capacity_backoff_first_retry_keeps_short_wait():
    wait, policy = capacity_overload_backoff(1, 2.0)
    assert wait == 2.0
    assert policy == "capacity_overload_short"


def test_capacity_backoff_walks_long_schedule():
    # Long tier: 5/10/20/40 with light jitter (delay in [base, base*1.2])
    waits = [capacity_overload_backoff(a, 2.0)[0] for a in range(2, 8)]
    expected_bases = [5.0, 10.0, 20.0, 40.0, 40.0, 40.0]
    for got, base in zip(waits, expected_bases):
        assert base <= got <= base * 1.2, f"{got} not in [{base}, {base * 1.2}]"


def test_capacity_backoff_caps_at_last_long_entry():
    # After the 4-entry table, further attempts stay at 40s
    waits = [capacity_overload_backoff(a, 2.0)[0] for a in range(6, 10)]
    for w in waits:
        assert 40.0 <= w <= 40.0 * 1.2


def test_capacity_backoff_policy_labels():
    assert (
        capacity_overload_backoff(1, 2.0)[1] == "capacity_overload_short"
    )
    for a in range(2, 10):
        assert capacity_overload_backoff(a, 2.0)[1] == "capacity_overload_long"


def test_capacity_ceiling_reaches_all_long_tiers():
    # 1 short + 4 long + 1 (the pre-backoff ceiling check) = 6
    assert capacity_overload_retry_ceiling() == 6


def test_capacity_ceiling_exceeds_default_api_max_retries():
    """The ceiling must sit past the default api_max_retries (3) so the
    long-backoff tiers actually run."""
    ceiling = capacity_overload_retry_ceiling()
    assert ceiling > 3

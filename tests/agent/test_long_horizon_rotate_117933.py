"""Long-horizon 429 + multi-entry pool → rotate on the first 429 instead of
sleeping through the wait (#117933). A bare/short 429 keeps the existing
retry-once-then-rotate for transient blips. Single-account pools keep retry."""
import time as _time
from unittest.mock import MagicMock

import agent.agent_runtime_helpers as arh


class _Entry:
    def __init__(self, eid, status="ok"):
        self.id = eid
        self.runtime_api_key = "k" + eid
        self.last_status = status


class _Pool:
    def __init__(self, entries):
        self._e = entries

    def entries(self):
        return self._e

    def current(self):
        return self._e[0]


def _run(pool, error_context, has_retried_429=False):
    calls = []

    def rotate_and_swap(status, label):
        calls.append((status, label))
        return True

    recovered, retried = arh._recover_rate_limit(
        pool, has_retried_429=has_retried_429, error_context=error_context,
        api_key_hint="ka", credential_id="a", rotate_and_swap=rotate_and_swap,
    )
    return recovered, retried, calls


def test_long_horizon_multi_pool_rotates_on_first_429():
    """Reset horizon 600 s + 2-entry pool → rotate immediately (#117933)."""
    pool = _Pool([_Entry("a"), _Entry("b")])
    ctx = {"reason": "rate_limit_error",
           "message": "This request would exceed your organization's rate limit",
           "reset_at": _time.time() + 600, "retry_after": 600}
    recovered, retried, calls = _run(pool, ctx)
    assert recovered is True
    assert calls == [(429, "rate limit")]


def test_short_429_multi_pool_keeps_retry_once():
    """A bare 429 (no horizon) keeps the retry-once-then-rotate path."""
    pool = _Pool([_Entry("a"), _Entry("b")])
    ctx = {"reason": "rate_limit_error",
           "message": "This request would exceed your organization's rate limit"}
    recovered, retried, calls = _run(pool, ctx, has_retried_429=False)
    assert recovered is False
    assert retried is True
    assert calls == []


def test_single_account_pool_keeps_retry():
    """A single-entry pool keeps the retry path (no alternative to rotate to)."""
    pool = _Pool([_Entry("a")])
    ctx = {"reason": "rate_limit_error",
           "message": "This request would exceed your organization's rate limit",
           "reset_at": _time.time() + 600, "retry_after": 600}
    recovered, retried, calls = _run(pool, ctx, has_retried_429=False)
    assert recovered is False
    assert retried is True
    assert calls == []


def test_exhausted_status_still_rotates_immediately():
    """The existing already-exhausted path is unchanged."""
    pool = _Pool([_Entry("a", status=arh.STATUS_EXHAUSTED), _Entry("b")])
    ctx = {"reason": "rate_limit_error",
           "message": "This request would exceed your organization's rate limit",
           "reset_at": _time.time() + 600, "retry_after": 600}
    recovered, retried, calls = _run(pool, ctx)
    assert recovered is True
    assert calls == [(429, "rate limit, pre-exhausted")]

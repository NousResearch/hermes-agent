"""A rate-limit Retry-After must not be honored on a pool-exhausted seat.

A 429 ``Retry-After`` tells us when the SERVER's throttle window reopens. It
says nothing about whether the credential seat the request rode still has
quota. When the credential pool has already marked that seat exhausted and has
nothing to rotate to, those are two different clocks — and honoring the header
sleeps the full 600s cap only to wake up on the same dead seat.

Observed shape (a weekly-capped subscription seat behind a relay):

    credential pool: marking <KEY> exhausted (status=429), rotating
    credential pool: no available entries (all exhausted or empty)
    Honoring server Retry-After=600.0s (reason=rate_limit, attempt=2/3)
    Retrying API call in 600.0s (attempt 1/3) ... provider=<relay>

The seat had ~31h until its real reset; the header said 600s. Under a
budget-limited caller (a scheduled run with a wall-clock cap) the process is
killed mid-sleep, so the configured fallback chain is never reached: repeated
runs, zero output.

The fix is scoped to ``rate_limit`` + pool-reports-no-available-seat. The 600s
cap and the retry ceiling are untouched — both are guards, not the cause, and
are pinned below as negative controls.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.agent_runtime_helpers import pool_seat_exhaustion_state
from agent.turn_recovery import compute_error_backoff


class _Headers(dict):
    def get(self, key, default=None):  # case-insensitive enough for the seam
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default


class _Err(RuntimeError):
    def __init__(self, retry_after="600"):
        super().__init__("HTTP 429: rate limit")
        self.response = SimpleNamespace(headers=_Headers({"Retry-After": retry_after}))


class _Pool:
    def __init__(self, *, available, next_at=None, provider="relay-1", raises=False):
        self.provider = provider
        self._available = available
        self._next_at = next_at
        self._raises = raises

    def has_available(self):
        if self._raises:
            raise RuntimeError("pool exploded")
        return self._available

    def next_available_at(self):
        return self._next_at


def _agent(pool, provider="relay-1"):
    agent = MagicMock()
    agent._client_log_context.return_value = ""
    agent._credential_pool = pool
    agent.provider = provider
    agent.base_url = "http://relay.test/anthropic"
    return agent


def _backoff(agent, *, retry_after="600", is_rate_limited=True):
    return compute_error_backoff(
        agent,
        _Err(retry_after),
        retry_count=1,
        max_retries=3,
        is_rate_limited=is_rate_limited,
        is_zai_coding_overload=False,
        base_url="http://relay.test/anthropic",
        model="test/model",
    )


class TestExhaustedSeatDeclinesRetryAfter:
    @pytest.mark.real_retry_backoff
    def test_exhausted_seat_does_not_sleep_the_full_window(self):
        """The measured incident. Honoring would return exactly 600.0."""
        wait = _backoff(_agent(_Pool(available=False)))
        assert wait != 600.0
        assert wait < 600.0

    @pytest.mark.real_retry_backoff
    def test_healthy_seat_still_honors_the_same_header(self):
        """The discriminator is the SEAT, not the header."""
        assert _backoff(_agent(_Pool(available=True))) == 600.0

    @pytest.mark.real_retry_backoff
    def test_no_pool_is_unchanged_legacy_behavior(self):
        assert _backoff(_agent(None)) == 600.0

    @pytest.mark.real_retry_backoff
    def test_recovery_inside_the_window_is_still_honored(self):
        """Carve-out: the seat genuinely returns inside the wait we were about
        to serve anyway, so honoring still beats leaving the primary."""
        pool = _Pool(available=False, next_at=time.time() + 120.0)
        assert _backoff(_agent(pool)) == 600.0

    @pytest.mark.real_retry_backoff
    def test_recovery_beyond_the_window_is_declined(self):
        """~31h until a weekly cap resets, header says 600s."""
        pool = _Pool(available=False, next_at=time.time() + 31 * 3600.0)
        assert _backoff(_agent(pool)) != 600.0

    @pytest.mark.real_retry_backoff
    def test_recovery_compared_against_the_CAPPED_wait(self):
        """A header of 9999 caps to 600, so a 900s recovery is outside the real
        window even though it is inside the raw header value."""
        pool = _Pool(available=False, next_at=time.time() + 900.0)
        assert _backoff(_agent(pool), retry_after="9999") != 600.0

    @pytest.mark.real_retry_backoff
    def test_a_raising_pool_fails_closed_to_legacy_behavior(self):
        """A broken probe must never silently disable the honor policy."""
        assert _backoff(_agent(_Pool(available=False, raises=True))) == 600.0


class TestNonRateLimitIsOutOfScope:
    """A 5xx / overload Retry-After is about the SERVER's capacity, not the
    seat's quota. Pinned so a future widening is a deliberate act."""

    @pytest.mark.real_retry_backoff
    def test_non_rate_limit_retry_after_honored_on_exhausted_seat(self):
        wait = _backoff(_agent(_Pool(available=False)), is_rate_limited=False)
        assert wait == 600.0


class TestGuardsThatMustNotHaveMoved:
    @pytest.mark.real_retry_backoff
    def test_600s_cap_still_applies_on_a_healthy_seat(self):
        assert _backoff(_agent(_Pool(available=True)), retry_after="99999") == 600.0

    @pytest.mark.real_retry_backoff
    def test_zero_retry_after_still_falls_through(self):
        """An expired cooldown carries no usable wait on any seat."""
        assert _backoff(_agent(_Pool(available=True)), retry_after="0") != 0.0


class TestPoolSeatExhaustionProbe:
    def test_reports_exhausted_when_pool_has_no_available_entry(self):
        assert pool_seat_exhaustion_state(_agent(_Pool(available=False))) == (True, None)

    def test_reports_healthy_when_an_entry_is_available(self):
        assert pool_seat_exhaustion_state(_agent(_Pool(available=True))) == (False, None)

    def test_converts_next_available_at_to_relative_seconds(self):
        pool = _Pool(available=False, next_at=time.time() + 1800.0)
        exhausted, recovery = pool_seat_exhaustion_state(_agent(pool))
        assert exhausted is True
        assert recovery == pytest.approx(1800.0, abs=5.0)

    def test_past_recovery_clamps_to_zero_not_negative(self):
        pool = _Pool(available=False, next_at=time.time() - 60.0)
        assert pool_seat_exhaustion_state(_agent(pool))[1] == 0.0

    def test_no_pool_is_not_exhaustion(self):
        assert pool_seat_exhaustion_state(_agent(None)) == (False, None)

    def test_cross_provider_pool_is_not_read(self):
        """On a fallback provider the bound pool belongs to the PRIMARY.
        Reading it would decline a Retry-After on another provider's quota."""
        pool = _Pool(available=False, provider="relay-1")
        assert pool_seat_exhaustion_state(_agent(pool, provider="other")) == (False, None)

    def test_a_raising_pool_fails_closed(self):
        pool = _Pool(available=False, raises=True)
        assert pool_seat_exhaustion_state(_agent(pool)) == (False, None)

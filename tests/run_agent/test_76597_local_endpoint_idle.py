"""Regression guard for #76597: local endpoints in idle/sleep mode must NOT be
treated as crashed (and must not eagerly fail over to remote fallback
providers).

LM Studio (``--sleep-idle-seconds``), Ollama, and other local servers close the
TCP socket while intentionally idle.  At the socket level a sleeping endpoint
is indistinguishable from a crashed one (``ECONNRESET`` / ``WinError 10053``).
Before this fix, Hermes:

  1. classified the connection reset as a transport failure,
  2. after 1 retry, eagerly switched to fallback providers (often
     rate-limited/dead — exhausting in seconds),
  3. crashed the gateway process once retries + fallbacks were exhausted.

The fix keeps the eager-fallback gate from firing for local endpoints on
transport failures, raises the retry ceiling so the endpoint gets a wake-up
window, and surfaces a distinct "endpoint may be sleeping" message.  These
tests mirror the decision helper (``should_eagerly_fallback`` /
``local_endpoint_wakeup_retry_ceiling``) that the turn loop now uses, in
lock-step with the source per tests/run_agent convention.
"""
from __future__ import annotations

from agent.retry_utils import (
    local_endpoint_wakeup_retry_ceiling,
    should_eagerly_fallback,
)
from agent.turn_retry_state import TurnRetryState


class TestLocalEndpointDoesNotEagerlyFallback:
    """Transport failures on local endpoints retry (wake-up), never fail over."""

    def test_local_transport_failure_never_eagerly_falls_back(self):
        """Sleeping local endpoint: even at retry_count >= 2, do NOT fall back."""
        for retry_count in (0, 1, 2, 5):
            assert not should_eagerly_fallback(
                is_rate_limited=False,
                is_transport_failure=True,
                retry_count=retry_count,
                is_local_endpoint=True,
            ), (
                f"Local endpoint transport failure at retry_count={retry_count} "
                "must retry (wake-up window) instead of eagerly failing over — #76597"
            )

    def test_non_local_transport_failure_still_falls_back(self):
        """Cloud endpoint behavior is unchanged: 1 retry, then fall back."""
        assert not should_eagerly_fallback(
            is_rate_limited=False,
            is_transport_failure=True,
            retry_count=1,
            is_local_endpoint=False,
        ), "Transient hiccup on a cloud endpoint deserves its single retry"
        assert should_eagerly_fallback(
            is_rate_limited=False,
            is_transport_failure=True,
            retry_count=2,
            is_local_endpoint=False,
        ), "Unreachable cloud provider must still fail over after 1 retry"

    def test_rate_limit_falls_back_even_for_local(self):
        """A 429 from a local server is a real condition — fall back as usual."""
        assert should_eagerly_fallback(
            is_rate_limited=True,
            is_transport_failure=True,
            retry_count=0,
            is_local_endpoint=True,
        )

    def test_no_fallback_on_success_or_unrelated_errors(self):
        """Rate limit + transport are the only eager-fallback triggers."""
        assert not should_eagerly_fallback(
            is_rate_limited=False,
            is_transport_failure=False,
            retry_count=9,
            is_local_endpoint=False,
        )


class TestLocalEndpointWakeUpWindow:
    """The retry ceiling must exceed api_max_retries (default 3) so the
    backoff schedule has room to give a sleeping endpoint time to wake."""

    def test_wakeup_ceiling_exceeds_default_retries(self):
        assert local_endpoint_wakeup_retry_ceiling() > 3, (
            "Wake-up ceiling must exceed the default api_max_retries (3) so "
            "the backoff schedule can actually run — #76597"
        )

    def test_wakeup_ceiling_is_reasonable(self):
        # ~2-3 minutes of jittered backoff (base 2s → max 60s) — enough for
        # LM Studio/Ollama to come back from sleep, not so long that a truly
        # crashed endpoint stalls the user forever.
        assert 5 <= local_endpoint_wakeup_retry_ceiling() <= 8


class TestTurnRetryStateIdleHint:
    """One-shot status hint so the user sees WHY Hermes is retrying."""

    def test_idle_hint_defaults_to_false(self):
        state = TurnRetryState()
        assert state.local_idle_hint_shown is False

    def test_idle_hint_is_one_shot(self):
        state = TurnRetryState()
        state.local_idle_hint_shown = True
        assert state.local_idle_hint_shown is True

"""Unit tests for the eager cross-provider fallback decision.

Regression for #143148: a hard HTTP 402 (``Insufficient Balance`` /
``insufficient_quota``) used to end the turn even though ``fallback_providers``
was configured. Because moving spend to another account is an operator decision,
the new billing branch is **opt-in** (``fallback.on_billing``, default false);
``billing_unverified`` 402s never jump providers.
"""

from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason
from agent.turn_recovery import _should_eager_fallback


def _classified(reason: FailoverReason, *, billing_unverified: bool = False):
    return SimpleNamespace(reason=reason, billing_unverified=billing_unverified)


def _decide(classified, *, retry_count=0, is_rate_limited=False, is_transport_failure=False, cap_budget=None):
    return _should_eager_fallback(
        classified,
        retry_count=retry_count,
        is_rate_limited=is_rate_limited,
        is_transport_failure=is_transport_failure,
        wrapped_output_cap_budget=cap_budget,
    )


@pytest.mark.parametrize("enabled,expected", [(False, False), (True, True)])
def test_billing_follows_opt_in_flag(monkeypatch, enabled, expected):
    """fallback.on_billing default false keeps the terminal path; true enables failover (#143148)."""
    monkeypatch.setattr("agent.turn_recovery.billing_failover_enabled", lambda: enabled)
    assert _decide(_classified(FailoverReason.billing)) is expected


def test_unverified_billing_never_jumps_even_when_enabled(monkeypatch):
    """An ambiguous 402 must not jump providers, even with the opt-in on."""
    monkeypatch.setattr("agent.turn_recovery.billing_failover_enabled", lambda: True)
    assert _decide(_classified(FailoverReason.billing, billing_unverified=True)) is False


def test_rate_limit_triggers_fallback_without_cap_budget(monkeypatch):
    monkeypatch.setattr("agent.turn_recovery.billing_failover_enabled", lambda: False)
    assert _decide(_classified(FailoverReason.rate_limit), is_rate_limited=True) is True


def test_rate_limit_with_parsed_cap_budget_does_not_fall_back(monkeypatch):
    """Relay-wrapped output-cap 429s are fixed by the max_tokens clamp, not a switch."""
    monkeypatch.setattr("agent.turn_recovery.billing_failover_enabled", lambda: False)
    assert (
        _decide(
            _classified(FailoverReason.rate_limit),
            is_rate_limited=True,
            cap_budget=4096,
        )
        is False
    )


@pytest.mark.parametrize("retry_count,expected", [(0, False), (1, False), (2, True), (5, True)])
def test_transport_failure_needs_two_retries(monkeypatch, retry_count, expected):
    monkeypatch.setattr("agent.turn_recovery.billing_failover_enabled", lambda: False)
    assert (
        _decide(
            _classified(FailoverReason.timeout),
            retry_count=retry_count,
            is_transport_failure=True,
        )
        is expected
    )


def test_auth_failure_is_not_eager_fallback(monkeypatch):
    """Auth has its own escalate-once branch; the eager gate must leave it alone."""
    monkeypatch.setattr("agent.turn_recovery.billing_failover_enabled", lambda: False)
    assert _decide(_classified(FailoverReason.auth)) is False


def test_billing_with_production_rate_limit_flag_stays_terminal(monkeypatch):
    """Production passes is_rate_limited=True for billing (_RATE_LIMIT_REASONS).

    The rate-limit shortcut must not bypass the opt-in: default false stays terminal.
    """
    monkeypatch.setattr("agent.turn_recovery.billing_failover_enabled", lambda: False)
    assert _decide(_classified(FailoverReason.billing), is_rate_limited=True) is False


def test_billing_with_production_rate_limit_flag_and_opt_in(monkeypatch):
    monkeypatch.setattr("agent.turn_recovery.billing_failover_enabled", lambda: True)
    assert _decide(_classified(FailoverReason.billing), is_rate_limited=True) is True


def test_billing_unverified_with_rate_limit_flag_never_jumps(monkeypatch):
    monkeypatch.setattr("agent.turn_recovery.billing_failover_enabled", lambda: True)
    assert _decide(_classified(FailoverReason.billing, billing_unverified=True), is_rate_limited=True) is False

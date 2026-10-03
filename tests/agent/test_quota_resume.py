"""Behaviour contracts for agent.quota_resume.

The plan is a *decision*, so the tests pin the decision boundaries: which
failures may be waited out, which reset source wins, and the cases where a
deadline is known but arming a timer would be wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from agent.quota_resume import (
    DEFAULT_MAX_WAIT_SECONDS,
    SOURCE_CREDENTIAL_POOL,
    SOURCE_PROVIDER_ERROR,
    SOURCE_USAGE_API,
    QuotaResumePlan,
    coerce_reset_timestamp,
    plan_quota_resume,
)

NOW = 1_800_000_000.0
HOUR = 3600.0


class _Pool:
    def __init__(self, value=None, raises: bool = False):
        self._value, self._raises = value, raises

    def next_available_at(self):
        if self._raises:
            raise RuntimeError("pool exploded")
        return self._value


@dataclass(frozen=True)
class _Window:
    label: str
    used_percent: Optional[float]
    reset_at: Optional[object]


@dataclass(frozen=True)
class _Snapshot:
    windows: tuple
    available: bool = True


def _install_usage_api(monkeypatch, snapshot, *, raises: bool = False):
    """Stub agent.account_usage.fetch_account_usage where quota_resume imports it."""
    import agent.account_usage as account_usage

    def _fake(provider, **_kwargs):
        if raises:
            raise RuntimeError("usage api down")
        return snapshot

    monkeypatch.setattr(account_usage, "fetch_account_usage", _fake)


# ── timestamp coercion ────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "value, expected",
    [
        (NOW + HOUR, NOW + HOUR),                       # epoch seconds
        ((NOW + HOUR) * 1000, NOW + HOUR),              # epoch milliseconds
        (str(NOW + HOUR), NOW + HOUR),                  # numeric string
        ("2027-01-15T10:30:00Z", 1_800_009_000.0),      # ISO-8601 Zulu
        ("2027-01-15T10:30:00+00:00", 1_800_009_000.0),
    ],
)
def test_coerce_reset_timestamp_accepts_provider_shapes(value, expected):
    assert coerce_reset_timestamp(value, now=NOW) == pytest.approx(expected, abs=1.0)


@pytest.mark.parametrize("value", [None, "", "not-a-date", True, False, {}, []])
def test_coerce_reset_timestamp_rejects_unusable(value):
    assert coerce_reset_timestamp(value, now=NOW) is None


def test_naive_iso_is_treated_as_utc():
    aware = coerce_reset_timestamp("2027-01-15T10:30:00+00:00", now=NOW)
    assert coerce_reset_timestamp("2027-01-15T10:30:00", now=NOW) == aware


# ── eligibility gate ──────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "reason",
    ["billing", "overloaded", "auth", "auth_permanent", "context_overflow",
     "server_error", "timeout", "model_not_found", "unknown", "", None],
)
def test_only_rate_limit_is_resumable(reason):
    """A deadline must never arm a timer for a wall that waiting cannot clear."""
    plan = plan_quota_resume(
        failure_reason=reason,
        error_context={"reset_at": NOW + HOUR},
        provider="anthropic",
        now=NOW,
        allow_usage_api=False,
    )
    assert plan.eligible is False
    assert plan.resume_at is None


def test_overload_with_no_deadline_is_not_resumable():
    """The captured Anthropic overload shape: HTTP 200 SSE error, no reset field."""
    plan = plan_quota_resume(
        failure_reason="overloaded",
        error_context={"reason": "overloaded_error", "message": "Overloaded"},
        provider="anthropic",
        now=NOW,
        allow_usage_api=False,
    )
    assert plan.eligible is False


def test_rate_limit_without_any_deadline_is_not_eligible():
    """No trustworthy source means no guess — the user keeps control."""
    plan = plan_quota_resume(
        failure_reason="rate_limit",
        error_context={"message": "Too many requests"},
        provider="some-unknown-provider",
        credential_pool=_Pool(None),
        now=NOW,
        allow_usage_api=False,
    )
    assert plan.eligible is False
    assert plan.resume_at is None
    assert plan.source == ""


def test_failure_reason_accepts_enum_like_value():
    class _Reason:
        value = "rate_limit"

    plan = plan_quota_resume(
        failure_reason=_Reason(),
        error_context={"reset_at": NOW + HOUR},
        provider="anthropic",
        now=NOW,
        allow_usage_api=False,
    )
    assert plan.eligible is True


# ── source precedence ─────────────────────────────────────────────────────────

def test_provider_error_deadline_wins_over_pool(monkeypatch):
    _install_usage_api(monkeypatch, None)
    plan = plan_quota_resume(
        failure_reason="rate_limit",
        error_context={"reset_at": NOW + HOUR},
        provider="anthropic",
        credential_pool=_Pool(NOW + 5 * HOUR),
        now=NOW,
        grace_seconds=0.0,
    )
    assert (plan.eligible, plan.source, plan.resume_at) == (True, SOURCE_PROVIDER_ERROR, NOW + HOUR)


def test_resets_in_seconds_alone_yields_a_deadline():
    """Codex sends resets_at *and* resets_in_seconds; a provider sending only the
    offset must not silently lose its deadline."""
    plan = plan_quota_resume(
        failure_reason="rate_limit",
        error_context={"reason": "usage_limit_reached", "resets_in_seconds": 900},
        provider="openai-codex",
        now=NOW,
        grace_seconds=0.0,
        allow_usage_api=False,
    )
    assert (plan.eligible, plan.source) == (True, SOURCE_PROVIDER_ERROR)
    assert plan.resume_at == NOW + 900


def test_pool_deadline_used_when_error_is_silent():
    plan = plan_quota_resume(
        failure_reason="rate_limit",
        error_context={"message": "rate limited"},
        provider="anthropic",
        credential_pool=_Pool(NOW + 2 * HOUR),
        now=NOW,
        grace_seconds=0.0,
        allow_usage_api=False,
    )
    assert (plan.eligible, plan.source, plan.resume_at) == (True, SOURCE_CREDENTIAL_POOL, NOW + 2 * HOUR)


def test_pool_returning_none_falls_through_to_usage_api(monkeypatch):
    """None from the pool means a credential can serve now — not 'no deadline'."""
    _install_usage_api(monkeypatch, _Snapshot(windows=(
        _Window("Current session", 100.0, NOW + 3 * HOUR),
    )))
    plan = plan_quota_resume(
        failure_reason="rate_limit",
        error_context={},
        provider="anthropic",
        credential_pool=_Pool(None),
        now=NOW,
        grace_seconds=0.0,
    )
    assert (plan.eligible, plan.source, plan.resume_at) == (True, SOURCE_USAGE_API, NOW + 3 * HOUR)


def test_usage_api_skipped_for_providers_without_one(monkeypatch):
    called = {"n": 0}

    import agent.account_usage as account_usage

    def _fake(provider, **_kwargs):
        called["n"] += 1
        return _Snapshot(windows=(_Window("w", 100.0, NOW + HOUR),))

    monkeypatch.setattr(account_usage, "fetch_account_usage", _fake)
    plan = plan_quota_resume(
        failure_reason="rate_limit", error_context={}, provider="openrouter", now=NOW,
    )
    assert plan.eligible is False
    assert called["n"] == 0


def test_usage_api_not_consulted_when_error_already_gave_a_deadline(monkeypatch):
    called = {"n": 0}

    import agent.account_usage as account_usage

    def _fake(provider, **_kwargs):
        called["n"] += 1
        return None

    monkeypatch.setattr(account_usage, "fetch_account_usage", _fake)
    plan_quota_resume(
        failure_reason="rate_limit",
        error_context={"reset_at": NOW + HOUR},
        provider="anthropic",
        now=NOW,
    )
    assert called["n"] == 0


# ── usage-API window selection ────────────────────────────────────────────────

def test_usage_api_ignores_windows_with_headroom(monkeypatch):
    """A fresh window's reset time is a rollover, not the wall we just hit."""
    _install_usage_api(monkeypatch, _Snapshot(windows=(
        _Window("Current session", 12.0, NOW + HOUR),        # plenty left
        _Window("Current week", 100.0, NOW + 6 * HOUR),      # actually exhausted
    )))
    plan = plan_quota_resume(
        failure_reason="rate_limit", error_context={}, provider="anthropic",
        now=NOW, grace_seconds=0.0,
    )
    assert (plan.eligible, plan.resume_at) == (True, NOW + 6 * HOUR)


def test_usage_api_picks_earliest_exhausted_window(monkeypatch):
    _install_usage_api(monkeypatch, _Snapshot(windows=(
        _Window("Opus week", 100.0, NOW + 10 * HOUR),
        _Window("Current session", 99.5, NOW + 2 * HOUR),
    )))
    plan = plan_quota_resume(
        failure_reason="rate_limit", error_context={}, provider="anthropic",
        now=NOW, grace_seconds=0.0,
    )
    assert plan.resume_at == NOW + 2 * HOUR


def test_usage_api_accepts_datetime_reset(monkeypatch):
    reset = datetime.fromtimestamp(NOW, tz=timezone.utc) + timedelta(hours=4)
    _install_usage_api(monkeypatch, _Snapshot(windows=(
        _Window("Current session", 100.0, reset),
    )))
    plan = plan_quota_resume(
        failure_reason="rate_limit", error_context={}, provider="anthropic",
        now=NOW, grace_seconds=0.0,
    )
    assert plan.resume_at == pytest.approx(NOW + 4 * HOUR, abs=1.0)


def test_unavailable_usage_snapshot_yields_no_plan(monkeypatch):
    _install_usage_api(monkeypatch, _Snapshot(windows=(), available=False))
    plan = plan_quota_resume(
        failure_reason="rate_limit", error_context={}, provider="anthropic", now=NOW,
    )
    assert plan.eligible is False


# ── bounds and degradation ────────────────────────────────────────────────────

def test_grace_period_is_added_to_the_deadline():
    plan = plan_quota_resume(
        failure_reason="rate_limit",
        error_context={"reset_at": NOW + HOUR},
        provider="anthropic", now=NOW, grace_seconds=45.0, allow_usage_api=False,
    )
    assert plan.resume_at == NOW + HOUR + 45.0


def test_deadline_beyond_max_wait_reports_but_does_not_arm():
    plan = plan_quota_resume(
        failure_reason="rate_limit",
        error_context={"reset_at": NOW + DEFAULT_MAX_WAIT_SECONDS + HOUR},
        provider="openai-codex", now=NOW, allow_usage_api=False,
    )
    assert plan.eligible is False
    assert plan.resume_at == NOW + DEFAULT_MAX_WAIT_SECONDS + HOUR
    assert plan.source == SOURCE_PROVIDER_ERROR


def test_past_deadline_reports_but_does_not_arm():
    plan = plan_quota_resume(
        failure_reason="rate_limit",
        error_context={"reset_at": NOW - HOUR},
        provider="anthropic", now=NOW, allow_usage_api=False,
    )
    assert plan.eligible is False
    assert plan.resume_at == NOW - HOUR


def test_pool_exception_degrades_to_no_plan():
    plan = plan_quota_resume(
        failure_reason="rate_limit", error_context={}, provider="anthropic",
        credential_pool=_Pool(raises=True), now=NOW, allow_usage_api=False,
    )
    assert plan.eligible is False


def test_usage_api_exception_degrades_to_no_plan(monkeypatch):
    _install_usage_api(monkeypatch, None, raises=True)
    plan = plan_quota_resume(
        failure_reason="rate_limit", error_context={}, provider="anthropic", now=NOW,
    )
    assert plan.eligible is False


def test_malformed_error_context_is_survivable():
    for bad in ("string", 42, [], None):
        plan = plan_quota_resume(
            failure_reason="rate_limit", error_context=bad, provider="anthropic",
            now=NOW, allow_usage_api=False,
        )
        assert plan.eligible is False


# ── wire form ─────────────────────────────────────────────────────────────────

def test_to_dict_omits_empty_fields():
    assert QuotaResumePlan().to_dict() == {"eligible": False}


def test_to_dict_round_trips_a_real_plan():
    plan = plan_quota_resume(
        failure_reason="rate_limit",
        error_context={"reset_at": NOW + HOUR},
        provider="anthropic", now=NOW, grace_seconds=0.0, allow_usage_api=False,
    )
    assert plan.to_dict() == {
        "eligible": True, "resume_at": NOW + HOUR,
        "source": SOURCE_PROVIDER_ERROR, "provider": "anthropic", "reason": "rate_limit",
    }


def test_seconds_until_resume_never_negative():
    plan = QuotaResumePlan(eligible=True, resume_at=time.time() - 500)
    assert plan.seconds_until_resume == 0.0

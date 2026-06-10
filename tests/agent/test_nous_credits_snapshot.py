"""Tests for build_nous_credits_snapshot (L6-A, magnitudes-only)."""

from __future__ import annotations

from agent.account_usage import build_nous_credits_snapshot
from hermes_cli.nous_account import (
    NousPaidServiceAccessInfo,
    NousPortalAccountInfo,
    NousPortalSubscriptionInfo,
)


def _account(**kwargs) -> NousPortalAccountInfo:
    kwargs.setdefault("logged_in", True)
    kwargs.setdefault("source", "account_api")
    kwargs.setdefault("fresh", True)
    return NousPortalAccountInfo(**kwargs)


def _all_lines(snapshot) -> list[str]:
    return list(snapshot.details)


def test_healthy():
    info = _account(
        paid_service_access=True,
        paid_service_access_info=NousPaidServiceAccessInfo(
            subscription_credits_remaining=18.0,
            purchased_credits_remaining=12.34,
            total_usable_credits=30.34,
        ),
        subscription=NousPortalSubscriptionInfo(
            plan="Pro",
            current_period_end="2026-07-01",
        ),
    )
    snap = build_nous_credits_snapshot(info)
    assert snap is not None
    assert snap.available is True
    assert snap.plan == "Pro"
    assert snap.provider == "nous"
    assert snap.title == "Nous credits"
    blob = "\n".join(_all_lines(snap))
    assert "$18.00" in blob
    assert "$12.34" in blob
    assert "$30.34" in blob
    assert "Renews: 2026-07-01" in blob
    assert "/billing" in blob
    # money-rule: magnitudes-only, never a percentage
    assert "%" not in blob


def test_money_rule_no_percent():
    info = _account(
        paid_service_access=True,
        paid_service_access_info=NousPaidServiceAccessInfo(
            subscription_credits_remaining=18.0,
            purchased_credits_remaining=12.34,
            total_usable_credits=30.34,
        ),
        subscription=NousPortalSubscriptionInfo(plan="Pro"),
    )
    snap = build_nous_credits_snapshot(info)
    assert snap is not None
    for line in snap.details:
        assert "%" not in line


def test_depleted():
    info = _account(
        paid_service_access=False,
        paid_service_access_info=NousPaidServiceAccessInfo(
            subscription_credits_remaining=0.0,
            purchased_credits_remaining=0.0,
            total_usable_credits=0.0,
        ),
        subscription=NousPortalSubscriptionInfo(plan="Pro"),
    )
    snap = build_nous_credits_snapshot(info)
    assert snap is not None
    blob = "\n".join(_all_lines(snap))
    assert "access depleted" in blob
    assert "/billing" in blob


def test_purchased_only():
    info = _account(
        paid_service_access=True,
        paid_service_access_info=NousPaidServiceAccessInfo(
            subscription_credits_remaining=None,
            purchased_credits_remaining=30.0,
            total_usable_credits=30.0,
        ),
        subscription=None,
    )
    snap = build_nous_credits_snapshot(info)
    assert snap is not None
    blob = "\n".join(_all_lines(snap))
    assert "Subscription credits" not in blob
    assert "Top-up credits: $30.00" in blob
    assert snap.plan is None


def test_logged_out():
    info = _account(
        logged_in=False,
        paid_service_access=True,
        paid_service_access_info=NousPaidServiceAccessInfo(
            total_usable_credits=10.0,
        ),
    )
    assert build_nous_credits_snapshot(info) is None


def test_none():
    assert build_nous_credits_snapshot(None) is None


def test_never_raises_empty():
    info = _account(
        paid_service_access=True,
        paid_service_access_info=None,
        subscription=None,
    )
    # No usable numbers and not depleted -> None, without raising.
    assert build_nous_credits_snapshot(info) is None

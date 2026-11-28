"""Tests for build_nous_credits_snapshot (L6-A, magnitudes-only)."""

from __future__ import annotations

from unittest.mock import Mock

from agent import account_usage_nous
from agent.account_usage_cache import has_account_usage
from agent.account_usage_nous import build_nous_credits_snapshot
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
    blob = "\n".join(_all_lines(snap))
    assert "$18.00" in blob
    assert "$12.34" in blob
    assert "$30.34" in blob
    assert "2026-07-01" in blob
    # money-rule: magnitudes-only, never a percentage
    assert "%" not in blob


def test_logged_out():
    info = _account(
        logged_in=False,
        paid_service_access=True,
        paid_service_access_info=NousPaidServiceAccessInfo(
            total_usable_credits=10.0,
        ),
    )
    assert build_nous_credits_snapshot(info) is None


def test_cli_fetch_reuses_portal_snapshot_without_interactive_topup_cta(monkeypatch):
    info = _account(
        paid_service_access=True,
        paid_service_access_info=NousPaidServiceAccessInfo(
            subscription_credits_remaining=18.0,
            purchased_credits_remaining=12.0,
            total_usable_credits=30.0,
        ),
        subscription=NousPortalSubscriptionInfo(plan="Pro", current_period_end="2026-07-01"),
    )
    monkeypatch.setattr(account_usage_nous, "_nous_logged_in", lambda: True)
    fetch_portal = Mock(return_value=info)
    monkeypatch.setattr(account_usage_nous, "_fetch_portal_account", fetch_portal)

    snapshot = account_usage_nous.fetch_nous_account_usage()

    assert snapshot is not None and snapshot.provider == "nous"
    fetch_portal.assert_called_once_with(10.0, stored_token_only=True)
    assert not any(line.startswith(("Top up:", "Top up via", "(or run")) for line in snapshot.details)
    assert not has_account_usage("nous")  # Keep interactive rendering and model-picker polling unchanged.

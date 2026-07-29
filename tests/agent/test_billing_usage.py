"""Tests for the shared dollar usage model (agent/billing_usage.py).

Behavior contracts: status classification, bar math, fail-open, and the
dollars-only / topup-split invariants the billing UX requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from agent.billing_usage import LOW_BALANCE_THRESHOLD_USD, UsageBar, usage_model_from_account


# ── Lightweight stand-ins for the NousPortalAccountInfo shape ────────────────


@dataclass
class _Access:
    subscription_credits_remaining: Optional[float] = None
    purchased_credits_remaining: Optional[float] = None
    total_usable_credits: Optional[float] = None


@dataclass
class _Sub:
    plan: Optional[str] = None
    monthly_credits: Optional[float] = None
    current_period_end: Optional[str] = None


@dataclass
class _Account:
    logged_in: bool = True
    paid_service_access: Optional[bool] = None
    paid_service_access_info: Optional[_Access] = None
    subscription: Optional[_Sub] = None


def _acct(**over):
    return _Account(**over)


class _Boom:
    @property
    def logged_in(self):
        raise RuntimeError("kaboom")


@pytest.mark.parametrize("account", [None, _acct(logged_in=False), _Boom()])
def test_fails_open_to_unavailable(account):
    assert usage_model_from_account(account).available is False


@pytest.mark.parametrize(
    "account,expected",
    [
        # no plan, no balance -> free
        (_acct(paid_service_access_info=_Access()), "free"),
        # paid access explicitly lost -> depleted
        (_acct(paid_service_access=False, subscription=_Sub(plan="Plus", monthly_credits=20.0),
               paid_service_access_info=_Access(subscription_credits_remaining=0.0, total_usable_credits=0.0)), "depleted"),
        # above threshold -> healthy
        (_acct(paid_service_access=True, subscription=_Sub(plan="Plus", monthly_credits=20.0),
               paid_service_access_info=_Access(subscription_credits_remaining=14.0, total_usable_credits=14.0)), "healthy"),
        # under $5 spendable -> low
        (_acct(paid_service_access=True, subscription=_Sub(plan="Plus", monthly_credits=20.0),
               paid_service_access_info=_Access(subscription_credits_remaining=3.4, total_usable_credits=3.4)), "low"),
        # exactly $5 -> healthy (the threshold boundary is exclusive)
        (_acct(paid_service_access=True, subscription=_Sub(plan="Plus", monthly_credits=20.0),
               paid_service_access_info=_Access(subscription_credits_remaining=5.0, total_usable_credits=5.0)), "healthy"),
        # top-up only, no plan -> usable (healthy), not free
        (_acct(paid_service_access=True, paid_service_access_info=_Access(purchased_credits_remaining=30.0, total_usable_credits=30.0)), "healthy"),
    ],
)
def test_status_classification(account, expected):
    m = usage_model_from_account(account)
    assert m.available is True
    assert m.status == expected


def test_threshold_constant_is_five():
    assert LOW_BALANCE_THRESHOLD_USD == 5.0


def test_healthy_carries_plan_name_and_renewal():
    m = usage_model_from_account(
        _acct(paid_service_access=True, subscription=_Sub(plan="Plus", monthly_credits=20.0, current_period_end="2026-07-01"),
              paid_service_access_info=_Access(subscription_credits_remaining=14.0, total_usable_credits=14.0))
    )
    assert m.plan_name == "Plus" and m.renews_at == "2026-07-01"


def test_plan_bar_spent_and_pct():
    m = usage_model_from_account(
        _acct(paid_service_access=True, subscription=_Sub(plan="Plus", monthly_credits=20.0),
              paid_service_access_info=_Access(subscription_credits_remaining=14.0, total_usable_credits=14.0))
    )
    bar = m.plan_bar
    assert bar is not None and bar.kind == "plan"
    assert (bar.remaining_usd, bar.total_usd, bar.pct_used) == (14.0, 20.0, 30)
    assert bar.spent_usd == pytest.approx(6.0)


def test_plan_bar_clamps_over_cap_to_zero_spent():
    # Rollover/debt: remaining > cap clamps to the cap and reads as zero spent.
    m = usage_model_from_account(
        _acct(paid_service_access=True, subscription=_Sub(plan="Plus", monthly_credits=20.0),
              paid_service_access_info=_Access(subscription_credits_remaining=25.0, total_usable_credits=25.0))
    )
    assert m.plan_bar.remaining_usd == 20.0 and m.plan_bar.spent_usd == 0.0


def test_topup_bar_is_full_with_no_denominator():
    m = usage_model_from_account(
        _acct(paid_service_access=True, subscription=_Sub(plan="Plus", monthly_credits=20.0),
              paid_service_access_info=_Access(subscription_credits_remaining=14.0, purchased_credits_remaining=12.0, total_usable_credits=26.0))
    )
    tb = m.topup_bar
    assert tb is not None and tb.kind == "topup"
    assert tb.remaining_usd == 12.0 and tb.fill_fraction == 1.0 and tb.pct_used is None
    assert m.total_spendable_usd == 26.0 and m.has_topup is True


def test_no_plan_bar_without_monthly_cap():
    m = usage_model_from_account(
        _acct(paid_service_access=True, paid_service_access_info=_Access(purchased_credits_remaining=8.0, total_usable_credits=8.0))
    )
    assert m.plan_bar is None and m.topup_bar is not None


def test_non_finite_values_are_ignored():
    m = usage_model_from_account(
        _acct(paid_service_access=True, subscription=_Sub(plan="Plus", monthly_credits=float("nan")),
              paid_service_access_info=_Access(subscription_credits_remaining=float("inf")))
    )
    assert m.plan_bar is None


def test_usage_bar_fill_fraction_clamped():
    assert UsageBar(kind="plan", remaining_usd=30.0, total_usd=20.0).fill_fraction == 1.0
    assert UsageBar(kind="plan", remaining_usd=-5.0, total_usd=20.0).fill_fraction == 0.0
    assert UsageBar(kind="plan", remaining_usd=0.0, total_usd=0.0).fill_fraction == 0.0

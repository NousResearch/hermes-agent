"""Tests for agent.subscription_view — the surface-agnostic /subscription core.

Behavior contracts (not change-detectors): the manage-URL builder's shape, the
payload parser's field mapping + fail-open posture, and the dev-fixture states
that drive the CLI/TUI without a live portal.
"""

from decimal import Decimal

import pytest

from agent.subscription_view import (
    CurrentSubscription,
    SubscriptionState,
    SubscriptionTier,
    build_subscription_state,
    dev_fixture_subscription_state,
    format_tier_row,
    is_upgrade,
    selectable_tiers,
    subscription_change_preview_from_payload,
    subscription_manage_url,
    subscription_state_from_payload,
)


def _tier(tid, order, dpm, mc, *, is_current=False, is_enabled=True):
    return SubscriptionTier(
        tier_id=tid,
        name=tid.title(),
        tier_order=order,
        dollars_per_month=Decimal(dpm) if dpm is not None else None,
        monthly_credits=Decimal(mc) if mc is not None else None,
        is_current=is_current,
        is_enabled=is_enabled,
    )


# ── subscription_manage_url ──────────────────────────────────────────




def test_manage_url_omits_org_when_absent():
    s = SubscriptionState(logged_in=True, org_id=None, portal_url="https://p.example.com/")
    url = subscription_manage_url(s)
    assert url == "https://p.example.com/manage-subscription"
    assert "org_id" not in url


















# ── shared plan-catalog helpers (format_tier_row / selectable_tiers / is_upgrade) ──




def test_format_tier_row_hides_absent_or_zero_credits():
    # A None / zero-credits tier hides the suffix — never "· — credits/mo" / "· $0 credits/mo".
    assert format_tier_row(_tier("plus", 1, "20", "0")) == "Plus · $20/mo"
    assert format_tier_row(_tier("plus", 1, "20", None)) == "Plus · $20/mo"
    assert "credits/mo" not in format_tier_row(_tier("plus", 1, "20", "0"))




def test_is_upgrade_by_tier_order():
    state = SubscriptionState(
        logged_in=True,
        current=CurrentSubscription(tier_id="plus", tier_name="Plus"),
        tiers=(_tier("plus", 1, "20", "22"), _tier("ultra", 3, "200", "220")),
    )
    assert is_upgrade(state, "ultra") is True
    assert is_upgrade(state, "plus") is False




# ── payload parser ───────────────────────────────────────────────────


def test_parser_maps_camelCase_payload_fields():
    payload = {
        "org": {"name": "Acme", "id": "org_1", "role": "ADMIN"},
        "context": "personal",
        "current": {
            "tierId": "plus",
            "tierName": "Plus",
            "monthlyCredits": "1000",
            "creditsRemaining": "420",
            "cycleEndsAt": "2026-07-01",
            "cancelAtPeriodEnd": True,
            "cancellationEffectiveAt": "2026-07-01",
        },
    }
    s = subscription_state_from_payload(payload, portal_url="https://p/billing")

    assert s.logged_in is True
    assert s.org_name == "Acme" and s.org_id == "org_1"
    assert s.is_admin is True and s.can_change_plan is True
    assert s.current is not None
    assert s.current.tier_name == "Plus"
    assert s.current.cancel_at_period_end is True
    assert s.current.monthly_credits == Decimal("1000")






@pytest.mark.parametrize(
    "role,can_change_plan_raw,is_admin,can_change_plan",
    [
        ("OWNER", None, True, True),
        ("ADMIN", None, True, True),
        ("FINANCE_ADMIN", True, False, True),
        ("SECURITY_ADMIN", None, False, False),
        ("MEMBER", None, False, False),
    ],
)
def test_parser_five_roles(
    role, can_change_plan_raw, is_admin, can_change_plan
):
    payload = {"org": {"role": role}}
    if can_change_plan_raw is not None:
        payload["canChangePlan"] = can_change_plan_raw

    state = subscription_state_from_payload(payload, portal_url=None)

    assert state.is_admin is is_admin
    assert state.can_change_plan_raw is can_change_plan_raw
    assert state.can_change_plan is can_change_plan








# ── tier catalog parsing (the picker) ────────────────────────────────


def test_parser_maps_tiers_catalog():
    payload = {
        "tiers": [
            {
                "tierId": "free",
                "name": "Free",
                "tierOrder": 0,
                "dollarsPerMonthDisplay": "0",
                "monthlyCredits": "0",
                "isCurrent": False,
                "isEnabled": True,
            },
            {
                "tierId": "plus",
                "name": "Plus",
                "tierOrder": 1,
                "dollarsPerMonthDisplay": "20",
                "monthlyCredits": "1000",
                "isCurrent": True,
                "isEnabled": True,
            },
        ],
    }
    s = subscription_state_from_payload(payload, portal_url=None)
    assert len(s.tiers) == 2
    free, plus = s.tiers
    # The free tier's 0s must survive (coalesce-on-None, not falsy `or`).
    assert free.tier_id == "free" and free.tier_order == 0
    assert free.dollars_per_month == Decimal("0")
    assert plus.is_current is True
    assert plus.dollars_per_month == Decimal("20") and plus.monthly_credits == Decimal("1000")




# ── preview parser (POST /preview) ───────────────────────────────────


def test_preview_parser_charge_now():
    p = subscription_change_preview_from_payload(
        {
            "effect": "charge_now",
            "reason": None,
            "currentTierId": "plus",
            "currentTierName": "Plus",
            "targetTierId": "ultra",
            "targetTierName": "Ultra",
            "monthlyCreditsDelta": "6000",
            "amountDueNowCents": 1234,
            "effectiveAt": None,
        }
    )
    assert p.effect == "charge_now"
    assert p.amount_due_now_cents == 1234
    assert p.target_tier_name == "Ultra"
    assert p.monthly_credits_delta == Decimal("6000")








# ── dev fixtures (env-driven, no live portal) ────────────────────────


def test_no_fixture_when_env_unset(monkeypatch):
    monkeypatch.delenv("HERMES_DEV_SUBSCRIPTION_FIXTURE", raising=False)
    assert dev_fixture_subscription_state() is None




def test_dev_fixture_exposes_tier_catalog(monkeypatch):
    # A picker needs a catalog: the mid fixture lists tiers with the active one flagged.
    monkeypatch.setenv("HERMES_DEV_SUBSCRIPTION_FIXTURE", "mid")
    s = dev_fixture_subscription_state()
    assert s is not None and len(s.tiers) >= 2
    current = [t for t in s.tiers if t.is_current]
    assert len(current) == 1 and current[0].tier_id == "plus"




def test_build_subscription_state_uses_fixture(monkeypatch):
    # build_subscription_state must short-circuit to the fixture (no portal call).
    monkeypatch.setenv("HERMES_DEV_SUBSCRIPTION_FIXTURE", "mid")
    s = build_subscription_state()
    assert s.logged_in is True
    assert s.current is not None and s.current.tier_id == "plus"
    # The manage URL is buildable from the fixture's portal_url + org_id.
    url = subscription_manage_url(s)
    assert url is not None
    assert url.endswith("/manage-subscription?org_id=org_acme")

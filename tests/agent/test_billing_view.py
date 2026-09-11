"""Unit tests for the Phase 2b Remote Spending core + HTTP client.

Covers:
- Decimal money parsing/formatting (server emits decimal strings, not 2dp).
- BillingState payload parsing (role tiering, presets, bounds, sub-structs).
- Error-code → typed-exception mapping (the live-verified contract matrix).
- Fail-open builder behavior.
- Idempotency key generation.
- Custom-amount validation against bounds + multipleOf 0.01.

No network: HTTP-layer tests drive _raise_for_error directly and monkeypatch the
request function for the builder.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

import agent.billing_view as bv
from agent.billing_view import (
    AutoReload,
    AutoReloadCard,
    BillingState,
    CardInfo,
    MonthlyCap,
    PaymentMethodInfo,
    billing_state_from_payload,
    build_billing_state,
    format_money,
    new_idempotency_key,
    parse_money,
    validate_charge_amount,
)
import hermes_cli.nous_billing as nb
from hermes_cli.nous_billing import (
    BillingAuthError,
    BillingError,
    BillingRateLimited,
    BillingScopeRequired,
    BillingStripeUnavailable,
    BillingTransient,
    BillingUpgradeCapExceeded,
    _raise_for_error,
    resolve_portal_base_url,
)


# ---------------------------------------------------------------------------
# Decimal money
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# BillingState payload parsing
# ---------------------------------------------------------------------------


def _member_payload() -> dict:
    return {
        "org": {"id": "o1", "slug": "acme", "name": "Acme", "role": "MEMBER"},
        "balanceUsd": "142.5",
        "cliBillingEnabled": True,
        "chargePresets": ["100", "250", "500"],
        "bounds": {"minUsd": "10", "maxUsd": "10000"},
        "card": None,
        "monthlyCap": None,
        "autoReload": None,
    }


def _owner_payload() -> dict:
    p = _member_payload()
    p["org"]["role"] = "OWNER"
    p["card"] = {"brand": "visa", "last4": "4242"}
    p["monthlyCap"] = {
        "limitUsd": "1000",
        "spentThisMonthUsd": "180",
        "isDefaultCeiling": True,
    }
    p["autoReload"] = {"enabled": True, "thresholdUsd": "20", "reloadToUsd": "100"}
    return p


def test_state_member_tier_parse():
    s = billing_state_from_payload(_member_payload())
    assert s.logged_in
    assert s.role == "MEMBER"
    assert s.balance_usd == Decimal("142.5")
    assert s.cli_billing_enabled is True
    assert s.charge_presets == (Decimal("100"), Decimal("250"), Decimal("500"))
    assert s.min_usd == Decimal("10") and s.max_usd == Decimal("10000")
    assert s.card is None and s.monthly_cap is None and s.auto_reload is None
    assert s.is_admin is False
    assert s.can_charge is False  # not admin


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
def test_state_five_roles(
    role, can_change_plan_raw, is_admin, can_change_plan
):
    payload = _member_payload()
    payload["org"]["role"] = role
    if can_change_plan_raw is not None:
        payload["canChangePlan"] = can_change_plan_raw

    state = billing_state_from_payload(payload)

    assert state.is_admin is is_admin
    assert state.can_change_plan_raw is can_change_plan_raw
    assert state.can_change_plan is can_change_plan
    if role == "SECURITY_ADMIN":
        assert state.card is None
        assert state.monthly_cap is None
        assert state.auto_reload is None








def test_state_owner_tier_parse():
    s = billing_state_from_payload(_owner_payload())
    assert s.is_admin is True
    assert s.can_charge is True  # admin + kill-switch on
    assert s.card == CardInfo(brand="visa", last4="4242")
    assert s.card is not None and s.card.masked == "visa ····4242"
    assert s.monthly_cap == MonthlyCap(
        limit_usd=Decimal("1000"),
        spent_this_month_usd=Decimal("180"),
        is_default_ceiling=True,
    )
    assert s.auto_reload == AutoReload(
        enabled=True, threshold_usd=Decimal("20"), reload_to_usd=Decimal("100")
    )
















# ---------------------------------------------------------------------------
# Error-code → typed-exception mapping (live-verified contract)
# ---------------------------------------------------------------------------


class _Headers:
    def __init__(self, d):
        self._d = d

    def get(self, k):
        return self._d.get(k)


def test_401_maps_to_auth_error():
    with pytest.raises(BillingAuthError) as ei:
        _raise_for_error(401, {"error": "invalid_token"})
    assert ei.value.status == 401


def test_403_insufficient_scope_maps_to_scope_required():
    with pytest.raises(BillingScopeRequired) as ei:
        _raise_for_error(403, {"error": "insufficient_scope", "portalUrl": "/billing"})
    assert ei.value.error == "insufficient_scope"
    # portalUrl is resolved to an absolute URL (relative-by-design from the server).
    assert (ei.value.portal_url or "").startswith("http")
    assert (ei.value.portal_url or "").endswith("/billing")




@pytest.mark.parametrize(
    "status,error,expected_type",
    [
        (503, "stripe_unavailable", BillingStripeUnavailable),
        (429, "upgrade_cap_exceeded", BillingUpgradeCapExceeded),
    ],
)
def test_specific_billing_throttle_errors_remain_distinguishable(
    status, error, expected_type
):
    with pytest.raises(expected_type) as ei:
        _raise_for_error(
            status,
            {"error": error},
            _Headers({"Retry-After": "90"}),
        )

    assert ei.value.error == error
    assert ei.value.retry_after == 90
    assert isinstance(ei.value, BillingTransient)
    assert not isinstance(ei.value, BillingRateLimited)






def test_400_amount_out_of_bounds_is_base_error():
    with pytest.raises(BillingError) as ei:
        _raise_for_error(400, {"error": "amount_out_of_bounds", "message": "too big"})
    assert ei.value.status == 400
    assert "too big" in str(ei.value)


# ---------------------------------------------------------------------------
# post_charge requires idempotency key (client-side guard)
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Base-URL resolution precedence
# ---------------------------------------------------------------------------






def test_portal_base_url_default(monkeypatch):
    monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
    monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    assert resolve_portal_base_url() == nb.DEFAULT_PORTAL_BASE_URL


# ---------------------------------------------------------------------------
# Fail-open builder
# ---------------------------------------------------------------------------




def test_build_billing_state_fail_open_on_http_error(monkeypatch):
    def _boom(*a, **kw):
        raise BillingError("portal exploded", status=500)

    monkeypatch.setattr(nb, "get_billing_state", _boom)
    s = build_billing_state()
    assert s.logged_in is False
    assert "portal exploded" in (s.error or "")






# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_new_idempotency_key_unique_and_uuid_shaped():
    a, b = new_idempotency_key(), new_idempotency_key()
    assert a != b
    assert len(a) == 36 and a.count("-") == 4


# ---------------------------------------------------------------------------
# Amount validation (Screen 3 custom input)
# ---------------------------------------------------------------------------






@pytest.mark.parametrize(
    "raw,err_substr",
    [
        ("", "dollar amount"),
        ("0", "greater than"),
        ("-5", "greater than"),
        ("10.005", "cent"),       # multipleOf 0.01 — sub-cent rejected
        ("5", "Minimum"),         # below bounds.minUsd
        ("99999", "Maximum"),     # above bounds.maxUsd
    ],
)
def test_validate_amount_rejections(raw, err_substr):
    v = validate_charge_amount(raw, min_usd=Decimal("10"), max_usd=Decimal("10000"))
    assert not v.ok
    assert err_substr.lower() in (v.error or "").lower()


# ---------------------------------------------------------------------------
# HERMES_DEV_BILLING_FIXTURE — offline card/scope state scaffolding (T0)
# ---------------------------------------------------------------------------




@pytest.mark.parametrize(
    "name,has_card,is_admin,billing_on",
    [
        ("nocard", False, True, True),
        ("card", True, True, True),
        ("card-autoreload", True, True, True),
        ("notadmin", True, False, True),
        ("billing-off", False, True, False),
    ],
)
def test_billing_fixture_card_and_gate_invariants(monkeypatch, name, has_card, is_admin, billing_on):
    """Each fixture state honors the card/admin/kill-switch contract the gate reads."""
    monkeypatch.setenv("HERMES_DEV_BILLING_FIXTURE", name)
    s = build_billing_state()
    assert s.logged_in is True
    assert (s.card is not None) is has_card
    assert s.is_admin is is_admin
    assert s.cli_billing_enabled is billing_on





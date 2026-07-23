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


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("142.5", Decimal("142.5")),   # decimal string, NOT 2dp — the headline case
        ("100", Decimal("100")),
        ("10000", Decimal("10000")),
        ("0.01", Decimal("0.01")),
        (250, Decimal("250")),
        ("  50  ", Decimal("50")),
    ],
)
def test_parse_money_valid(raw, expected):
    assert parse_money(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "abc", "1.2.3", "$5", {}])
def test_parse_money_invalid_returns_none(raw):
    assert parse_money(raw) is None


def test_parse_money_never_uses_binary_float():
    # If a float ever sneaks through, we still get an exact decimal, not 0.1+0.2 junk.
    assert parse_money(0.1) == Decimal("0.1")


@pytest.mark.parametrize(
    "value,expected",
    [
        (Decimal("142.5"), "$142.50"),
        (Decimal("100"), "$100"),
        (Decimal("0.01"), "$0.01"),
        (Decimal("1000"), "$1000"),
        (None, "—"),
    ],
)
def test_format_money(value, expected):
    assert format_money(value) == expected


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


@pytest.mark.parametrize(
    "role,server_capability",
    [("MEMBER", True), ("OWNER", False)],
)
def test_state_can_change_plan_prefers_server_capability(role, server_capability):
    payload = _member_payload()
    payload["org"]["role"] = role
    payload["canChangePlan"] = server_capability

    state = billing_state_from_payload(payload)

    assert state.can_change_plan is server_capability


def test_state_can_change_plan_falls_back_to_legacy_role_check():
    owner = _member_payload()
    owner["org"]["role"] = "OWNER"
    member = _member_payload()

    assert billing_state_from_payload(owner).can_change_plan is True
    assert billing_state_from_payload(member).can_change_plan is False


def test_can_charge_finance_admin_with_server_capability():
    """Server capability can grant FINANCE_ADMIN charge access."""
    payload = _member_payload()
    payload["org"]["role"] = "FINANCE_ADMIN"
    payload["canChangePlan"] = True

    state = billing_state_from_payload(payload)

    assert state.is_admin is False
    assert state.can_change_plan is True
    assert state.can_charge is True


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


@pytest.mark.parametrize(
    "raw_card,expected",
    [
        (
            {"kind": "canonical", "paymentMethodId": "ignored", "brand": "ignored"},
            AutoReloadCard(kind="canonical"),
        ),
        (
            {
                "kind": "distinct",
                "paymentMethodId": "pm_auto",
                "brand": None,
                "last4": None,
            },
            AutoReloadCard(
                kind="distinct",
                payment_method_id="pm_auto",
                brand=None,
                last4=None,
            ),
        ),
        (
            {"kind": "none", "last4": "ignored"},
            AutoReloadCard(kind="none"),
        ),
    ],
)
def test_state_parses_auto_reload_card_variants(raw_card, expected):
    payload = _owner_payload()
    payload["autoReload"]["card"] = raw_card

    state = billing_state_from_payload(payload)

    assert state.auto_reload is not None
    assert state.auto_reload.card == expected


@pytest.mark.parametrize("raw_card", [None, "canonical", {}, {"kind": "future"}])
def test_state_ignores_unrecognized_auto_reload_card(raw_card):
    payload = _owner_payload()
    payload["autoReload"]["card"] = raw_card

    state = billing_state_from_payload(payload)

    assert state.auto_reload is not None
    assert state.auto_reload.card is None


def test_state_can_charge_false_when_killswitch_off():
    p = _owner_payload()
    p["cliBillingEnabled"] = False
    s = billing_state_from_payload(p)
    assert s.is_admin is True
    assert s.can_charge is False  # kill-switch off gates the action


def test_state_handles_garbage_substructs():
    p = _member_payload()
    p["card"] = "not-a-dict"
    p["monthlyCap"] = 42
    p["chargePresets"] = ["100", "bad", "250"]  # bad preset dropped, not crash
    s = billing_state_from_payload(p)
    assert s.card is None and s.monthly_cap is None
    assert s.charge_presets == (Decimal("100"), Decimal("250"))


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


@pytest.mark.parametrize("status", [429, 503])
def test_rate_limited_maps_with_retry_after(status):
    with pytest.raises(BillingRateLimited) as ei:
        _raise_for_error(
            status,
            {"error": "rate_limited"},
            _Headers({"Retry-After": "60"}),
        )
    assert ei.value.retry_after == 60
    # Critically: a rate limit is NOT a generic BillingError-only — surfaces branch on type.
    assert isinstance(ei.value, BillingRateLimited)


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


@pytest.mark.parametrize(
    "error",
    [
        "no_payment_method",
        "cli_billing_disabled",
        "role_required",
        "monthly_cap_exceeded",
        "org_access_denied",
    ],
)
def test_other_403s_map_to_base_error_with_portal_url(error):
    with pytest.raises(BillingError) as ei:
        _raise_for_error(403, {"error": error, "portalUrl": "/billing?topup=open"})
    # Not a scope/auth/rate subclass — the generic gate-denial path.
    assert not isinstance(ei.value, (BillingScopeRequired, BillingAuthError, BillingRateLimited))
    assert ei.value.error == error
    # portalUrl resolved to an absolute deep-link (server sends it relative).
    assert (ei.value.portal_url or "").startswith("http")
    assert (ei.value.portal_url or "").endswith("/billing?topup=open")


def test_monthly_cap_exceeded_carries_remaining_in_payload():
    with pytest.raises(BillingError) as ei:
        _raise_for_error(
            403,
            {
                "error": "monthly_cap_exceeded",
                "remainingUsd": "12.50",
                "isDefaultCeiling": True,
                "portalUrl": "/billing",
            },
        )
    assert ei.value.payload["remainingUsd"] == "12.50"
    assert ei.value.payload["isDefaultCeiling"] is True


def test_400_amount_out_of_bounds_is_base_error():
    with pytest.raises(BillingError) as ei:
        _raise_for_error(400, {"error": "amount_out_of_bounds", "message": "too big"})
    assert ei.value.status == 400
    assert "too big" in str(ei.value)


# ---------------------------------------------------------------------------
# post_charge requires idempotency key (client-side guard)
# ---------------------------------------------------------------------------


def test_post_charge_requires_idempotency_key():
    with pytest.raises(BillingError) as ei:
        nb.post_charge(amount_usd=50, idempotency_key="")
    assert ei.value.error == "idempotency_key_required"


def test_get_charge_status_requires_id():
    with pytest.raises(BillingError) as ei:
        nb.get_charge_status("")
    assert ei.value.error == "invalid_charge_id"


# ---------------------------------------------------------------------------
# Base-URL resolution precedence
# ---------------------------------------------------------------------------


def test_portal_base_url_env_override(monkeypatch):
    monkeypatch.setenv("HERMES_PORTAL_BASE_URL", "https://preview.example.com/")
    assert resolve_portal_base_url() == "https://preview.example.com"


def test_portal_base_url_falls_back_to_state(monkeypatch):
    monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
    monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    assert (
        resolve_portal_base_url({"portal_base_url": "https://stored.example.com/"})
        == "https://stored.example.com"
    )


def test_portal_base_url_default(monkeypatch):
    monkeypatch.delenv("HERMES_PORTAL_BASE_URL", raising=False)
    monkeypatch.delenv("NOUS_PORTAL_BASE_URL", raising=False)
    assert resolve_portal_base_url() == nb.DEFAULT_PORTAL_BASE_URL


# ---------------------------------------------------------------------------
# Fail-open builder
# ---------------------------------------------------------------------------


def test_build_billing_state_logged_out_on_auth_error(monkeypatch):
    def _auth(*a, **kw):
        raise BillingAuthError("nope", status=401)

    monkeypatch.setattr(nb, "get_billing_state", _auth)
    s = build_billing_state()
    assert s.logged_in is False
    assert s.error is None  # cleanly logged out, not an error


def test_build_billing_state_fail_open_on_http_error(monkeypatch):
    def _boom(*a, **kw):
        raise BillingError("portal exploded", status=500)

    monkeypatch.setattr(nb, "get_billing_state", _boom)
    s = build_billing_state()
    assert s.logged_in is False
    assert "portal exploded" in (s.error or "")


def test_build_billing_state_parses_and_prefers_server_portal_url(monkeypatch):
    payload = _owner_payload()
    payload["portalUrl"] = "https://portal.example.com/billing?topup=open"
    monkeypatch.setattr(nb, "get_billing_state", lambda *a, **kw: payload)
    s = build_billing_state()
    assert s.logged_in is True
    assert s.portal_url == "https://portal.example.com/billing?topup=open"
    assert s.balance_usd == Decimal("142.5")


def test_build_billing_state_builds_fallback_portal_url(monkeypatch):
    payload = _member_payload()  # no portalUrl key
    monkeypatch.setattr(nb, "get_billing_state", lambda *a, **kw: payload)
    monkeypatch.setattr(bv, "_fallback_portal_url", lambda base: "FALLBACK")
    # resolve_portal_base_url is imported into bv via local import; patch nb's.
    s = build_billing_state()
    assert s.portal_url == "FALLBACK"


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


def test_validate_amount_ok():
    v = validate_charge_amount("100", min_usd=Decimal("10"), max_usd=Decimal("10000"))
    assert v.ok and v.amount == Decimal("100")


def test_validate_amount_strips_dollar_sign():
    v = validate_charge_amount("$250", min_usd=Decimal("10"), max_usd=Decimal("10000"))
    assert v.ok and v.amount == Decimal("250")


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


def test_billing_fixture_unset_returns_none(monkeypatch):
    """No env var → fixture is inert (the real portal path runs)."""
    monkeypatch.delenv("HERMES_DEV_BILLING_FIXTURE", raising=False)
    assert bv._dev_fixture_billing_state() is None


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


def test_billing_fixture_autoreload_state(monkeypatch):
    """card-autoreload pairs a card with an enabled auto-reload (drives that screen)."""
    monkeypatch.setenv("HERMES_DEV_BILLING_FIXTURE", "card-autoreload")
    s = build_billing_state()
    assert s.card is not None
    assert s.auto_reload is not None and s.auto_reload.enabled is True


def test_billing_fixture_logged_out_and_unknown(monkeypatch):
    monkeypatch.setenv("HERMES_DEV_BILLING_FIXTURE", "logged-out")
    assert build_billing_state().logged_in is False
    monkeypatch.setenv("HERMES_DEV_BILLING_FIXTURE", "bogus-state")
    s = build_billing_state()
    assert s.logged_in is False and "bogus-state" in (s.error or "")

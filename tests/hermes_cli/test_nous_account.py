"""Tests for normalized Nous Portal account entitlement helpers."""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import pytest

from hermes_cli.nous_account import (
    NousPaidServiceAccessInfo,
    NousPortalAccountInfo,
    format_nous_portal_entitlement_message,
    get_nous_portal_account_info,
    nous_portal_topup_url,
    reset_nous_portal_account_info_cache,
)


def _jwt(claims: dict[str, Any]) -> str:
    def _part(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{_part({'alg': 'none', 'typ': 'JWT'})}.{_part(claims)}.sig"


def _state(token: str) -> dict[str, Any]:
    return {
        "access_token": token,
        "portal_base_url": "https://portal.example.test",
        "client_id": "hermes-cli",
    }


def _account_payload(
    *,
    allowed: bool,
    subscription: dict[str, Any] | None,
    subscription_credits: float,
    purchased_credits: float,
    member_spend_cap_exceeded: bool | None = None,
    member_spend_cap_usd: float | str | None = None,
    member_spend_usd: float | str | None = None,
    member_spend_cap_remaining_usd: float | str | None = None,
) -> dict[str, Any]:
    psa: dict[str, Any] = {
        "allowed": allowed,
        "paid_access": allowed,
        "reason": "usable_credits" if allowed else "no_usable_credits",
        "organisation_id": "org_123",
        "effective_at_ms": 123456789,
        "has_active_subscription": subscription is not None,
        "active_subscription_is_paid": bool(
            subscription and subscription.get("monthly_charge", 0) > 0
        ),
        "subscription_tier": subscription.get("tier") if subscription else None,
        "subscription_monthly_charge": (
            subscription.get("monthly_charge") if subscription else None
        ),
        "subscription_credits_remaining": subscription_credits,
        "purchased_credits_remaining": purchased_credits,
        "total_usable_credits": subscription_credits + purchased_credits,
    }
    if member_spend_cap_exceeded is not None:
        psa["member_spend_cap_exceeded"] = member_spend_cap_exceeded
        psa["reason"] = "member_spend_cap_exceeded"
    if member_spend_cap_usd is not None:
        psa["member_spend_cap_usd"] = member_spend_cap_usd
    if member_spend_usd is not None:
        psa["member_spend_usd"] = member_spend_usd
    if member_spend_cap_remaining_usd is not None:
        psa["member_spend_cap_remaining_usd"] = member_spend_cap_remaining_usd
    return {
        "user": {
            "email": "alice@example.test",
            "privy_did": "did:privy:alice",
        },
        "organisation": {
            "id": "org_123",
        },
        "subscription": subscription,
        "purchased_credits_remaining": purchased_credits,
        "paid_service_access": psa,
    }


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_nous_portal_account_info_cache()
    yield
    reset_nous_portal_account_info_cache()






@pytest.mark.parametrize(
    ("payload", "expected_paid"),
    [
        (
            _account_payload(
                allowed=True,
                subscription={
                    "plan": "Tier 2",
                    "tier": 2,
                    "monthly_charge": 20,
                    "current_period_end": "2026-05-01T00:00:00.000Z",
                    "credits_remaining": 12.25,
                    "rollover_credits": 3.5,
                },
                subscription_credits=12.25,
                purchased_credits=7.75,
            ),
            True,
        ),
        (
            _account_payload(
                allowed=False,
                subscription={
                    "plan": "Tier 2",
                    "tier": 2,
                    "monthly_charge": 20,
                    "current_period_end": "2026-05-01T00:00:00.000Z",
                    "credits_remaining": 0,
                    "rollover_credits": 0,
                },
                subscription_credits=0,
                purchased_credits=0,
            ),
            False,
        ),
        (
            _account_payload(
                allowed=True,
                subscription=None,
                subscription_credits=0,
                purchased_credits=7.75,
            ),
            True,
        ),
        (
            _account_payload(
                allowed=False,
                subscription=None,
                subscription_credits=0,
                purchased_credits=0,
            ),
            False,
        ),
    ],
)
def test_fresh_account_payload_normalization(monkeypatch, payload, expected_paid):
    token = _jwt({"sub": "user_123", "org_id": "org_123", "exp": int(time.time()) + 900})
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: _state(token))
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda: "fresh-token")
    monkeypatch.setattr("hermes_cli.nous_account._fetch_nous_account_info", lambda *a, **kw: payload)

    info = get_nous_portal_account_info(force_fresh=True)

    assert isinstance(info, NousPortalAccountInfo)
    assert info.source == "account_api"
    assert info.fresh is True
    assert info.email == "alice@example.test"
    assert info.privy_did == "did:privy:alice"
    assert info.org_id == "org_123"
    assert info.paid_service_access is expected_paid
    assert info.is_paid is expected_paid
    assert info.is_free_tier is (not expected_paid)


def test_no_oauth_token_reports_inference_key_present(monkeypatch):
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: {})

    class _Entry:
        label = "manual-nous"
        access_token = ""
        agent_key = "opaque-runtime-key"
        agent_key_expires_at = "2099-01-01T00:00:00+00:00"
        expires_at = None
        inference_base_url = "https://inference.example.test/v1"
        base_url = "https://inference.example.test/v1"
        priority = 0

        @property
        def runtime_api_key(self):
            return self.agent_key

        @property
        def runtime_base_url(self):
            return self.inference_base_url

    class _Pool:
        def has_credentials(self):
            return True

        def entries(self):
            return [_Entry()]

    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: _Pool())

    info = get_nous_portal_account_info()

    assert info.logged_in is False
    assert info.source == "inference_key"
    assert info.inference_credential_present is True
    assert info.credential_source == "pool:manual-nous"
    assert info.paid_service_access is None


def test_pool_oauth_entry_force_fresh_uses_account_api(monkeypatch):
    token = _jwt(
        {
            "sub": "user_123",
            "org_id": "org_123",
            "exp": int(time.time()) + 900,
            "paid_access": False,
        }
    )
    payload = _account_payload(
        allowed=True,
        subscription=None,
        subscription_credits=0,
        purchased_credits=3,
    )
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: {})
    monkeypatch.setattr("hermes_cli.nous_account._fetch_nous_account_info", lambda *a, **kw: payload)

    class _Entry:
        label = "dashboard device_code"
        auth_type = "oauth"
        access_token = token
        refresh_token = "refresh-token"
        agent_key = "opaque-runtime-key"
        agent_key_expires_at = "2099-01-01T00:00:00+00:00"
        expires_at = "2099-01-01T00:00:00+00:00"
        portal_base_url = "https://portal.example.test"
        inference_base_url = "https://inference.example.test/v1"
        base_url = "https://inference.example.test/v1"
        priority = 0

        @property
        def runtime_api_key(self):
            return self.agent_key

        @property
        def runtime_base_url(self):
            return self.inference_base_url

    class _Pool:
        def has_credentials(self):
            return True

        def entries(self):
            return [_Entry()]

    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: _Pool())

    info = get_nous_portal_account_info(force_fresh=True)

    assert info.logged_in is True
    assert info.source == "account_api"
    assert info.fresh is True
    assert info.paid_service_access is True
    assert info.credential_source == "pool:dashboard device_code"


# ── member spend cap exceeded ───────────────────────────────────────────────


def test_member_spend_cap_exceeded_message(monkeypatch):
    """When the Portal returns member_spend_cap_exceeded, the entitlement
    message should explain the cap — not say 'no credits'."""
    payload = _account_payload(
        allowed=False,
        subscription=None,
        subscription_credits=0,
        purchased_credits=222990.17,
        member_spend_cap_exceeded=True,
        member_spend_cap_usd="500",
        member_spend_usd="520.51",
        member_spend_cap_remaining_usd="0",
    )
    token = _jwt({"sub": "user_123", "org_id": "org_123", "exp": int(time.time()) + 900})
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: _state(token))
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda: "fresh-token")
    monkeypatch.setattr("hermes_cli.nous_account._fetch_nous_account_info", lambda *a, **kw: payload)

    info = get_nous_portal_account_info(force_fresh=True)

    assert info.paid_service_access is False
    assert info.paid_service_access_info is not None
    assert info.paid_service_access_info.member_spend_cap_exceeded is True
    assert info.paid_service_access_info.member_spend_cap_usd == 500.0
    assert info.paid_service_access_info.member_spend_usd == 520.51

    msg = format_nous_portal_entitlement_message(info, capability="Nous model access")
    assert msg is not None
    # Must mention spend cap, not "no active subscription or usable credits"
    assert "spend cap" in msg
    assert "$500.00" in msg
    assert "$520.51" in msg
    # Should NOT say "no active subscription"
    assert "no active subscription" not in msg
    # Should still show available credits
    assert "$222990.17" in msg


def test_member_spend_cap_exceeded_without_amounts(monkeypatch):
    """Even if the Portal doesn't include cap/spend amounts, the message
    should still mention the spend cap rather than credits."""
    payload = _account_payload(
        allowed=False,
        subscription=None,
        subscription_credits=0,
        purchased_credits=100,
        member_spend_cap_exceeded=True,
    )
    token = _jwt({"sub": "user_123", "org_id": "org_123", "exp": int(time.time()) + 900})
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: _state(token))
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda: "fresh-token")
    monkeypatch.setattr("hermes_cli.nous_account._fetch_nous_account_info", lambda *a, **kw: payload)

    info = get_nous_portal_account_info(force_fresh=True)
    msg = format_nous_portal_entitlement_message(info, capability="Nous model access")
    assert msg is not None
    assert "spend cap" in msg
    assert "no active subscription" not in msg




# ── org slug/name parsing + top-up URL builder ──────────────────────────────







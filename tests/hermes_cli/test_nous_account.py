"""Tests for normalized Nous Portal account entitlement helpers."""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import pytest

import hermes_cli.nous_account as nous_account
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
) -> dict[str, Any]:
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
        "paid_service_access": {
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
        },
    }


class _FakePortalResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.read_sizes: list[int] = []

    def __enter__(self) -> "_FakePortalResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size is None or size < 0:
            return self.body
        return self.body[:size]


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_nous_portal_account_info_cache()
    yield
    reset_nous_portal_account_info_cache()


def test_fetch_nous_account_info_bounds_response_read(monkeypatch):
    response = _FakePortalResponse(
        json.dumps({"user": {"email": "alice@example.test"}}).encode()
    )
    captured = []

    def _urlopen(req, timeout):
        captured.append((req, timeout))
        return response

    monkeypatch.setattr(nous_account.urllib.request, "urlopen", _urlopen)

    payload = nous_account._fetch_nous_account_info(
        "portal-token",
        "https://portal.example.test/",
    )

    assert payload == {"user": {"email": "alice@example.test"}}
    assert response.read_sizes == [
        nous_account._NOUS_ACCOUNT_RESPONSE_BODY_MAX_BYTES + 1
    ]
    assert captured[0][0].full_url == "https://portal.example.test/api/oauth/account"
    assert captured[0][1] == 8


def test_fetch_nous_account_info_rejects_oversized_response(monkeypatch):
    response = _FakePortalResponse(
        b"x" * (nous_account._NOUS_ACCOUNT_RESPONSE_BODY_MAX_BYTES + 1)
    )

    monkeypatch.setattr(
        nous_account.urllib.request,
        "urlopen",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(nous_account._NousAccountResponseTooLarge):
        nous_account._fetch_nous_account_info("portal-token")

    assert response.read_sizes == [
        nous_account._NOUS_ACCOUNT_RESPONSE_BODY_MAX_BYTES + 1
    ]


def test_valid_jwt_with_paid_access_true(monkeypatch):
    token = _jwt(
        {
            "sub": "user_123",
            "org_id": "org_123",
            "client_id": "hermes-cli",
            "product_id": "nous-hermes-agent",
            "nous_client": "hermes-agent",
            "exp": int(time.time()) + 900,
            "paid_access": True,
            "subscription_tier": 2,
        }
    )
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: _state(token))

    info = get_nous_portal_account_info()

    assert info.source == "jwt"
    assert info.fresh is False
    assert info.logged_in is True
    assert info.user_id == "user_123"
    assert info.org_id == "org_123"
    assert info.product_id == "nous-hermes-agent"
    assert info.paid_service_access is True
    assert info.is_paid is True
    assert info.is_free_tier is False


def test_valid_jwt_with_paid_access_false(monkeypatch):
    token = _jwt(
        {
            "sub": "user_123",
            "org_id": "org_123",
            "exp": int(time.time()) + 900,
            "paid_access": False,
        }
    )
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: _state(token))

    info = get_nous_portal_account_info()

    assert info.source == "jwt"
    assert info.paid_service_access is False
    assert info.is_paid is False
    assert info.is_free_tier is True


def test_valid_jwt_missing_paid_access_is_unknown_not_paid(monkeypatch):
    token = _jwt(
        {
            "sub": "user_123",
            "org_id": "org_123",
            "exp": int(time.time()) + 900,
        }
    )
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: _state(token))

    info = get_nous_portal_account_info()

    assert info.source == "jwt"
    assert info.paid_service_access is None
    assert info.is_paid is False
    assert info.is_free_tier is False


def test_expired_jwt_falls_back_to_fresh_account(monkeypatch):
    token = _jwt(
        {
            "sub": "user_123",
            "org_id": "org_123",
            "exp": int(time.time()) - 60,
            "paid_access": False,
        }
    )
    payload = _account_payload(
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
    )
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: _state(token))
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda: "fresh-token")
    monkeypatch.setattr("hermes_cli.nous_account._fetch_nous_account_info", lambda *a, **kw: payload)

    info = get_nous_portal_account_info()

    assert info.source == "account_api"
    assert info.fresh is True
    assert info.paid_service_access is True
    assert info.subscription is not None
    assert info.subscription.monthly_charge == 20
    assert info.paid_service_access_info is not None
    assert info.paid_service_access_info.total_usable_credits == 20


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


def test_force_fresh_uses_account_api_even_when_jwt_is_valid(monkeypatch):
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
        purchased_credits=5,
    )
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: _state(token))
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda: "fresh-token")
    monkeypatch.setattr("hermes_cli.nous_account._fetch_nous_account_info", lambda *a, **kw: payload)

    info = get_nous_portal_account_info(force_fresh=True)

    assert info.source == "account_api"
    assert info.paid_service_access is True


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


def test_pool_oauth_entry_uses_jwt_snapshot(monkeypatch):
    token = _jwt(
        {
            "sub": "user_123",
            "org_id": "org_123",
            "client_id": "hermes-cli",
            "exp": int(time.time()) + 900,
            "paid_access": True,
        }
    )
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: {})

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

    info = get_nous_portal_account_info()

    assert info.logged_in is True
    assert info.source == "jwt"
    assert info.paid_service_access is True
    assert info.credential_source == "pool:dashboard device_code"


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


def test_entitlement_message_returns_none_for_paid_access():
    info = NousPortalAccountInfo(
        logged_in=True,
        source="account_api",
        fresh=True,
        paid_service_access=True,
        portal_base_url="https://portal.example.test",
    )

    assert format_nous_portal_entitlement_message(info, capability="paid models") is None


def test_entitlement_message_for_inference_key_without_portal_login():
    info = NousPortalAccountInfo(
        logged_in=False,
        source="inference_key",
        fresh=False,
        inference_credential_present=True,
        portal_base_url="https://portal.example.test",
    )

    message = format_nous_portal_entitlement_message(
        info,
        capability="managed tools",
    )

    assert message is not None
    assert "Nous inference credentials are configured" in message
    assert "cannot verify your Nous Portal paid access" in message
    assert "Log in with `hermes model`" in message


def test_entitlement_message_for_active_paid_subscription_with_no_credits():
    info = NousPortalAccountInfo(
        logged_in=True,
        source="account_api",
        fresh=True,
        paid_service_access=False,
        portal_base_url="https://portal.example.test",
        paid_service_access_info=NousPaidServiceAccessInfo(
            allowed=False,
            reason="no_usable_credits",
            has_active_subscription=True,
            active_subscription_is_paid=True,
            subscription_credits_remaining=0,
            purchased_credits_remaining=0,
            total_usable_credits=0,
        ),
    )

    message = format_nous_portal_entitlement_message(
        info,
        capability="managed tools",
    )

    assert message is not None
    assert "credits are exhausted" in message
    assert "managed tools" in message
    assert "https://portal.example.test/billing" in message


def test_entitlement_message_for_no_subscription_or_credits():
    info = NousPortalAccountInfo(
        logged_in=True,
        source="account_api",
        fresh=True,
        paid_service_access=False,
        portal_base_url="https://portal.example.test",
        paid_service_access_info=NousPaidServiceAccessInfo(
            allowed=False,
            reason="no_usable_credits",
            has_active_subscription=False,
            subscription_credits_remaining=0,
            purchased_credits_remaining=0,
            total_usable_credits=0,
        ),
    )

    message = format_nous_portal_entitlement_message(info, capability="paid models")

    assert message is not None
    assert "no active subscription or usable credits" in message
    assert "Subscribe or add credits" in message


def test_entitlement_message_for_unknown_entitlement_is_explicit():
    info = NousPortalAccountInfo(
        logged_in=True,
        source="error",
        fresh=False,
        paid_service_access=None,
        portal_base_url="https://portal.example.test",
        error="account_api_timeout",
    )

    message = format_nous_portal_entitlement_message(info, capability="Tool Gateway")

    assert message is not None
    assert "could not verify" in message
    assert "account_api_timeout" in message
    assert "Run `hermes model`" in message


def test_entitlement_message_for_account_missing():
    info = NousPortalAccountInfo(
        logged_in=True,
        source="account_api",
        fresh=True,
        paid_service_access=False,
        paid_service_access_info=NousPaidServiceAccessInfo(
            allowed=False,
            reason="account_missing",
        ),
    )

    message = format_nous_portal_entitlement_message(info, capability="Tool Gateway")

    assert message is not None
    assert "could not find a Nous Portal account or organisation" in message


# ── org slug/name parsing + top-up URL builder ──────────────────────────────


def test_account_payload_parses_org_slug_and_name(monkeypatch):
    token = _jwt({"sub": "user_123", "org_id": "org_123", "exp": int(time.time()) + 900})
    payload = {
        "user": {"email": "alice@example.test"},
        "organisation": {"id": "org_123", "slug": "acme", "name": "Acme Inc"},
        "paid_service_access": {"allowed": True, "paid_access": True},
    }
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: _state(token))
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda: "fresh-token")
    monkeypatch.setattr("hermes_cli.nous_account._fetch_nous_account_info", lambda *a, **kw: payload)

    info = get_nous_portal_account_info(force_fresh=True)

    assert info.source == "account_api"
    assert info.org_slug == "acme"
    assert info.org_name == "Acme Inc"


def test_account_payload_org_without_slug_leaves_fields_none(monkeypatch):
    # Mirrors current main: organisation: { id } only (slug nullable on the portal).
    token = _jwt({"sub": "user_123", "org_id": "org_123", "exp": int(time.time()) + 900})
    payload = {
        "user": {"email": "alice@example.test"},
        "organisation": {"id": "org_123"},
        "paid_service_access": {"allowed": True, "paid_access": True},
    }
    monkeypatch.setattr("hermes_cli.auth.get_provider_auth_state", lambda provider: _state(token))
    monkeypatch.setattr("hermes_cli.auth.resolve_nous_access_token", lambda: "fresh-token")
    monkeypatch.setattr("hermes_cli.nous_account._fetch_nous_account_info", lambda *a, **kw: payload)

    info = get_nous_portal_account_info(force_fresh=True)

    assert info.org_id == "org_123"
    assert info.org_slug is None
    assert info.org_name is None


def test_topup_url_is_org_pinned_when_slug_present():
    info = NousPortalAccountInfo(
        logged_in=True,
        source="account_api",
        fresh=True,
        portal_base_url="https://portal.example.test",
        org_slug="acme",
    )
    assert (
        nous_portal_topup_url(info)
        == "https://portal.example.test/orgs/acme/billing?topup=open"
    )


def test_topup_url_falls_back_to_legacy_when_slug_null():
    info = NousPortalAccountInfo(
        logged_in=True,
        source="account_api",
        fresh=True,
        portal_base_url="https://portal.example.test",
        org_slug=None,
    )
    url = nous_portal_topup_url(info)
    assert url == "https://portal.example.test/billing?topup=open"
    assert "/orgs/" not in url


def test_topup_url_strips_trailing_slash_and_encodes_slug():
    info = NousPortalAccountInfo(
        logged_in=True,
        source="account_api",
        fresh=True,
        portal_base_url="https://portal.example.test/",
        org_slug="a/b team",
    )
    assert (
        nous_portal_topup_url(info)
        == "https://portal.example.test/orgs/a%2Fb%20team/billing?topup=open"
    )


def test_topup_url_defaults_to_production_portal_for_none():
    url = nous_portal_topup_url(None)
    assert url == "https://portal.nousresearch.com/billing?topup=open"

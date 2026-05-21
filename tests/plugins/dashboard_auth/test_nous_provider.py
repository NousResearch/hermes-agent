"""Tests for the bundled Nous dashboard-auth plugin.

Covers four shapes from Phase 4 of ``.hermes/plans/2026-05-21-dashboard-oauth-auth.md``:

1. Plugin entry-point registration gating (env var checks).
2. ``start_login`` shape (PKCE/state, authorize URL parameters).
3. ``complete_login`` httpx-mocked happy path + error mapping.
4. ``verify_session`` JWT verification — RSA keypair, audience/issuer pinning,
   ``agent_instance_id`` cross-check, ``oauth_contract_version`` tolerance.

Also exercises ``revoke_session`` (no-op) and ``refresh_session``
(unconditional ``RefreshExpiredError``).

All HTTP is mocked: nothing in this file talks to a real Portal.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import urllib.parse
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import plugins.dashboard_auth.nous as nous_plugin
from hermes_cli.dashboard_auth import (
    InvalidCodeError,
    LoginStart,
    ProviderError,
    RefreshExpiredError,
    Session,
    assert_protocol_compliance,
)


# ---------------------------------------------------------------------------
# RSA keypair fixture (module-scope — keygen is slow)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_keypair() -> Dict[str, Any]:
    """Generate an RS256 keypair + matching JWK for verify_session tests."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_numbers = key.public_key().public_numbers()

    def _b64url_uint(n: int) -> str:
        length = (n.bit_length() + 7) // 8
        return (
            base64.urlsafe_b64encode(n.to_bytes(length, "big")).rstrip(b"=").decode()
        )

    jwk = {
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": "test-key-1",
        "n": _b64url_uint(public_numbers.n),
        "e": _b64url_uint(public_numbers.e),
    }
    return {"private_pem": private_pem, "jwk": jwk, "kid": jwk["kid"]}


# ---------------------------------------------------------------------------
# Token-mint helper
# ---------------------------------------------------------------------------


def _mint_token(
    rsa_keypair: Dict[str, Any],
    *,
    iss: str = "https://portal.example.com",
    aud: str = "agent:inst123",
    sub: str = "usr_abc",
    agent_instance_id: str | None = "inst123",
    oauth_contract_version: Any = 1,
    org_id: str | None = "org_xyz",
    scope: str = "agent_dashboard:access",
    ttl_seconds: int = 900,
    extra_claims: Dict[str, Any] | None = None,
) -> str:
    now = int(time.time())
    claims = {
        "iss": iss,
        "aud": aud,
        "sub": sub,
        "iat": now,
        "exp": now + ttl_seconds,
        "scope": scope,
    }
    if agent_instance_id is not None:
        claims["agent_instance_id"] = agent_instance_id
    if oauth_contract_version is not None:
        claims["oauth_contract_version"] = oauth_contract_version
    if org_id is not None:
        claims["org_id"] = org_id
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(
        claims,
        rsa_keypair["private_pem"],
        algorithm="RS256",
        headers={"kid": rsa_keypair["kid"]},
    )


def _patched_jwks(provider: nous_plugin.NousDashboardAuthProvider, rsa_keypair):
    """Patch the provider's JWKS client to return our fixture key."""
    fake_key = MagicMock()
    fake_key.key = serialization.load_pem_private_key(
        rsa_keypair["private_pem"].encode(), password=None
    ).public_key()
    fake_client = MagicMock()
    fake_client.get_signing_key_from_jwt.return_value = fake_key
    provider._jwks_client = fake_client


# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_protocol_compliance(self):
        assert_protocol_compliance(nous_plugin.NousDashboardAuthProvider)

    def test_name_and_display(self):
        p = nous_plugin.NousDashboardAuthProvider(
            client_id="agent:inst1", portal_url="https://portal.example.com"
        )
        assert p.name == "nous"
        assert p.display_name == "Nous Research"

    def test_extracts_agent_instance_id(self):
        p = nous_plugin.NousDashboardAuthProvider(
            client_id="agent:abc-123", portal_url="https://portal.example.com"
        )
        assert p._agent_instance_id == "abc-123"

    def test_strips_trailing_slash_from_portal_url(self):
        p = nous_plugin.NousDashboardAuthProvider(
            client_id="agent:x", portal_url="https://portal.example.com/"
        )
        assert p._portal_url == "https://portal.example.com"

    def test_rejects_malformed_client_id(self):
        with pytest.raises(ValueError, match="agent:"):
            nous_plugin.NousDashboardAuthProvider(
                client_id="hermes-dashboard", portal_url="https://x"
            )


# ---------------------------------------------------------------------------
# Plugin entry point: env-gated registration
# ---------------------------------------------------------------------------


class TestPluginRegister:
    def test_skips_when_client_id_missing(self, monkeypatch):
        monkeypatch.delenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", raising=False)
        monkeypatch.delenv("HERMES_DASHBOARD_PORTAL_URL", raising=False)
        ctx = MagicMock()
        nous_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        # Skip reason is surfaced for the gate's fail-closed message.
        assert "HERMES_DASHBOARD_OAUTH_CLIENT_ID" in nous_plugin.LAST_SKIP_REASON

    def test_registers_with_default_portal_url_when_only_client_id_set(
        self, monkeypatch
    ):
        """Phase 7 follow-up: HERMES_DASHBOARD_PORTAL_URL is optional —
        defaults to the production Nous Portal. The user shouldn't have
        to set it for the common production deployment path."""
        monkeypatch.setenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", "agent:inst1")
        monkeypatch.delenv("HERMES_DASHBOARD_PORTAL_URL", raising=False)
        ctx = MagicMock()
        nous_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert isinstance(registered, nous_plugin.NousDashboardAuthProvider)
        assert registered._portal_url == "https://portal.nousresearch.com"
        # Skip reason cleared on successful registration.
        assert nous_plugin.LAST_SKIP_REASON == ""

    def test_skips_when_client_id_malformed(self, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", "hermes-dashboard")
        monkeypatch.setenv("HERMES_DASHBOARD_PORTAL_URL", "https://p.example")
        ctx = MagicMock()
        nous_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        # Skip reason names the offending value + contract shape.
        assert "agent:" in nous_plugin.LAST_SKIP_REASON
        assert "hermes-dashboard" in nous_plugin.LAST_SKIP_REASON

    def test_registers_with_explicit_portal_url(self, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", "agent:inst1")
        monkeypatch.setenv("HERMES_DASHBOARD_PORTAL_URL", "https://p.example")
        ctx = MagicMock()
        nous_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._client_id == "agent:inst1"
        assert registered._portal_url == "https://p.example"

    def test_strips_whitespace_from_env_vars(self, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", "  agent:x  ")
        monkeypatch.setenv("HERMES_DASHBOARD_PORTAL_URL", "  https://p.example  ")
        ctx = MagicMock()
        nous_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()

    def test_empty_portal_url_env_uses_default(self, monkeypatch):
        """Explicit empty string still falls back to the production
        default — same handling as 'unset' so an empty Fly secret can't
        accidentally point the dashboard at nowhere."""
        monkeypatch.setenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", "agent:inst1")
        monkeypatch.setenv("HERMES_DASHBOARD_PORTAL_URL", "")
        ctx = MagicMock()
        nous_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._portal_url == "https://portal.nousresearch.com"


# ---------------------------------------------------------------------------
# start_login
# ---------------------------------------------------------------------------


class TestStartLogin:
    @pytest.fixture
    def provider(self):
        return nous_plugin.NousDashboardAuthProvider(
            client_id="agent:inst1", portal_url="https://portal.example.com"
        )

    def test_returns_login_start(self, provider):
        result = provider.start_login(
            redirect_uri="https://hermes.fly.dev/auth/callback"
        )
        assert isinstance(result, LoginStart)

    def test_redirect_url_targets_portal_authorize(self, provider):
        result = provider.start_login(
            redirect_uri="https://hermes.fly.dev/auth/callback"
        )
        assert result.redirect_url.startswith(
            "https://portal.example.com/oauth/authorize?"
        )

    def test_authorize_url_has_required_params(self, provider):
        result = provider.start_login(
            redirect_uri="https://hermes.fly.dev/auth/callback"
        )
        parsed = urllib.parse.urlparse(result.redirect_url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        assert params["response_type"] == "code"
        assert params["client_id"] == "agent:inst1"
        assert params["redirect_uri"] == "https://hermes.fly.dev/auth/callback"
        assert params["scope"] == "agent_dashboard:access"
        assert params["code_challenge_method"] == "S256"
        assert "state" in params
        assert "code_challenge" in params

    def test_code_verifier_in_cookie_payload_43_to_128_chars(self, provider):
        result = provider.start_login(
            redirect_uri="https://hermes.fly.dev/auth/callback"
        )
        assert "hermes_session_pkce" in result.cookie_payload
        pkce = result.cookie_payload["hermes_session_pkce"]
        # Shape: ``state=…;verifier=…`` (matches stub-provider convention so
        # the auth-route layer's parser works uniformly across providers).
        parts = dict(seg.split("=", 1) for seg in pkce.split(";") if "=" in seg)
        verifier = parts["verifier"]
        # RFC 7636 §4.1
        assert 43 <= len(verifier) <= 128

    def test_state_in_cookie_payload_matches_url_param(self, provider):
        result = provider.start_login(
            redirect_uri="https://hermes.fly.dev/auth/callback"
        )
        parsed = urllib.parse.urlparse(result.redirect_url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        pkce = result.cookie_payload["hermes_session_pkce"]
        parts = dict(seg.split("=", 1) for seg in pkce.split(";") if "=" in seg)
        assert parts["state"] == params["state"]

    def test_code_challenge_is_s256_of_verifier(self, provider):
        result = provider.start_login(
            redirect_uri="https://hermes.fly.dev/auth/callback"
        )
        parsed = urllib.parse.urlparse(result.redirect_url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        pkce = result.cookie_payload["hermes_session_pkce"]
        parts = dict(seg.split("=", 1) for seg in pkce.split(";") if "=" in seg)
        verifier = parts["verifier"]
        expected_challenge = (
            base64.urlsafe_b64encode(
                hashlib.sha256(verifier.encode("ascii")).digest()
            )
            .rstrip(b"=")
            .decode()
        )
        assert params["code_challenge"] == expected_challenge

    def test_two_calls_produce_different_state_and_verifier(self, provider):
        a = provider.start_login(
            redirect_uri="https://hermes.fly.dev/auth/callback"
        )
        b = provider.start_login(
            redirect_uri="https://hermes.fly.dev/auth/callback"
        )
        assert a.cookie_payload["hermes_session_pkce"] != b.cookie_payload[
            "hermes_session_pkce"
        ]

    def test_rejects_non_http_scheme(self, provider):
        with pytest.raises(ProviderError, match="http"):
            provider.start_login(redirect_uri="ftp://x/auth/callback")

    def test_rejects_http_with_non_localhost(self, provider):
        with pytest.raises(ProviderError, match="localhost"):
            provider.start_login(
                redirect_uri="http://hermes.fly.dev/auth/callback"
            )

    def test_allows_http_localhost(self, provider):
        # Should not raise.
        provider.start_login(redirect_uri="http://localhost:8080/auth/callback")
        provider.start_login(redirect_uri="http://127.0.0.1:8080/auth/callback")

    def test_rejects_wrong_callback_path(self, provider):
        with pytest.raises(ProviderError, match="/auth/callback"):
            provider.start_login(redirect_uri="https://x.example/oauth/cb")


# ---------------------------------------------------------------------------
# complete_login (httpx mocked)
# ---------------------------------------------------------------------------


class TestCompleteLogin:
    @pytest.fixture
    def provider(self, rsa_keypair):
        p = nous_plugin.NousDashboardAuthProvider(
            client_id="agent:inst123", portal_url="https://portal.example.com"
        )
        _patched_jwks(p, rsa_keypair)
        return p

    def _mock_post(self, status_code: int, body: Any, *, ctype: str = "application/json"):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        if isinstance(body, dict):
            resp.text = json.dumps(body)
            resp.json = MagicMock(return_value=body)
        else:
            resp.text = body
            # _parse_json_body bails on non-application/json before .json()
            # is called, but be safe for callers that pass a non-dict body
            # with ctype=application/json.
            resp.json = MagicMock(side_effect=ValueError("not json"))
        resp.headers = {"content-type": ctype}
        return resp

    def test_happy_path_returns_session(self, provider, rsa_keypair):
        access_token = _mint_token(rsa_keypair)
        mock_resp = self._mock_post(
            200, {"access_token": access_token, "token_type": "Bearer"}
        )
        with patch("plugins.dashboard_auth.nous.httpx.post", return_value=mock_resp):
            session = provider.complete_login(
                code="abc",
                state="state-val",
                code_verifier="vfy",
                redirect_uri="https://hermes.fly.dev/auth/callback",
            )
        assert isinstance(session, Session)
        assert session.user_id == "usr_abc"
        assert session.provider == "nous"
        assert session.access_token == access_token
        assert session.refresh_token == ""  # contract V1
        assert session.org_id == "org_xyz"
        assert session.email == ""
        assert session.display_name == ""

    def test_400_raises_invalid_code(self, provider):
        mock_resp = self._mock_post(400, {"error": "invalid_grant"})
        with patch("plugins.dashboard_auth.nous.httpx.post", return_value=mock_resp):
            with pytest.raises(InvalidCodeError, match="invalid_grant"):
                provider.complete_login(
                    code="bad", state="s", code_verifier="v",
                    redirect_uri="https://hermes.fly.dev/auth/callback",
                )

    def test_500_raises_provider_error(self, provider):
        mock_resp = self._mock_post(500, "internal server error", ctype="text/plain")
        mock_resp.text = "internal server error"
        with patch("plugins.dashboard_auth.nous.httpx.post", return_value=mock_resp):
            with pytest.raises(ProviderError, match="500"):
                provider.complete_login(
                    code="x", state="s", code_verifier="v",
                    redirect_uri="https://hermes.fly.dev/auth/callback",
                )

    def test_missing_access_token_raises(self, provider):
        mock_resp = self._mock_post(200, {"token_type": "Bearer"})
        with patch("plugins.dashboard_auth.nous.httpx.post", return_value=mock_resp):
            with pytest.raises(ProviderError, match="access_token"):
                provider.complete_login(
                    code="x", state="s", code_verifier="v",
                    redirect_uri="https://hermes.fly.dev/auth/callback",
                )

    def test_unexpected_token_type_raises(self, provider, rsa_keypair):
        access_token = _mint_token(rsa_keypair)
        mock_resp = self._mock_post(
            200, {"access_token": access_token, "token_type": "DPoP"}
        )
        with patch("plugins.dashboard_auth.nous.httpx.post", return_value=mock_resp):
            with pytest.raises(ProviderError, match="token_type"):
                provider.complete_login(
                    code="x", state="s", code_verifier="v",
                    redirect_uri="https://hermes.fly.dev/auth/callback",
                )

    def test_network_error_raises_provider_error(self, provider):
        with patch(
            "plugins.dashboard_auth.nous.httpx.post",
            side_effect=httpx.ConnectError("conn refused"),
        ):
            with pytest.raises(ProviderError, match="unreachable"):
                provider.complete_login(
                    code="x", state="s", code_verifier="v",
                    redirect_uri="https://hermes.fly.dev/auth/callback",
                )

    def test_captures_refresh_token_if_present_forward_compat(
        self, provider, rsa_keypair
    ):
        """Forward-compat: contract V1 doesn't issue, but if a future Portal
        does, we should preserve it in the Session for later use."""
        access_token = _mint_token(rsa_keypair)
        mock_resp = self._mock_post(
            200,
            {
                "access_token": access_token,
                "token_type": "Bearer",
                "refresh_token": "rt-opaque",
            },
        )
        with patch("plugins.dashboard_auth.nous.httpx.post", return_value=mock_resp):
            session = provider.complete_login(
                code="x", state="s", code_verifier="v",
                redirect_uri="https://hermes.fly.dev/auth/callback",
            )
        assert session.refresh_token == "rt-opaque"


# ---------------------------------------------------------------------------
# verify_session
# ---------------------------------------------------------------------------


class TestVerifySession:
    @pytest.fixture
    def provider(self, rsa_keypair):
        p = nous_plugin.NousDashboardAuthProvider(
            client_id="agent:inst123", portal_url="https://portal.example.com"
        )
        _patched_jwks(p, rsa_keypair)
        return p

    def test_happy_path_returns_session(self, provider, rsa_keypair):
        token = _mint_token(rsa_keypair)
        session = provider.verify_session(access_token=token)
        assert session is not None
        assert session.user_id == "usr_abc"
        assert session.org_id == "org_xyz"

    def test_expired_token_returns_none(self, provider, rsa_keypair):
        token = _mint_token(rsa_keypair, ttl_seconds=-1)
        assert provider.verify_session(access_token=token) is None

    def test_wrong_audience_raises_provider_error(self, provider, rsa_keypair):
        token = _mint_token(rsa_keypair, aud="agent:other-instance")
        with pytest.raises(ProviderError, match="verification failed"):
            provider.verify_session(access_token=token)

    def test_wrong_issuer_raises_provider_error(self, provider, rsa_keypair):
        token = _mint_token(rsa_keypair, iss="https://evil.example")
        with pytest.raises(ProviderError, match="verification failed"):
            provider.verify_session(access_token=token)

    def test_missing_sub_raises(self, provider, rsa_keypair):
        # PyJWT's "require" set includes sub, so this surfaces as
        # InvalidTokenError → ProviderError before we ever touch _session_from_claims.
        token = _mint_token(rsa_keypair, sub="")
        # Empty sub still encodes successfully; PyJWT's require check only
        # asserts presence. Our own _session_from_claims rejects empty.
        with pytest.raises(ProviderError, match="sub"):
            provider.verify_session(access_token=token)

    def test_agent_instance_id_mismatch_rejected(self, provider, rsa_keypair):
        token = _mint_token(rsa_keypair, agent_instance_id="some-other-id")
        with pytest.raises(ProviderError, match="agent_instance_id mismatch"):
            provider.verify_session(access_token=token)

    def test_agent_instance_id_missing_is_tolerated(self, provider, rsa_keypair):
        token = _mint_token(rsa_keypair, agent_instance_id=None)
        session = provider.verify_session(access_token=token)
        assert session is not None

    def test_contract_version_missing_warns_but_succeeds(
        self, provider, rsa_keypair, caplog
    ):
        import logging
        token = _mint_token(rsa_keypair, oauth_contract_version=None)
        with caplog.at_level(logging.WARNING, logger="plugins.dashboard_auth.nous"):
            session = provider.verify_session(access_token=token)
        assert session is not None
        assert any(
            "oauth_contract_version" in r.message for r in caplog.records
        )

    def test_contract_version_mismatch_rejected(self, provider, rsa_keypair):
        token = _mint_token(rsa_keypair, oauth_contract_version=2)
        with pytest.raises(ProviderError, match="oauth_contract_version"):
            provider.verify_session(access_token=token)

    def test_jwks_unreachable_raises_provider_error(self, provider, rsa_keypair):
        token = _mint_token(rsa_keypair)
        # Replace the patched client so it raises.
        bad_client = MagicMock()
        bad_client.get_signing_key_from_jwt.side_effect = jwt.PyJWKClientError(
            "fetch failed"
        )
        provider._jwks_client = bad_client
        with pytest.raises(ProviderError, match="JWKS"):
            provider.verify_session(access_token=token)


# ---------------------------------------------------------------------------
# refresh_session + revoke_session (V1 contract: trivial)
# ---------------------------------------------------------------------------


class TestRefreshAndRevoke:
    @pytest.fixture
    def provider(self):
        return nous_plugin.NousDashboardAuthProvider(
            client_id="agent:inst1", portal_url="https://portal.example.com"
        )

    def test_refresh_always_raises(self, provider):
        with pytest.raises(RefreshExpiredError):
            provider.refresh_session(refresh_token="anything")

    def test_refresh_raises_even_with_empty_token(self, provider):
        with pytest.raises(RefreshExpiredError):
            provider.refresh_session(refresh_token="")

    def test_revoke_is_noop(self, provider):
        # Must not raise; returns None implicitly.
        assert provider.revoke_session(refresh_token="anything") is None
        assert provider.revoke_session(refresh_token="") is None

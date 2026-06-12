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
# Plugin entry point: config.yaml + env-override precedence
# ---------------------------------------------------------------------------


class TestConfigYamlSource:
    """``dashboard.oauth.{client_id,portal_url}`` in ``config.yaml`` is the
    canonical surface for these settings. ``HERMES_DASHBOARD_OAUTH_CLIENT_ID``
    and ``HERMES_DASHBOARD_PORTAL_URL`` are operator overrides that win when
    set — this is the contract Fly.io's platform-secret injection relies on,
    and the contract that lets local devs experiment without setting env
    vars.

    Each test pins exactly one tier of the precedence chain so a regression
    that flips the order is caught:

        env (when truthy) > config.yaml (when truthy) > plugin default
    """

    @pytest.fixture
    def patch_config(self, monkeypatch):
        """Yield a callable that replaces ``hermes_cli.config.load_config``
        with a stub returning the given dict. Tests pass the intended
        ``dashboard.oauth`` block; the stub returns the wrapping structure."""

        def _set(oauth_block: Dict[str, Any] | None) -> None:
            cfg = {}
            if oauth_block is not None:
                cfg = {"dashboard": {"oauth": oauth_block}}
            monkeypatch.setattr(
                "hermes_cli.config.load_config", lambda: cfg
            )

        return _set

    def test_config_yaml_only_client_id_registers(self, patch_config, monkeypatch):
        """No env var, only config.yaml — plugin reads from config and
        registers successfully. This is the path Teknium's review pushed
        for (".env is for secrets only")."""
        monkeypatch.delenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", raising=False)
        monkeypatch.delenv("HERMES_DASHBOARD_PORTAL_URL", raising=False)
        patch_config({"client_id": "agent:from-config"})
        ctx = MagicMock()
        nous_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._client_id == "agent:from-config"
        # Defaults to production portal URL when neither config nor env
        # specifies one.
        assert registered._portal_url == "https://portal.nousresearch.com"

    def test_config_yaml_client_id_and_portal_url(self, patch_config, monkeypatch):
        monkeypatch.delenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", raising=False)
        monkeypatch.delenv("HERMES_DASHBOARD_PORTAL_URL", raising=False)
        patch_config({
            "client_id": "agent:from-config",
            "portal_url": "https://staging.portal.example",
        })
        ctx = MagicMock()
        nous_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._client_id == "agent:from-config"
        assert registered._portal_url == "https://staging.portal.example"

    def test_env_overrides_config_client_id(self, patch_config, monkeypatch):
        """Env wins. Critical for Fly.io: the Portal injects
        HERMES_DASHBOARD_OAUTH_CLIENT_ID at deploy time and we MUST
        honour it even if a stale config.yaml ships in the image."""
        monkeypatch.setenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", "agent:from-env")
        patch_config({"client_id": "agent:from-config"})
        ctx = MagicMock()
        nous_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._client_id == "agent:from-env", (
            "env var must override config.yaml — Fly secret injection "
            "depends on this precedence"
        )

    def test_env_overrides_config_portal_url(self, patch_config, monkeypatch):
        monkeypatch.setenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", "agent:x")
        monkeypatch.setenv(
            "HERMES_DASHBOARD_PORTAL_URL", "https://env.portal.example",
        )
        patch_config({
            "client_id": "agent:x",
            "portal_url": "https://config.portal.example",
        })
        ctx = MagicMock()
        nous_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._portal_url == "https://env.portal.example"

    def test_empty_env_string_does_not_shadow_config(
        self, patch_config, monkeypatch
    ):
        """``HERMES_DASHBOARD_OAUTH_CLIENT_ID=`` (set but empty) is
        common in CI/Fly when a secret is provisioned-but-not-populated.
        It MUST NOT shadow a valid config.yaml value with an empty
        string — operators would lose the gate."""
        monkeypatch.setenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", "")
        patch_config({"client_id": "agent:from-config"})
        ctx = MagicMock()
        nous_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._client_id == "agent:from-config"

    def test_neither_source_skips_with_helpful_reason(
        self, patch_config, monkeypatch
    ):
        """Neither env nor config.yaml set — skip with a reason that
        mentions BOTH surfaces so operators don't guess wrong about
        which one to populate."""
        monkeypatch.delenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", raising=False)
        patch_config(None)
        ctx = MagicMock()
        nous_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        # Old behaviour: skip reason mentions the env var.
        assert "HERMES_DASHBOARD_OAUTH_CLIENT_ID" in nous_plugin.LAST_SKIP_REASON
        # New behaviour: skip reason ALSO mentions the config.yaml path
        # so the user knows it's a valid alternative.
        assert "dashboard.oauth.client_id" in nous_plugin.LAST_SKIP_REASON, (
            f"skip reason omits the config.yaml surface — operators "
            f"won't know it exists. got: {nous_plugin.LAST_SKIP_REASON!r}"
        )

    def test_config_yaml_load_failure_falls_through_cleanly(
        self, monkeypatch
    ):
        """If load_config() raises (e.g. malformed YAML, IOError), the
        plugin must not crash — it falls through to the env-only path
        and either succeeds (if env is set) or surfaces the standard
        'not set' skip reason."""
        monkeypatch.delenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", raising=False)

        def _broken_load():
            raise OSError("config.yaml not readable")

        monkeypatch.setattr(
            "hermes_cli.config.load_config", _broken_load
        )
        ctx = MagicMock()
        # Must not raise.
        nous_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()

    def test_config_yaml_with_non_dict_oauth_section(
        self, monkeypatch
    ):
        """cfg_get handles 'config has a string where a section was
        expected' robustly. Verify the plugin inherits that resilience
        so a malformed user config doesn't crash startup."""
        monkeypatch.delenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"dashboard": {"oauth": "wrong type"}},
        )
        ctx = MagicMock()
        nous_plugin.register(ctx)
        # Falls through to the no-env-and-no-config path.
        ctx.register_dashboard_auth_provider.assert_not_called()


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

    def test_allows_http_with_arbitrary_host(self, provider):
        # http:// is permitted for any host now, not just localhost — the
        # Portal-side check is authoritative on which redirect_uris are
        # accepted; this client-side fast-fail must not reject self-hosted
        # dashboards reached over plain HTTP (LAN IPs, internal hostnames,
        # TLS-terminating reverse proxies). Should not raise.
        provider.start_login(redirect_uri="http://hermes.fly.dev/auth/callback")
        provider.start_login(redirect_uri="http://192.168.1.50:8080/auth/callback")
        provider.start_login(redirect_uri="http://my-internal-host/auth/callback")

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
            200,
            {
                "access_token": access_token,
                "token_type": "Bearer",
                "refresh_token": "rt_initial_value",
            },
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
        # The dashboard auth-code grant now issues a refresh token (NAS #293);
        # complete_login must surface it so the middleware persists it.
        assert session.refresh_token == "rt_initial_value"
        assert session.org_id == "org_xyz"
        assert session.email == ""
        assert session.display_name == ""

    def test_happy_path_tolerates_missing_refresh_token(self, provider, rsa_keypair):
        # If Portal omits refresh_token (older deploy), the session is still
        # valid as access-token-only; refresh_token defaults to "".
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
        assert session.refresh_token == ""

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

    def test_verification_failure_message_surfaces_token_claims(
        self, provider, rsa_keypair
    ):
        """Operators need to see the actual iss/aud the token carries to debug
        config drift between HERMES_DASHBOARD_PORTAL_URL/CLIENT_ID and Portal."""
        token = _mint_token(rsa_keypair, iss="https://evil.example")
        with pytest.raises(ProviderError) as excinfo:
            provider.verify_session(access_token=token)
        msg = str(excinfo.value)
        # Both the observed (token) and expected (configured) values appear.
        assert "'https://evil.example'" in msg
        assert "'https://portal.example.com'" in msg  # configured portal URL

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
# refresh_session + revoke_session
# ---------------------------------------------------------------------------


class TestRefreshAndRevoke:
    @pytest.fixture
    def provider(self, rsa_keypair):
        p = nous_plugin.NousDashboardAuthProvider(
            client_id="agent:inst123", portal_url="https://portal.example.com"
        )
        _patched_jwks(p, rsa_keypair)
        return p

    def _mock_post(self, status_code, body, *, ctype="application/json"):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        if isinstance(body, dict):
            resp.text = json.dumps(body)
            resp.json = MagicMock(return_value=body)
        else:
            resp.text = body
            resp.json = MagicMock(side_effect=ValueError("not json"))
        resp.headers = {"content-type": ctype}
        return resp

    def test_refresh_happy_path_returns_rotated_session(self, provider, rsa_keypair):
        # Portal returns a fresh access token AND a rotated refresh token.
        access_token = _mint_token(rsa_keypair)
        mock_resp = self._mock_post(
            200,
            {
                "access_token": access_token,
                "token_type": "Bearer",
                "refresh_token": "rt_rotated_value",
            },
        )
        with patch(
            "plugins.dashboard_auth.nous.httpx.post", return_value=mock_resp
        ) as mock_post:
            session = provider.refresh_session(refresh_token="rt_old_value")

        assert isinstance(session, Session)
        assert session.access_token == access_token
        # The ROTATED refresh token must be surfaced so the middleware can
        # persist it back to the cookie.
        assert session.refresh_token == "rt_rotated_value"
        assert session.provider == "nous"

        # Posts grant_type=refresh_token with the RT in BOTH the body (Portal's
        # schema requires it there) and the X-Refresh-Token header (log
        # redaction). Verified against the live preview deploy.
        _, kwargs = mock_post.call_args
        assert kwargs["data"]["grant_type"] == "refresh_token"
        assert kwargs["data"]["client_id"] == "agent:inst123"
        assert kwargs["data"]["refresh_token"] == "rt_old_value"
        assert kwargs["headers"]["x-nous-refresh-token"] == "rt_old_value"

    def test_refresh_400_raises_refresh_expired(self, provider):
        # Expired / revoked / reuse-detected RT → Portal 400 → force re-login.
        mock_resp = self._mock_post(400, {"error": "invalid_grant"})
        with patch("plugins.dashboard_auth.nous.httpx.post", return_value=mock_resp):
            with pytest.raises(RefreshExpiredError, match="invalid_grant"):
                provider.refresh_session(refresh_token="rt_dead")

    def test_refresh_empty_token_raises_refresh_expired_without_network(self, provider):
        # No RT present — fail fast as a dead session, never hit the network.
        with patch("plugins.dashboard_auth.nous.httpx.post") as mock_post:
            with pytest.raises(RefreshExpiredError):
                provider.refresh_session(refresh_token="")
        mock_post.assert_not_called()

    def test_refresh_network_error_raises_provider_error(self, provider):
        with patch(
            "plugins.dashboard_auth.nous.httpx.post",
            side_effect=httpx.RequestError("boom"),
        ):
            with pytest.raises(ProviderError, match="unreachable"):
                provider.refresh_session(refresh_token="rt_x")

    def test_refresh_500_raises_provider_error(self, provider):
        mock_resp = self._mock_post(500, "oops", ctype="text/plain")
        with patch("plugins.dashboard_auth.nous.httpx.post", return_value=mock_resp):
            with pytest.raises(ProviderError):
                provider.refresh_session(refresh_token="rt_x")

    def test_revoke_is_noop(self, provider):
        # Must not raise; returns None implicitly.
        assert provider.revoke_session(refresh_token="anything") is None
        assert provider.revoke_session(refresh_token="") is None

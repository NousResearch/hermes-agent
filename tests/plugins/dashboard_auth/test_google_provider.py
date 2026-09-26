"""Tests for the bundled Google OAuth dashboard-auth plugin.

Covers, by analogy with ``test_self_hosted_provider.py`` and
``test_nous_provider.py``:

1. Plugin entry-point registration gating (env + config.yaml precedence).
2. ``start_login`` shape (PKCE/state, authorize URL parameters, including
   the Google-specific ``prompt=select_account``).
3. ``complete_login`` httpx-mocked happy path + error mapping (ID-token
   grant), including the ``email_verified`` guard and the
   ``DASHBOARD_ALLOWED_USERS`` allowlist.
4. ``verify_session`` — local ID-token verification (no userinfo round
   trip), audience/issuer pinning, the ``hd`` claim → ``org_id`` mapping.
5. ``refresh_session`` rotation + error mapping, ``revoke_session``
   (best-effort POST to Google's fixed revocation endpoint).
6. OIDC discovery: endpoint extraction, soft-TTL caching, https enforcement.

All HTTP is mocked: nothing here talks to real Google endpoints.
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

import plugins.dashboard_auth.google as google_plugin
from hermes_cli.dashboard_auth import (
    InvalidCodeError,
    LoginStart,
    ProviderError,
    RefreshExpiredError,
    Session,
    assert_protocol_compliance,
)

_ISSUER = "https://accounts.google.com"
_CLIENT_ID = "abc123.apps.googleusercontent.com"
_CLIENT_SECRET = "GOCSPX-test-secret"

_DISCOVERY_DOC = {
    "issuer": _ISSUER,
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint": "https://oauth2.googleapis.com/token",
    "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
}


# ---------------------------------------------------------------------------
# RSA keypair fixture (module-scope — keygen is slow)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_keypair() -> Dict[str, Any]:
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
# Token-mint helper — Google ID-token claim shape
# ---------------------------------------------------------------------------


def _mint_id_token(
    rsa_keypair: Dict[str, Any],
    *,
    iss: str = _ISSUER,
    aud: str = _CLIENT_ID,
    sub: str = "110169484474386276334",
    email: str | None = "alice@example.com",
    email_verified: bool | None = True,
    name: str | None = "Alice Example",
    hd: str | None = None,
    ttl_seconds: int = 3600,
    extra_claims: Dict[str, Any] | None = None,
) -> str:
    now = int(time.time())
    claims: Dict[str, Any] = {
        "iss": iss,
        "aud": aud,
        "sub": sub,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    if email is not None:
        claims["email"] = email
    if email_verified is not None:
        claims["email_verified"] = email_verified
    if name is not None:
        claims["name"] = name
    if hd is not None:
        claims["hd"] = hd
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(
        claims,
        rsa_keypair["private_pem"],
        algorithm="RS256",
        headers={"kid": rsa_keypair["kid"]},
    )


def _make_provider(
    rsa_keypair,
    *,
    client_id: str = _CLIENT_ID,
    client_secret: str = _CLIENT_SECRET,
) -> google_plugin.GoogleDashboardAuthProvider:
    """Construct a provider with discovery + JWKS stubbed (no network)."""
    p = google_plugin.GoogleDashboardAuthProvider(
        client_id=client_id, client_secret=client_secret
    )
    p._discovery = dict(_DISCOVERY_DOC)
    p._discovery_fetched_at = time.time()
    fake_key = MagicMock()
    fake_key.key = serialization.load_pem_private_key(
        rsa_keypair["private_pem"].encode(), password=None
    ).public_key()
    fake_client = MagicMock()
    fake_client.get_signing_key_from_jwt.return_value = fake_key
    p._jwks_client = fake_client
    return p


def _mock_post(status_code: int, body: Any, *, ctype: str = "application/json"):
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


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_protocol_compliance(self):
        assert_protocol_compliance(google_plugin.GoogleDashboardAuthProvider)

    def test_name_and_display(self):
        p = google_plugin.GoogleDashboardAuthProvider(
            client_id=_CLIENT_ID, client_secret=_CLIENT_SECRET
        )
        assert p.name == "google"
        assert p.display_name == "Google"

    def test_requires_client_id(self):
        with pytest.raises(ValueError, match="client_id"):
            google_plugin.GoogleDashboardAuthProvider(
                client_id="", client_secret=_CLIENT_SECRET
            )

    def test_requires_client_secret(self):
        with pytest.raises(ValueError, match="client_secret"):
            google_plugin.GoogleDashboardAuthProvider(
                client_id=_CLIENT_ID, client_secret=""
            )


# ---------------------------------------------------------------------------
# OIDC discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def _mock_get(self, status_code, body, *, ctype="application/json"):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.json = MagicMock(return_value=body)
        resp.text = json.dumps(body) if isinstance(body, dict) else str(body)
        resp.headers = {"content-type": ctype}
        return resp

    def test_fetches_and_caches(self):
        p = google_plugin.GoogleDashboardAuthProvider(
            client_id=_CLIENT_ID, client_secret=_CLIENT_SECRET
        )
        mock_resp = self._mock_get(200, dict(_DISCOVERY_DOC))
        with patch(
            "plugins.dashboard_auth.google.httpx.get", return_value=mock_resp
        ) as mock_get:
            disco1 = p._get_discovery()
            disco2 = p._get_discovery()
        assert disco1["token_endpoint"] == _DISCOVERY_DOC["token_endpoint"]
        assert disco1["authorization_endpoint"] == _DISCOVERY_DOC["authorization_endpoint"]
        assert disco1["jwks_uri"] == _DISCOVERY_DOC["jwks_uri"]
        # Cached — only one network call.
        assert mock_get.call_count == 1
        assert disco2 is disco1

    def test_stale_cache_is_refetched(self):
        p = google_plugin.GoogleDashboardAuthProvider(
            client_id=_CLIENT_ID, client_secret=_CLIENT_SECRET
        )
        p._discovery = dict(_DISCOVERY_DOC)
        p._discovery_fetched_at = time.time() - google_plugin._DISCOVERY_CACHE_TTL_SEC - 1
        mock_resp = self._mock_get(200, dict(_DISCOVERY_DOC))
        with patch(
            "plugins.dashboard_auth.google.httpx.get", return_value=mock_resp
        ) as mock_get:
            p._get_discovery()
        assert mock_get.call_count == 1

    def test_missing_field_raises(self):
        p = google_plugin.GoogleDashboardAuthProvider(
            client_id=_CLIENT_ID, client_secret=_CLIENT_SECRET
        )
        incomplete = dict(_DISCOVERY_DOC)
        del incomplete["jwks_uri"]
        mock_resp = self._mock_get(200, incomplete)
        with patch(
            "plugins.dashboard_auth.google.httpx.get", return_value=mock_resp
        ):
            with pytest.raises(ProviderError, match="jwks_uri"):
                p._get_discovery()

    def test_non_https_endpoint_rejected(self):
        p = google_plugin.GoogleDashboardAuthProvider(
            client_id=_CLIENT_ID, client_secret=_CLIENT_SECRET
        )
        bad = dict(_DISCOVERY_DOC)
        bad["token_endpoint"] = "http://oauth2.googleapis.com/token"
        mock_resp = self._mock_get(200, bad)
        with patch(
            "plugins.dashboard_auth.google.httpx.get", return_value=mock_resp
        ):
            with pytest.raises(ProviderError, match="https"):
                p._get_discovery()

    def test_discovery_unreachable_raises_provider_error(self):
        p = google_plugin.GoogleDashboardAuthProvider(
            client_id=_CLIENT_ID, client_secret=_CLIENT_SECRET
        )
        with patch(
            "plugins.dashboard_auth.google.httpx.get",
            side_effect=httpx.ConnectError("conn refused"),
        ):
            with pytest.raises(ProviderError, match="unreachable"):
                p._get_discovery()


# ---------------------------------------------------------------------------
# start_login
# ---------------------------------------------------------------------------


class TestStartLogin:
    @pytest.fixture
    def provider(self, rsa_keypair):
        return _make_provider(rsa_keypair)

    def test_returns_login_start(self, provider):
        result = provider.start_login(redirect_uri="https://dash.example.com/auth/callback")
        assert isinstance(result, LoginStart)

    def test_authorize_url_has_required_params(self, provider):
        result = provider.start_login(redirect_uri="https://dash.example.com/auth/callback")
        parsed = urllib.parse.urlparse(result.redirect_url)
        assert result.redirect_url.startswith(_DISCOVERY_DOC["authorization_endpoint"])
        params = dict(urllib.parse.parse_qsl(parsed.query))
        assert params["response_type"] == "code"
        assert params["client_id"] == _CLIENT_ID
        assert params["redirect_uri"] == "https://dash.example.com/auth/callback"
        assert params["scope"] == "openid email profile"
        assert params["code_challenge_method"] == "S256"
        assert params["prompt"] == "select_account"
        assert params["access_type"] == "offline"
        assert "state" in params
        assert "code_challenge" in params

    def test_code_verifier_in_cookie_payload_43_to_128_chars(self, provider):
        result = provider.start_login(redirect_uri="https://dash.example.com/auth/callback")
        pkce = result.cookie_payload["hermes_session_pkce"]
        parts = dict(seg.split("=", 1) for seg in pkce.split(";") if "=" in seg)
        verifier = parts["verifier"]
        assert 43 <= len(verifier) <= 128  # RFC 7636 §4.1

    def test_state_in_cookie_matches_url(self, provider):
        result = provider.start_login(redirect_uri="https://dash.example.com/auth/callback")
        parsed = urllib.parse.urlparse(result.redirect_url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        pkce = result.cookie_payload["hermes_session_pkce"]
        parts = dict(seg.split("=", 1) for seg in pkce.split(";") if "=" in seg)
        assert parts["state"] == params["state"]

    def test_rejects_redirect_uri_wrong_path(self, provider):
        with pytest.raises(ProviderError, match="auth/callback"):
            provider.start_login(redirect_uri="https://dash.example.com/wrong")


# ---------------------------------------------------------------------------
# complete_login
# ---------------------------------------------------------------------------


class TestCompleteLogin:
    @pytest.fixture
    def provider(self, rsa_keypair):
        return _make_provider(rsa_keypair)

    def test_happy_path_returns_session(self, provider, rsa_keypair):
        id_token = _mint_id_token(rsa_keypair, hd="example.com")
        mock_resp = _mock_post(
            200,
            {"access_token": "ya29.opaque", "token_type": "Bearer", "id_token": id_token},
        )
        with patch("plugins.dashboard_auth.google.httpx.post", return_value=mock_resp):
            session = provider.complete_login(
                code="abc",
                state="s",
                code_verifier="v",
                redirect_uri="https://dash.example.com/auth/callback",
            )
        assert isinstance(session, Session)
        assert session.provider == "google"
        assert session.user_id == "110169484474386276334"
        assert session.email == "alice@example.com"
        assert session.display_name == "Alice Example"
        # 'hd' claim maps to org_id.
        assert session.org_id == "example.com"
        # The verified ID token is stored as the session's access token so
        # verify_session() can re-verify it locally.
        assert session.access_token == id_token

    def test_missing_id_token_raises(self, provider):
        mock_resp = _mock_post(200, {"access_token": "ya29.opaque", "token_type": "Bearer"})
        with patch("plugins.dashboard_auth.google.httpx.post", return_value=mock_resp):
            with pytest.raises(ProviderError, match="id_token"):
                provider.complete_login(
                    code="abc", state="s", code_verifier="v",
                    redirect_uri="https://dash.example.com/auth/callback",
                )

    def test_400_raises_invalid_code(self, provider):
        mock_resp = _mock_post(400, {"error": "invalid_grant"})
        with patch("plugins.dashboard_auth.google.httpx.post", return_value=mock_resp):
            with pytest.raises(InvalidCodeError, match="invalid_grant"):
                provider.complete_login(
                    code="bad", state="s", code_verifier="v",
                    redirect_uri="https://dash.example.com/auth/callback",
                )

    def test_500_raises_provider_error(self, provider):
        mock_resp = _mock_post(500, "internal error", ctype="text/plain")
        with patch("plugins.dashboard_auth.google.httpx.post", return_value=mock_resp):
            with pytest.raises(ProviderError, match="500"):
                provider.complete_login(
                    code="x", state="s", code_verifier="v",
                    redirect_uri="https://dash.example.com/auth/callback",
                )

    def test_network_error_raises_provider_error(self, provider):
        with patch(
            "plugins.dashboard_auth.google.httpx.post",
            side_effect=httpx.ConnectError("conn refused"),
        ):
            with pytest.raises(ProviderError, match="unreachable"):
                provider.complete_login(
                    code="x", state="s", code_verifier="v",
                    redirect_uri="https://dash.example.com/auth/callback",
                )

    def test_email_not_verified_rejected(self, provider, rsa_keypair):
        id_token = _mint_id_token(rsa_keypair, email_verified=False)
        mock_resp = _mock_post(
            200,
            {"access_token": "ya29.opaque", "token_type": "Bearer", "id_token": id_token},
        )
        with patch("plugins.dashboard_auth.google.httpx.post", return_value=mock_resp):
            with pytest.raises(InvalidCodeError, match="email_verified"):
                provider.complete_login(
                    code="x", state="s", code_verifier="v",
                    redirect_uri="https://dash.example.com/auth/callback",
                )

    def test_missing_email_verified_claim_tolerated(self, provider, rsa_keypair):
        # Some flows omit the claim entirely — must not be treated as False.
        id_token = _mint_id_token(rsa_keypair, email_verified=None)
        mock_resp = _mock_post(
            200,
            {"access_token": "ya29.opaque", "token_type": "Bearer", "id_token": id_token},
        )
        with patch("plugins.dashboard_auth.google.httpx.post", return_value=mock_resp):
            session = provider.complete_login(
                code="x", state="s", code_verifier="v",
                redirect_uri="https://dash.example.com/auth/callback",
            )
        assert session.email == "alice@example.com"

    def test_wrong_audience_raises_provider_error(self, provider, rsa_keypair):
        id_token = _mint_id_token(rsa_keypair, aud="someone-elses-client-id")
        mock_resp = _mock_post(
            200,
            {"access_token": "ya29.opaque", "token_type": "Bearer", "id_token": id_token},
        )
        with patch("plugins.dashboard_auth.google.httpx.post", return_value=mock_resp):
            with pytest.raises(ProviderError, match="verification failed"):
                provider.complete_login(
                    code="x", state="s", code_verifier="v",
                    redirect_uri="https://dash.example.com/auth/callback",
                )


# ---------------------------------------------------------------------------
# complete_login — DASHBOARD_ALLOWED_USERS / GATEWAY_ALLOWED_USERS allowlist
# ---------------------------------------------------------------------------


class TestAllowlist:
    @pytest.fixture
    def provider(self, rsa_keypair):
        return _make_provider(rsa_keypair)

    def _complete(self, provider, rsa_keypair, *, email="alice@example.com"):
        id_token = _mint_id_token(rsa_keypair, email=email)
        mock_resp = _mock_post(
            200,
            {"access_token": "ya29.opaque", "token_type": "Bearer", "id_token": id_token},
        )
        with patch("plugins.dashboard_auth.google.httpx.post", return_value=mock_resp):
            return provider.complete_login(
                code="x", state="s", code_verifier="v",
                redirect_uri="https://dash.example.com/auth/callback",
            )

    def test_no_allowlist_permits_any_authenticated_user(self, provider, rsa_keypair, monkeypatch):
        monkeypatch.delenv("DASHBOARD_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
        session = self._complete(provider, rsa_keypair)
        assert session.email == "alice@example.com"

    def test_allowlisted_email_permitted(self, provider, rsa_keypair, monkeypatch):
        monkeypatch.setenv("DASHBOARD_ALLOWED_USERS", "alice@example.com, bob@example.com")
        session = self._complete(provider, rsa_keypair)
        assert session.email == "alice@example.com"

    def test_allowlist_is_case_insensitive(self, provider, rsa_keypair, monkeypatch):
        monkeypatch.setenv("DASHBOARD_ALLOWED_USERS", "ALICE@EXAMPLE.COM")
        session = self._complete(provider, rsa_keypair)
        assert session.email == "alice@example.com"

    def test_non_allowlisted_email_rejected(self, provider, rsa_keypair, monkeypatch):
        monkeypatch.setenv("DASHBOARD_ALLOWED_USERS", "bob@example.com")
        with pytest.raises(InvalidCodeError, match="not on the dashboard allowlist"):
            self._complete(provider, rsa_keypair, email="alice@example.com")

    def test_falls_back_to_gateway_allowed_users(self, provider, rsa_keypair, monkeypatch):
        monkeypatch.delenv("DASHBOARD_ALLOWED_USERS", raising=False)
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "alice@example.com")
        session = self._complete(provider, rsa_keypair)
        assert session.email == "alice@example.com"

    def test_dashboard_allowlist_takes_precedence_over_gateway(self, provider, rsa_keypair, monkeypatch):
        monkeypatch.setenv("DASHBOARD_ALLOWED_USERS", "alice@example.com")
        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "someone-else@example.com")
        session = self._complete(provider, rsa_keypair)
        assert session.email == "alice@example.com"

    def test_falls_back_to_config_yaml_when_env_unset(self, provider, rsa_keypair, monkeypatch):
        monkeypatch.delenv("DASHBOARD_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"dashboard": {"oauth": {"google_allowed_users": "bob@example.com"}}},
        )
        with pytest.raises(InvalidCodeError, match="not on the dashboard allowlist"):
            self._complete(provider, rsa_keypair, email="alice@example.com")

    def test_env_allowlist_takes_precedence_over_config_yaml(self, provider, rsa_keypair, monkeypatch):
        monkeypatch.setenv("DASHBOARD_ALLOWED_USERS", "alice@example.com")
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"dashboard": {"oauth": {"google_allowed_users": "someone-else@example.com"}}},
        )
        session = self._complete(provider, rsa_keypair)
        assert session.email == "alice@example.com"

    def test_config_yaml_load_failure_falls_back_to_no_allowlist(self, provider, rsa_keypair, monkeypatch):
        monkeypatch.delenv("DASHBOARD_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)

        def _broken():
            raise RuntimeError("disk error")

        monkeypatch.setattr("hermes_cli.config.load_config", _broken)
        session = self._complete(provider, rsa_keypair)
        assert session.email == "alice@example.com"


# ---------------------------------------------------------------------------
# verify_session
# ---------------------------------------------------------------------------


class TestVerifySession:
    @pytest.fixture
    def provider(self, rsa_keypair):
        return _make_provider(rsa_keypair)

    def test_valid_token_returns_session(self, provider, rsa_keypair):
        id_token = _mint_id_token(rsa_keypair, hd="example.com")
        session = provider.verify_session(access_token=id_token)
        assert session is not None
        assert session.user_id == "110169484474386276334"
        assert session.org_id == "example.com"
        assert session.expires_at > int(time.time())

    def test_expired_token_returns_none(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, ttl_seconds=-1)
        assert provider.verify_session(access_token=token) is None

    def test_email_not_verified_returns_none(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, email_verified=False)
        assert provider.verify_session(access_token=token) is None

    def test_wrong_audience_raises_provider_error(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, aud="someone-elses-client-id")
        with pytest.raises(ProviderError, match="verification failed"):
            provider.verify_session(access_token=token)

    def test_wrong_issuer_raises_provider_error(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, iss="https://evil.example")
        with pytest.raises(ProviderError) as excinfo:
            provider.verify_session(access_token=token)
        assert "evil.example" in str(excinfo.value)

    def test_jwks_unreachable_raises_provider_error(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair)
        bad_client = MagicMock()
        bad_client.get_signing_key_from_jwt.side_effect = jwt.PyJWKClientError("fetch failed")
        provider._jwks_client = bad_client
        with pytest.raises(ProviderError, match="JWKS"):
            provider.verify_session(access_token=token)

    def test_jwks_client_sends_explicit_http_headers(self, provider):
        provider._jwks_client = None
        with patch("jwt.PyJWKClient") as client_cls:
            provider._get_jwks_client()
        client_cls.assert_called_once_with(
            _DISCOVERY_DOC["jwks_uri"],
            cache_keys=True,
            lifespan=google_plugin._JWKS_CACHE_SECONDS,
            headers={
                "Accept": "application/json",
                "User-Agent": "HermesAgent/1.0",
            },
        )


# ---------------------------------------------------------------------------
# refresh_session + revoke_session
# ---------------------------------------------------------------------------


class TestRefreshAndRevoke:
    @pytest.fixture
    def provider(self, rsa_keypair):
        return _make_provider(rsa_keypair)

    def test_no_refresh_token_raises(self, provider):
        with pytest.raises(RefreshExpiredError):
            provider.refresh_session(refresh_token="")

    def test_refresh_happy_path_returns_rotated_session(self, provider, rsa_keypair):
        id_token = _mint_id_token(rsa_keypair)
        mock_resp = _mock_post(
            200,
            {
                "access_token": "ya29.new",
                "token_type": "Bearer",
                "id_token": id_token,
                "refresh_token": "rt_rotated",
            },
        )
        with patch(
            "plugins.dashboard_auth.google.httpx.post", return_value=mock_resp
        ) as mock_post:
            session = provider.refresh_session(refresh_token="rt_old")
        assert session.refresh_token == "rt_rotated"
        assert session.provider == "google"
        _, kwargs = mock_post.call_args
        assert kwargs["data"]["grant_type"] == "refresh_token"
        assert kwargs["data"]["refresh_token"] == "rt_old"

    def test_refresh_keeps_previous_token_when_not_rotated(self, provider, rsa_keypair):
        # Google usually does not rotate the refresh token.
        id_token = _mint_id_token(rsa_keypair)
        mock_resp = _mock_post(
            200, {"access_token": "ya29.new", "token_type": "Bearer", "id_token": id_token}
        )
        with patch("plugins.dashboard_auth.google.httpx.post", return_value=mock_resp):
            session = provider.refresh_session(refresh_token="rt_old")
        assert session.refresh_token == "rt_old"

    def test_refresh_400_raises_refresh_expired(self, provider):
        mock_resp = _mock_post(400, {"error": "invalid_grant"})
        with patch("plugins.dashboard_auth.google.httpx.post", return_value=mock_resp):
            with pytest.raises(RefreshExpiredError, match="invalid_grant"):
                provider.refresh_session(refresh_token="rt_old")

    def test_refresh_rejects_email_removed_from_allowlist(self, provider, rsa_keypair, monkeypatch):
        monkeypatch.setenv("DASHBOARD_ALLOWED_USERS", "bob@example.com")
        id_token = _mint_id_token(rsa_keypair, email="alice@example.com")
        mock_resp = _mock_post(
            200, {"access_token": "ya29.new", "token_type": "Bearer", "id_token": id_token}
        )
        with patch("plugins.dashboard_auth.google.httpx.post", return_value=mock_resp):
            with pytest.raises(RefreshExpiredError, match="not on the dashboard allowlist"):
                provider.refresh_session(refresh_token="rt_old")

    def test_refresh_permits_allowlisted_email(self, provider, rsa_keypair, monkeypatch):
        monkeypatch.setenv("DASHBOARD_ALLOWED_USERS", "alice@example.com")
        id_token = _mint_id_token(rsa_keypair, email="alice@example.com")
        mock_resp = _mock_post(
            200, {"access_token": "ya29.new", "token_type": "Bearer", "id_token": id_token}
        )
        with patch("plugins.dashboard_auth.google.httpx.post", return_value=mock_resp):
            session = provider.refresh_session(refresh_token="rt_old")
        assert session.email == "alice@example.com"

    def test_revoke_is_best_effort_and_does_not_raise(self, provider):
        with patch(
            "plugins.dashboard_auth.google.httpx.post",
            side_effect=httpx.ConnectError("conn refused"),
        ):
            assert provider.revoke_session(refresh_token="rt") is None

    def test_revoke_noop_on_empty_token(self, provider):
        with patch("plugins.dashboard_auth.google.httpx.post") as mock_post:
            provider.revoke_session(refresh_token="")
        mock_post.assert_not_called()

    def test_revoke_posts_to_google_endpoint(self, provider):
        mock_resp = _mock_post(200, {})
        with patch(
            "plugins.dashboard_auth.google.httpx.post", return_value=mock_resp
        ) as mock_post:
            provider.revoke_session(refresh_token="rt")
        args, kwargs = mock_post.call_args
        assert args[0] == "https://oauth2.googleapis.com/revoke"
        assert kwargs["data"] == {"token": "rt"}

    def test_revoke_logs_non_200_status_without_raising(self, provider, caplog):
        mock_resp = _mock_post(400, {"error": "invalid_token"})
        with patch("plugins.dashboard_auth.google.httpx.post", return_value=mock_resp):
            with caplog.at_level("DEBUG", logger="plugins.dashboard_auth.google"):
                result = provider.revoke_session(refresh_token="already-expired")
        assert result is None
        assert any("status_code=400" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Plugin entry point: env-gated registration + config.yaml precedence
# ---------------------------------------------------------------------------


class TestPluginRegister:
    @pytest.fixture(autouse=True)
    def clear_env(self, monkeypatch):
        for var in (
            "HERMES_DASHBOARD_GOOGLE_CLIENT_ID",
            "HERMES_DASHBOARD_GOOGLE_CLIENT_SECRET",
        ):
            monkeypatch.delenv(var, raising=False)

    @pytest.fixture
    def patch_config(self, monkeypatch):
        def _set(oauth_block: Dict[str, Any] | None) -> None:
            cfg = {}
            if oauth_block is not None:
                cfg = {"dashboard": {"oauth": oauth_block}}
            monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

        return _set

    def test_skips_when_unconfigured(self, patch_config):
        patch_config(None)
        ctx = MagicMock()
        google_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "HERMES_DASHBOARD_GOOGLE_CLIENT_ID" in google_plugin.LAST_SKIP_REASON

    def test_skips_when_secret_missing(self, patch_config, monkeypatch):
        patch_config(None)
        monkeypatch.setenv("HERMES_DASHBOARD_GOOGLE_CLIENT_ID", "abc.apps.googleusercontent.com")
        ctx = MagicMock()
        google_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "SECRET" in google_plugin.LAST_SKIP_REASON

    def test_registers_from_env(self, patch_config, monkeypatch):
        patch_config(None)
        monkeypatch.setenv("HERMES_DASHBOARD_GOOGLE_CLIENT_ID", _CLIENT_ID)
        monkeypatch.setenv("HERMES_DASHBOARD_GOOGLE_CLIENT_SECRET", _CLIENT_SECRET)
        ctx = MagicMock()
        google_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert isinstance(registered, google_plugin.GoogleDashboardAuthProvider)
        assert registered._client_id == _CLIENT_ID
        assert google_plugin.LAST_SKIP_REASON == ""

    def test_registers_from_config_yaml(self, patch_config):
        patch_config(
            {"google_client_id": _CLIENT_ID, "google_client_secret": _CLIENT_SECRET}
        )
        ctx = MagicMock()
        google_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._client_id == _CLIENT_ID

    def test_env_overrides_config(self, patch_config, monkeypatch):
        patch_config(
            {"google_client_id": "from-config", "google_client_secret": "from-config-secret"}
        )
        monkeypatch.setenv("HERMES_DASHBOARD_GOOGLE_CLIENT_ID", "from-env")
        monkeypatch.setenv("HERMES_DASHBOARD_GOOGLE_CLIENT_SECRET", "from-env-secret")
        ctx = MagicMock()
        google_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._client_id == "from-env"

    def test_config_load_failure_falls_through(self, monkeypatch):
        def _broken():
            raise RuntimeError("disk error")

        monkeypatch.setattr("hermes_cli.config.load_config", _broken)
        ctx = MagicMock()
        google_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "HERMES_DASHBOARD_GOOGLE_CLIENT_ID" in google_plugin.LAST_SKIP_REASON

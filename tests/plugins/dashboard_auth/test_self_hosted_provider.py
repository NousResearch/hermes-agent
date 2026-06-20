"""Tests for the bundled self-hosted OIDC dashboard-auth plugin.

Covers, by analogy with ``test_nous_provider.py``:

1. Plugin entry-point registration gating (env + config.yaml precedence).
2. ``start_login`` shape (PKCE/state, authorize URL parameters, OIDC discovery).
3. ``complete_login`` httpx-mocked happy path + error mapping (ID-token grant).
4. ``verify_session`` ID-token verification — RSA keypair, audience/issuer
   pinning, standard OIDC claim mapping (sub/email/name/groups).
5. ``refresh_session`` rotation + error mapping, ``revoke_session`` (RFC 7009).
6. OIDC discovery: endpoint extraction, issuer pinning, https enforcement.

All HTTP is mocked: nothing here talks to a real IDP.
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

import plugins.dashboard_auth.self_hosted as oidc_plugin
from hermes_cli.dashboard_auth import (
    InvalidCodeError,
    LoginStart,
    ProviderError,
    RefreshExpiredError,
    Session,
    assert_protocol_compliance,
)

_ISSUER = "https://auth.example.com/application/o/hermes"
_CLIENT_ID = "hermes-dashboard"

_DISCOVERY_DOC = {
    "issuer": _ISSUER,
    "authorization_endpoint": f"{_ISSUER}/authorize",
    "token_endpoint": f"{_ISSUER}/token",
    "jwks_uri": f"{_ISSUER}/jwks",
    "revocation_endpoint": f"{_ISSUER}/revoke",
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
# Token-mint helper — standard OIDC ID-token claims
# ---------------------------------------------------------------------------


def _mint_id_token(
    rsa_keypair: Dict[str, Any],
    *,
    iss: str = _ISSUER,
    aud: str = _CLIENT_ID,
    sub: str = "usr_abc",
    email: str | None = "alice@example.com",
    name: str | None = "Alice Example",
    groups: Any = None,
    org_id: str | None = None,
    ttl_seconds: int = 900,
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
    if name is not None:
        claims["name"] = name
    if groups is not None:
        claims["groups"] = groups
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


def _make_provider(rsa_keypair, *, scopes: str | None = None):
    """Construct a provider with discovery + JWKS stubbed (no network)."""
    kwargs: Dict[str, Any] = {"issuer": _ISSUER, "client_id": _CLIENT_ID}
    if scopes is not None:
        kwargs["scopes"] = scopes
    p = oidc_plugin.SelfHostedOIDCProvider(**kwargs)
    # Pre-seed discovery so nothing hits the network.
    p._discovery = dict(_DISCOVERY_DOC)
    p._discovery_fetched_at = time.time()
    # Patch the JWKS client to return our fixture key.
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
        assert_protocol_compliance(oidc_plugin.SelfHostedOIDCProvider)

    def test_name_and_display(self):
        p = oidc_plugin.SelfHostedOIDCProvider(issuer=_ISSUER, client_id=_CLIENT_ID)
        assert p.name == "self-hosted"
        assert p.display_name == "Self-Hosted OIDC"

    def test_strips_trailing_slash_from_issuer(self):
        p = oidc_plugin.SelfHostedOIDCProvider(
            issuer=_ISSUER + "/", client_id=_CLIENT_ID
        )
        assert p._issuer == _ISSUER

    def test_requires_issuer(self):
        with pytest.raises(ValueError, match="issuer"):
            oidc_plugin.SelfHostedOIDCProvider(issuer="", client_id=_CLIENT_ID)

    def test_requires_client_id(self):
        with pytest.raises(ValueError, match="client_id"):
            oidc_plugin.SelfHostedOIDCProvider(issuer=_ISSUER, client_id="")

    def test_rejects_non_https_issuer(self):
        with pytest.raises(ProviderError, match="https"):
            oidc_plugin.SelfHostedOIDCProvider(
                issuer="http://auth.example.com", client_id=_CLIENT_ID
            )

    def test_allows_http_localhost_issuer(self):
        # Local dev against a loopback IDP is allowed.
        p = oidc_plugin.SelfHostedOIDCProvider(
            issuer="http://localhost:9000", client_id=_CLIENT_ID
        )
        assert p._issuer == "http://localhost:9000"

    def test_default_scopes(self):
        p = oidc_plugin.SelfHostedOIDCProvider(issuer=_ISSUER, client_id=_CLIENT_ID)
        assert p._scopes == "openid profile email"

    def test_empty_scopes_falls_back_to_default(self):
        p = oidc_plugin.SelfHostedOIDCProvider(
            issuer=_ISSUER, client_id=_CLIENT_ID, scopes="   "
        )
        assert p._scopes == "openid profile email"


# ---------------------------------------------------------------------------
# OIDC discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def _provider(self):
        return oidc_plugin.SelfHostedOIDCProvider(
            issuer=_ISSUER, client_id=_CLIENT_ID
        )

    def _mock_get(self, status_code, body, *, ctype="application/json"):
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = status_code
        resp.json = MagicMock(return_value=body)
        resp.text = json.dumps(body) if isinstance(body, dict) else str(body)
        resp.headers = {"content-type": ctype}
        return resp

    def test_discovery_url(self):
        p = self._provider()
        assert p._discovery_url() == (
            f"{_ISSUER}/.well-known/openid-configuration"
        )

    def test_fetches_and_caches(self):
        p = self._provider()
        mock_resp = self._mock_get(200, dict(_DISCOVERY_DOC))
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.get", return_value=mock_resp
        ) as mock_get:
            disco1 = p._get_discovery()
            disco2 = p._get_discovery()
        assert disco1["token_endpoint"] == f"{_ISSUER}/token"
        assert disco1["authorization_endpoint"] == f"{_ISSUER}/authorize"
        assert disco1["jwks_uri"] == f"{_ISSUER}/jwks"
        assert disco1["revocation_endpoint"] == f"{_ISSUER}/revoke"
        # Cached — only one network call.
        assert mock_get.call_count == 1
        assert disco2 is disco1

    def test_discovery_404_raises(self):
        p = self._provider()
        mock_resp = self._mock_get(404, {})
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.get", return_value=mock_resp
        ):
            with pytest.raises(ProviderError, match="404"):
                p._get_discovery()

    def test_discovery_unreachable_raises(self):
        p = self._provider()
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.get",
            side_effect=httpx.ConnectError("no route"),
        ):
            with pytest.raises(ProviderError, match="unreachable"):
                p._get_discovery()

    def test_discovery_missing_endpoint_raises(self):
        p = self._provider()
        doc = dict(_DISCOVERY_DOC)
        del doc["token_endpoint"]
        mock_resp = self._mock_get(200, doc)
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.get", return_value=mock_resp
        ):
            with pytest.raises(ProviderError, match="token_endpoint"):
                p._get_discovery()

    def test_discovery_issuer_mismatch_raises(self):
        p = self._provider()
        doc = dict(_DISCOVERY_DOC)
        doc["issuer"] = "https://evil.example"
        mock_resp = self._mock_get(200, doc)
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.get", return_value=mock_resp
        ):
            with pytest.raises(ProviderError, match="issuer mismatch"):
                p._get_discovery()

    def test_discovery_issuer_trailing_slash_tolerated(self):
        p = self._provider()
        doc = dict(_DISCOVERY_DOC)
        doc["issuer"] = _ISSUER + "/"  # only a trailing-slash difference
        mock_resp = self._mock_get(200, doc)
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.get", return_value=mock_resp
        ):
            disco = p._get_discovery()
        assert disco["token_endpoint"] == f"{_ISSUER}/token"

    def test_discovery_rejects_non_https_endpoint(self):
        p = self._provider()
        doc = dict(_DISCOVERY_DOC)
        doc["token_endpoint"] = "http://auth.example.com/token"  # not loopback
        mock_resp = self._mock_get(200, doc)
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.get", return_value=mock_resp
        ):
            with pytest.raises(ProviderError, match="https"):
                p._get_discovery()


# ---------------------------------------------------------------------------
# start_login
# ---------------------------------------------------------------------------


class TestStartLogin:
    @pytest.fixture
    def provider(self, rsa_keypair):
        return _make_provider(rsa_keypair)

    def test_returns_login_start(self, provider):
        result = provider.start_login(
            redirect_uri="https://hermes.example/auth/callback"
        )
        assert isinstance(result, LoginStart)

    def test_redirect_url_targets_authorize_endpoint(self, provider):
        result = provider.start_login(
            redirect_uri="https://hermes.example/auth/callback"
        )
        assert result.redirect_url.startswith(f"{_ISSUER}/authorize?")

    def test_authorize_url_has_required_params(self, provider):
        result = provider.start_login(
            redirect_uri="https://hermes.example/auth/callback"
        )
        parsed = urllib.parse.urlparse(result.redirect_url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        assert params["response_type"] == "code"
        assert params["client_id"] == _CLIENT_ID
        assert params["redirect_uri"] == "https://hermes.example/auth/callback"
        assert params["scope"] == "openid profile email"
        assert params["code_challenge_method"] == "S256"
        assert "state" in params
        assert "code_challenge" in params

    def test_custom_scopes_used(self, rsa_keypair):
        provider = _make_provider(rsa_keypair, scopes="openid email groups")
        result = provider.start_login(
            redirect_uri="https://hermes.example/auth/callback"
        )
        parsed = urllib.parse.urlparse(result.redirect_url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        assert params["scope"] == "openid email groups"

    def test_code_verifier_length(self, provider):
        result = provider.start_login(
            redirect_uri="https://hermes.example/auth/callback"
        )
        pkce = result.cookie_payload["hermes_session_pkce"]
        parts = dict(seg.split("=", 1) for seg in pkce.split(";") if "=" in seg)
        assert 43 <= len(parts["verifier"]) <= 128  # RFC 7636 §4.1

    def test_state_in_cookie_matches_url(self, provider):
        result = provider.start_login(
            redirect_uri="https://hermes.example/auth/callback"
        )
        parsed = urllib.parse.urlparse(result.redirect_url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        pkce = result.cookie_payload["hermes_session_pkce"]
        parts = dict(seg.split("=", 1) for seg in pkce.split(";") if "=" in seg)
        assert parts["state"] == params["state"]

    def test_code_challenge_is_s256_of_verifier(self, provider):
        result = provider.start_login(
            redirect_uri="https://hermes.example/auth/callback"
        )
        parsed = urllib.parse.urlparse(result.redirect_url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        pkce = result.cookie_payload["hermes_session_pkce"]
        parts = dict(seg.split("=", 1) for seg in pkce.split(";") if "=" in seg)
        expected = (
            base64.urlsafe_b64encode(
                hashlib.sha256(parts["verifier"].encode("ascii")).digest()
            )
            .rstrip(b"=")
            .decode()
        )
        assert params["code_challenge"] == expected

    def test_two_calls_differ(self, provider):
        a = provider.start_login(redirect_uri="https://hermes.example/auth/callback")
        b = provider.start_login(redirect_uri="https://hermes.example/auth/callback")
        assert (
            a.cookie_payload["hermes_session_pkce"]
            != b.cookie_payload["hermes_session_pkce"]
        )

    def test_rejects_wrong_callback_path(self, provider):
        with pytest.raises(ProviderError, match="/auth/callback"):
            provider.start_login(redirect_uri="https://x.example/oauth/cb")

    def test_allows_http_localhost_redirect(self, provider):
        provider.start_login(redirect_uri="http://localhost:8080/auth/callback")
        provider.start_login(redirect_uri="http://127.0.0.1:8080/auth/callback")


# ---------------------------------------------------------------------------
# complete_login
# ---------------------------------------------------------------------------


class TestCompleteLogin:
    @pytest.fixture
    def provider(self, rsa_keypair):
        return _make_provider(rsa_keypair)

    def test_happy_path_returns_session(self, provider, rsa_keypair):
        id_token = _mint_id_token(rsa_keypair)
        mock_resp = _mock_post(
            200,
            {
                "access_token": "opaque-at",
                "id_token": id_token,
                "token_type": "Bearer",
                "refresh_token": "rt_initial",
            },
        )
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post", return_value=mock_resp
        ):
            session = provider.complete_login(
                code="abc",
                state="s",
                code_verifier="vfy",
                redirect_uri="https://hermes.example/auth/callback",
            )
        assert isinstance(session, Session)
        assert session.user_id == "usr_abc"
        assert session.provider == "self-hosted"
        assert session.email == "alice@example.com"
        assert session.display_name == "Alice Example"
        # The verified ID token is stored in the access_token slot.
        assert session.access_token == id_token
        assert session.refresh_token == "rt_initial"

    def test_tolerates_missing_refresh_token(self, provider, rsa_keypair):
        id_token = _mint_id_token(rsa_keypair)
        mock_resp = _mock_post(
            200, {"id_token": id_token, "token_type": "Bearer"}
        )
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post", return_value=mock_resp
        ):
            session = provider.complete_login(
                code="abc",
                state="s",
                code_verifier="vfy",
                redirect_uri="https://hermes.example/auth/callback",
            )
        assert session.refresh_token == ""

    def test_missing_id_token_raises(self, provider):
        mock_resp = _mock_post(
            200, {"access_token": "opaque", "token_type": "Bearer"}
        )
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post", return_value=mock_resp
        ):
            with pytest.raises(ProviderError, match="id_token"):
                provider.complete_login(
                    code="x",
                    state="s",
                    code_verifier="v",
                    redirect_uri="https://hermes.example/auth/callback",
                )

    def test_400_raises_invalid_code(self, provider):
        mock_resp = _mock_post(400, {"error": "invalid_grant"})
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post", return_value=mock_resp
        ):
            with pytest.raises(InvalidCodeError, match="invalid_grant"):
                provider.complete_login(
                    code="bad",
                    state="s",
                    code_verifier="v",
                    redirect_uri="https://hermes.example/auth/callback",
                )

    def test_500_raises_provider_error(self, provider):
        mock_resp = _mock_post(500, "boom", ctype="text/plain")
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post", return_value=mock_resp
        ):
            with pytest.raises(ProviderError, match="500"):
                provider.complete_login(
                    code="x",
                    state="s",
                    code_verifier="v",
                    redirect_uri="https://hermes.example/auth/callback",
                )

    def test_network_error_raises_provider_error(self, provider):
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post",
            side_effect=httpx.ConnectError("conn refused"),
        ):
            with pytest.raises(ProviderError, match="unreachable"):
                provider.complete_login(
                    code="x",
                    state="s",
                    code_verifier="v",
                    redirect_uri="https://hermes.example/auth/callback",
                )

    def test_unexpected_token_type_raises(self, provider, rsa_keypair):
        id_token = _mint_id_token(rsa_keypair)
        mock_resp = _mock_post(
            200, {"id_token": id_token, "token_type": "DPoP"}
        )
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post", return_value=mock_resp
        ):
            with pytest.raises(ProviderError, match="token_type"):
                provider.complete_login(
                    code="x",
                    state="s",
                    code_verifier="v",
                    redirect_uri="https://hermes.example/auth/callback",
                )

    def test_posts_authorization_code_grant(self, provider, rsa_keypair):
        id_token = _mint_id_token(rsa_keypair)
        mock_resp = _mock_post(200, {"id_token": id_token, "token_type": "Bearer"})
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post", return_value=mock_resp
        ) as mock_post:
            provider.complete_login(
                code="the-code",
                state="s",
                code_verifier="the-verifier",
                redirect_uri="https://hermes.example/auth/callback",
            )
        _, kwargs = mock_post.call_args
        assert kwargs["data"]["grant_type"] == "authorization_code"
        assert kwargs["data"]["code"] == "the-code"
        assert kwargs["data"]["code_verifier"] == "the-verifier"
        assert kwargs["data"]["client_id"] == _CLIENT_ID


# ---------------------------------------------------------------------------
# verify_session
# ---------------------------------------------------------------------------


class TestVerifySession:
    @pytest.fixture
    def provider(self, rsa_keypair):
        return _make_provider(rsa_keypair)

    def test_happy_path(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair)
        session = provider.verify_session(access_token=token)
        assert session is not None
        assert session.user_id == "usr_abc"
        assert session.email == "alice@example.com"
        assert session.display_name == "Alice Example"

    def test_expired_returns_none(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, ttl_seconds=-1)
        assert provider.verify_session(access_token=token) is None

    def test_wrong_audience_raises(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, aud="some-other-client")
        with pytest.raises(ProviderError, match="verification failed"):
            provider.verify_session(access_token=token)

    def test_wrong_issuer_raises(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, iss="https://evil.example")
        with pytest.raises(ProviderError, match="verification failed"):
            provider.verify_session(access_token=token)

    def test_failure_message_surfaces_claims(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, iss="https://evil.example")
        with pytest.raises(ProviderError) as excinfo:
            provider.verify_session(access_token=token)
        msg = str(excinfo.value)
        assert "'https://evil.example'" in msg
        assert f"'{_ISSUER}'" in msg

    def test_missing_sub_raises(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, sub="")
        with pytest.raises(ProviderError, match="sub"):
            provider.verify_session(access_token=token)

    def test_display_name_falls_back_to_preferred_username(
        self, provider, rsa_keypair
    ):
        token = _mint_id_token(
            rsa_keypair,
            name=None,
            email=None,
            extra_claims={"preferred_username": "alice42"},
        )
        session = provider.verify_session(access_token=token)
        assert session is not None
        assert session.display_name == "alice42"

    def test_org_id_from_org_claim(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, org_id="acme-corp")
        session = provider.verify_session(access_token=token)
        assert session is not None
        assert session.org_id == "acme-corp"

    def test_org_id_from_groups_when_no_org_claim(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, groups=["admins", "users"])
        session = provider.verify_session(access_token=token)
        assert session is not None
        assert session.org_id == "admins,users"

    def test_org_id_empty_when_neither_present(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair)
        session = provider.verify_session(access_token=token)
        assert session is not None
        assert session.org_id == ""

    def test_jwks_unreachable_raises(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair)
        bad_client = MagicMock()
        bad_client.get_signing_key_from_jwt.side_effect = jwt.PyJWKClientError(
            "fetch failed"
        )
        provider._jwks_client = bad_client
        with pytest.raises(ProviderError, match="JWKS"):
            provider.verify_session(access_token=token)

    def test_jwks_client_sends_explicit_http_headers(self):
        provider = oidc_plugin.SelfHostedOIDCProvider(
            issuer=_ISSUER, client_id=_CLIENT_ID
        )
        provider._discovery = dict(_DISCOVERY_DOC)
        provider._discovery_fetched_at = time.time()

        with patch("jwt.PyJWKClient") as client_cls:
            provider._get_jwks_client()

        client_cls.assert_called_once_with(
            _DISCOVERY_DOC["jwks_uri"],
            cache_keys=True,
            lifespan=oidc_plugin._JWKS_CACHE_SECONDS,
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

    def test_refresh_happy_path_rotates(self, provider, rsa_keypair):
        id_token = _mint_id_token(rsa_keypair)
        mock_resp = _mock_post(
            200,
            {
                "id_token": id_token,
                "token_type": "Bearer",
                "refresh_token": "rt_rotated",
            },
        )
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post", return_value=mock_resp
        ) as mock_post:
            session = provider.refresh_session(refresh_token="rt_old")
        assert isinstance(session, Session)
        assert session.access_token == id_token
        assert session.refresh_token == "rt_rotated"
        assert session.provider == "self-hosted"
        _, kwargs = mock_post.call_args
        assert kwargs["data"]["grant_type"] == "refresh_token"
        assert kwargs["data"]["refresh_token"] == "rt_old"
        assert kwargs["data"]["client_id"] == _CLIENT_ID

    def test_refresh_keeps_previous_rt_when_idp_omits(self, provider, rsa_keypair):
        # Some IDPs don't rotate; keep the caller's existing RT alive.
        id_token = _mint_id_token(rsa_keypair)
        mock_resp = _mock_post(200, {"id_token": id_token, "token_type": "Bearer"})
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post", return_value=mock_resp
        ):
            session = provider.refresh_session(refresh_token="rt_kept")
        assert session.refresh_token == "rt_kept"

    def test_refresh_400_raises_refresh_expired(self, provider):
        mock_resp = _mock_post(400, {"error": "invalid_grant"})
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post", return_value=mock_resp
        ):
            with pytest.raises(RefreshExpiredError, match="invalid_grant"):
                provider.refresh_session(refresh_token="rt_dead")

    def test_refresh_empty_token_no_network(self, provider):
        with patch("plugins.dashboard_auth.self_hosted.httpx.post") as mock_post:
            with pytest.raises(RefreshExpiredError):
                provider.refresh_session(refresh_token="")
        mock_post.assert_not_called()

    def test_refresh_network_error_raises_provider_error(self, provider):
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post",
            side_effect=httpx.RequestError("boom"),
        ):
            with pytest.raises(ProviderError, match="unreachable"):
                provider.refresh_session(refresh_token="rt_x")

    def test_revoke_posts_to_revocation_endpoint(self, provider):
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post"
        ) as mock_post:
            provider.revoke_session(refresh_token="rt_x")
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == f"{_ISSUER}/revoke"
        assert kwargs["data"]["token"] == "rt_x"

    def test_revoke_empty_token_noop(self, provider):
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post"
        ) as mock_post:
            assert provider.revoke_session(refresh_token="") is None
        mock_post.assert_not_called()

    def test_revoke_swallows_errors(self, provider):
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post",
            side_effect=httpx.RequestError("down"),
        ):
            # Must not raise.
            assert provider.revoke_session(refresh_token="rt_x") is None

    def test_revoke_noop_when_no_revocation_endpoint(self, provider):
        provider._discovery["revocation_endpoint"] = ""
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post"
        ) as mock_post:
            assert provider.revoke_session(refresh_token="rt_x") is None
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Plugin entry point: env + config.yaml precedence
# ---------------------------------------------------------------------------


class TestPluginRegister:
    @pytest.fixture(autouse=True)
    def clear_env(self, monkeypatch):
        for var in (
            "HERMES_DASHBOARD_OIDC_ISSUER",
            "HERMES_DASHBOARD_OIDC_CLIENT_ID",
            "HERMES_DASHBOARD_OIDC_SCOPES",
        ):
            monkeypatch.delenv(var, raising=False)

    @pytest.fixture
    def patch_config(self, monkeypatch):
        def _set(oauth_block):
            cfg = {}
            if oauth_block is not None:
                cfg = {"dashboard": {"oauth": oauth_block}}
            monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

        return _set

    def test_skips_when_unconfigured(self, patch_config):
        patch_config(None)
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "HERMES_DASHBOARD_OIDC_ISSUER" in oidc_plugin.LAST_SKIP_REASON
        assert "self_hosted" in oidc_plugin.LAST_SKIP_REASON

    def test_skips_when_only_issuer_set(self, patch_config, monkeypatch):
        patch_config(None)
        monkeypatch.setenv("HERMES_DASHBOARD_OIDC_ISSUER", _ISSUER)
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()

    def test_registers_from_env(self, patch_config, monkeypatch):
        patch_config(None)
        monkeypatch.setenv("HERMES_DASHBOARD_OIDC_ISSUER", _ISSUER)
        monkeypatch.setenv("HERMES_DASHBOARD_OIDC_CLIENT_ID", _CLIENT_ID)
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert isinstance(registered, oidc_plugin.SelfHostedOIDCProvider)
        assert registered._issuer == _ISSUER
        assert registered._client_id == _CLIENT_ID
        assert registered._scopes == "openid profile email"
        assert oidc_plugin.LAST_SKIP_REASON == ""

    def test_registers_from_config_yaml(self, patch_config):
        patch_config(
            {"self_hosted": {"issuer": _ISSUER, "client_id": _CLIENT_ID}}
        )
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._issuer == _ISSUER
        assert registered._client_id == _CLIENT_ID

    def test_env_overrides_config(self, patch_config, monkeypatch):
        patch_config(
            {
                "self_hosted": {
                    "issuer": "https://config.example",
                    "client_id": "config-client",
                }
            }
        )
        monkeypatch.setenv("HERMES_DASHBOARD_OIDC_ISSUER", _ISSUER)
        monkeypatch.setenv("HERMES_DASHBOARD_OIDC_CLIENT_ID", _CLIENT_ID)
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._issuer == _ISSUER
        assert registered._client_id == _CLIENT_ID

    def test_empty_env_does_not_shadow_config(self, patch_config, monkeypatch):
        patch_config(
            {"self_hosted": {"issuer": _ISSUER, "client_id": _CLIENT_ID}}
        )
        monkeypatch.setenv("HERMES_DASHBOARD_OIDC_ISSUER", "")
        monkeypatch.setenv("HERMES_DASHBOARD_OIDC_CLIENT_ID", "")
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_called_once()
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._issuer == _ISSUER

    def test_custom_scopes_from_config(self, patch_config):
        patch_config(
            {
                "self_hosted": {
                    "issuer": _ISSUER,
                    "client_id": _CLIENT_ID,
                    "scopes": "openid email",
                }
            }
        )
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._scopes == "openid email"

    def test_config_load_failure_falls_through(self, monkeypatch):
        def _broken():
            raise OSError("unreadable")

        monkeypatch.setattr("hermes_cli.config.load_config", _broken)
        ctx = MagicMock()
        oidc_plugin.register(ctx)  # must not raise
        ctx.register_dashboard_auth_provider.assert_not_called()

    def test_non_dict_oauth_section_tolerated(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"dashboard": {"oauth": "wrong type"}},
        )
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()

    def test_non_https_issuer_skips_with_reason(self, patch_config, monkeypatch):
        patch_config(None)
        monkeypatch.setenv(
            "HERMES_DASHBOARD_OIDC_ISSUER", "http://insecure.example"
        )
        monkeypatch.setenv("HERMES_DASHBOARD_OIDC_CLIENT_ID", _CLIENT_ID)
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "construction failed" in oidc_plugin.LAST_SKIP_REASON

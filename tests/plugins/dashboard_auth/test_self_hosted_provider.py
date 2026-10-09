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


def _make_provider(
    rsa_keypair,
    *,
    scopes: str | None = None,
    client_secret: str | None = None,
    auth_methods: Any = "__unset__",
):
    """Construct a provider with discovery + JWKS stubbed (no network).

    ``client_secret`` flips the provider into confidential mode. ``auth_methods``
    overrides ``token_endpoint_auth_methods_supported`` in the seeded discovery
    doc (pass a list, or ``None`` to omit the key entirely); left unset, the
    discovery doc carries no auth-methods key (the absent-key default).
    """
    kwargs: Dict[str, Any] = {"issuer": _ISSUER, "client_id": _CLIENT_ID}
    if scopes is not None:
        kwargs["scopes"] = scopes
    if client_secret is not None:
        kwargs["client_secret"] = client_secret
    p = oidc_plugin.SelfHostedOIDCProvider(**kwargs)
    # Pre-seed discovery so nothing hits the network.
    disco = dict(_DISCOVERY_DOC)
    if auth_methods != "__unset__":
        if auth_methods is None:
            disco.pop("token_endpoint_auth_methods_supported", None)
        else:
            disco["token_endpoint_auth_methods_supported"] = auth_methods
    p._discovery = disco
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


    def test_strips_trailing_slash_from_issuer(self):
        p = oidc_plugin.SelfHostedOIDCProvider(
            issuer=_ISSUER + "/", client_id=_CLIENT_ID
        )
        assert p._issuer == _ISSUER

    def test_requires_issuer(self):
        with pytest.raises(ValueError, match="issuer"):
            oidc_plugin.SelfHostedOIDCProvider(issuer="", client_id=_CLIENT_ID)


    def test_rejects_non_https_issuer(self):
        with pytest.raises(ProviderError, match="https"):
            oidc_plugin.SelfHostedOIDCProvider(
                issuer="http://auth.example.com", client_id=_CLIENT_ID
            )


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


# ---------------------------------------------------------------------------
# OIDC discovery against a REAL HTTP server that redirects (regression)
# ---------------------------------------------------------------------------


class TestDiscoveryRealRedirect:
    """Discovery must follow a 3xx on the .well-known GET.

    The rest of the discovery suite mocks ``httpx.get`` with a canned 200, so
    it cannot see httpx's ``follow_redirects=False`` default. Many real IDPs
    answer the discovery GET with a redirect rather than a direct 200 —
    Authentik canonicalises the ``.well-known`` path, and any IDP behind a
    reverse proxy doing http→https upgrade redirects too. Before the fix the
    bare 3xx (empty body) tripped the ``status != 200`` guard and surfaced as
    ``provider_unreachable`` → HTTP 503 (the symptom in the user report:
    ``curl -o`` writing zero bytes is exactly a redirect with no body).

    This exercises the real httpx transport against a loopback server, so it
    fails without ``follow_redirects=True`` and passes with it — a behaviour
    contract, not a mock-shaped snapshot.
    """

    def _serve(self, handler_cls):
        import http.server
        import socketserver
        import threading

        # Bind :0 so the OS picks a free port (parallel-runner safe).
        httpd = socketserver.TCPServer(("127.0.0.1", 0), handler_cls)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        return httpd, port


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


    def test_state_in_cookie_matches_url(self, provider):
        result = provider.start_login(
            redirect_uri="https://hermes.example/auth/callback"
        )
        parsed = urllib.parse.urlparse(result.redirect_url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        pkce = result.cookie_payload["hermes_session_pkce"]
        parts = dict(seg.split("=", 1) for seg in pkce.split(";") if "=" in seg)
        assert parts["state"] == params["state"]


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


# ---------------------------------------------------------------------------
# Confidential client (client_secret) — token-endpoint client authentication
# ---------------------------------------------------------------------------


_GOOD_TOKEN_RESP_KEYS = {"token_type": "Bearer", "refresh_token": "rt_initial"}


def _decode_basic(header_value: str) -> tuple[str, str]:
    """Decode a ``Basic <b64>`` Authorization header back to (user, pass)."""
    assert header_value.startswith("Basic ")
    raw = base64.b64decode(header_value[len("Basic ") :]).decode("utf-8")
    user, _, pw = raw.partition(":")
    # client_id / secret are form-url-encoded before base64 (RFC 6749 §2.3.1).
    return urllib.parse.unquote(user), urllib.parse.unquote(pw)


class TestConfidentialClient:
    """A configured ``client_secret`` authenticates the client at the token
    endpoint (basic header or post body, auto-selected from discovery), while
    PKCE is still sent. A public client (no secret) is byte-identical to the
    pre-confidential-client behaviour — no secret anywhere, no auth header."""

    def _complete(self, provider, rsa_keypair):
        id_token = _mint_id_token(rsa_keypair)
        mock_resp = _mock_post(200, {"id_token": id_token, **_GOOD_TOKEN_RESP_KEYS})
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
        return kwargs

    # -- public client: nothing changes ------------------------------------

    def test_public_client_sends_no_secret_or_auth_header(self, rsa_keypair):
        # No client_secret configured → no Authorization header, no
        # client_secret in the body. Pins the unchanged public-client contract.
        provider = _make_provider(rsa_keypair)  # public
        kwargs = self._complete(provider, rsa_keypair)
        assert "Authorization" not in kwargs["headers"]
        assert "client_secret" not in kwargs["data"]
        # PKCE still present.
        assert kwargs["data"]["code_verifier"] == "the-verifier"
        # Header is exactly the pre-feature value.
        assert kwargs["headers"] == {"Accept": "application/json"}

    # -- basic (default & explicit) ----------------------------------------

    def test_confidential_defaults_to_basic_when_methods_absent(self, rsa_keypair):
        # Discovery advertises no auth methods → OIDC default is Basic.
        provider = _make_provider(
            rsa_keypair, client_secret="s3cr3t", auth_methods=None
        )
        kwargs = self._complete(provider, rsa_keypair)
        assert "client_secret" not in kwargs["data"]  # not in body for basic
        user, pw = _decode_basic(kwargs["headers"]["Authorization"])
        assert (user, pw) == (_CLIENT_ID, "s3cr3t")
        # PKCE still sent alongside the secret.
        assert kwargs["data"]["code_verifier"] == "the-verifier"


    # -- post --------------------------------------------------------------


    # -- url-encoding of reserved chars ------------------------------------

    def test_basic_url_encodes_reserved_chars_in_secret(self, rsa_keypair):
        # A secret with ':' / '@' / space must round-trip through the Basic
        # header exactly — these are exactly the chars that corrupt a naive
        # "id:secret" concatenation.
        tricky = "p@ss:wo rd/+="
        provider = _make_provider(
            rsa_keypair, client_secret=tricky, auth_methods=["client_secret_basic"]
        )
        kwargs = self._complete(provider, rsa_keypair)
        user, pw = _decode_basic(kwargs["headers"]["Authorization"])
        assert user == _CLIENT_ID
        assert pw == tricky

    # -- blank secret is treated as public ---------------------------------


    # -- refresh grant also authenticates ----------------------------------

    def test_refresh_grant_authenticates_confidential_client(self, rsa_keypair):
        provider = _make_provider(
            rsa_keypair, client_secret="s3cr3t", auth_methods=["client_secret_post"]
        )
        id_token = _mint_id_token(rsa_keypair)
        mock_resp = _mock_post(
            200, {"id_token": id_token, "token_type": "Bearer", "refresh_token": "rt2"}
        )
        with patch(
            "plugins.dashboard_auth.self_hosted.httpx.post", return_value=mock_resp
        ) as mock_post:
            provider.refresh_session(refresh_token="rt_old")
        _, kwargs = mock_post.call_args
        assert kwargs["data"]["grant_type"] == "refresh_token"
        assert kwargs["data"]["client_secret"] == "s3cr3t"


    # -- revocation also authenticates -------------------------------------


    # -- the secret never appears in logs ----------------------------------

    def test_secret_not_in_repr_or_log(self, rsa_keypair, caplog):
        import logging

        with caplog.at_level(logging.INFO):
            provider = _make_provider(
                rsa_keypair, client_secret="sup3r-s3cr3t", auth_methods=None
            )
        # The provider object's repr must not leak the secret.
        assert "sup3r-s3cr3t" not in repr(provider)
        assert "sup3r-s3cr3t" not in caplog.text


# ---------------------------------------------------------------------------
# verify_session
# ---------------------------------------------------------------------------


class TestVerifySession:
    @pytest.fixture
    def provider(self, rsa_keypair):
        return _make_provider(rsa_keypair)


    def test_expired_returns_none(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, ttl_seconds=-1)
        assert provider.verify_session(access_token=token) is None

    def test_wrong_audience_raises(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, aud="some-other-client")
        with pytest.raises(ProviderError, match="verification failed"):
            provider.verify_session(access_token=token)


    def test_failure_message_surfaces_claims(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair, iss="https://evil.example")
        with pytest.raises(ProviderError) as excinfo:
            provider.verify_session(access_token=token)
        msg = str(excinfo.value)
        assert "'https://evil.example'" in msg
        assert f"'{_ISSUER}'" in msg


    def test_jwks_unreachable_raises(self, provider, rsa_keypair):
        token = _mint_id_token(rsa_keypair)
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
        return _make_provider(rsa_keypair)


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
            "HERMES_DASHBOARD_OIDC_CLIENT_SECRET",
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


    def test_config_load_failure_falls_through(self, monkeypatch):
        def _broken():
            raise OSError("unreadable")

        monkeypatch.setattr("hermes_cli.config.load_config", _broken)
        ctx = MagicMock()
        oidc_plugin.register(ctx)  # must not raise
        ctx.register_dashboard_auth_provider.assert_not_called()


    # -- client_secret wiring ----------------------------------------------


    def test_secret_from_env(self, patch_config, monkeypatch):
        patch_config(None)
        monkeypatch.setenv("HERMES_DASHBOARD_OIDC_ISSUER", _ISSUER)
        monkeypatch.setenv("HERMES_DASHBOARD_OIDC_CLIENT_ID", _CLIENT_ID)
        monkeypatch.setenv("HERMES_DASHBOARD_OIDC_CLIENT_SECRET", "env-secret")
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._client_secret == "env-secret"


    def test_env_secret_overrides_config(self, patch_config, monkeypatch):
        patch_config(
            {
                "self_hosted": {
                    "issuer": _ISSUER,
                    "client_id": _CLIENT_ID,
                    "client_secret": "cfg-secret",
                }
            }
        )
        monkeypatch.setenv("HERMES_DASHBOARD_OIDC_CLIENT_SECRET", "env-secret")
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._client_secret == "env-secret"

    def test_empty_env_secret_does_not_shadow_config(self, patch_config, monkeypatch):
        patch_config(
            {
                "self_hosted": {
                    "issuer": _ISSUER,
                    "client_id": _CLIENT_ID,
                    "client_secret": "cfg-secret",
                }
            }
        )
        monkeypatch.setenv("HERMES_DASHBOARD_OIDC_CLIENT_SECRET", "")
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._client_secret == "cfg-secret"


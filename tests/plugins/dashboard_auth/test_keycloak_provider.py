"""Tests for the bundled Keycloak dashboard-auth plugin.

Keycloak reuses the generic self-hosted OIDC machinery end to end, so this suite
does not re-prove discovery/JWKS/grants (that lives in ``test_self_hosted_provider.py``).
It pins the three things the Keycloak provider *adds*:

1. Its own configuration surface: ``dashboard.oauth.keycloak.*`` +
   ``HERMES_DASHBOARD_KEYCLOAK_*`` env vars (env wins; skip reason when unset).
2. ``kc_idp_hint`` / ``prompt`` on the authorize URL — appended only when set,
   never clobbering the PKCE params, and absent by default (the shared-flow contract
   is unchanged for operators who want plain OIDC).
3. Provider identity: ``name == "keycloak"`` flows into ``Session.provider``, which
   is how the auth gate dispatches a session cookie back to this provider.

All HTTP is mocked: nothing here talks to a real Keycloak.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.parse
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import plugins.dashboard_auth.keycloak as kc_plugin
from hermes_cli.dashboard_auth import (
    InvalidCodeError,
    ProviderError,
    Session,
    assert_protocol_compliance,
)

# A typical realm issuer; discovery + jwks hang off this origin.
_ISSUER = "https://idp.example.com/realms/ebmedical"
_CLIENT_ID = "hermes-dashboard"

_DISCOVERY_DOC = {
    "issuer": _ISSUER,
    "authorization_endpoint": f"{_ISSUER}/protocol/openid-connect/auth",
    "token_endpoint": f"{_ISSUER}/protocol/openid-connect/token",
    "jwks_uri": f"{_ISSUER}/protocol/openid-connect/certs",
    "revocation_endpoint": f"{_ISSUER}/protocol/openid-connect/revoke",
}


@pytest.fixture(scope="module")
def rsa_keypair() -> dict[str, Any]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_numbers = key.public_key().public_numbers()

    def _b64url_uint(n: int) -> str:
        length = (n.bit_length() + 7) // 8
        return base64.urlsafe_b64encode(n.to_bytes(length, "big")).rstrip(b"=").decode()

    jwk = {
        "kty": "RSA",
        "use": "sig",
        "alg": "RS256",
        "kid": "test-key-1",
        "n": _b64url_uint(public_numbers.n),
        "e": _b64url_uint(public_numbers.e),
    }
    return {"private_pem": private_pem, "jwk": jwk, "kid": jwk["kid"]}


def _mint_id_token(
    rsa_keypair: dict[str, Any],
    *,
    iss: str = _ISSUER,
    aud: str = _CLIENT_ID,
    sub: str = "keycloak-subj-1",
    email: str | None = "alice@example.com",
    preferred_username: str | None = "alice",
    name: str | None = None,
    groups: Any = None,
    ttl_seconds: int = 900,
) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "iss": iss,
        "aud": aud,
        "sub": sub,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    if email is not None:
        claims["email"] = email
    if preferred_username is not None:
        claims["preferred_username"] = preferred_username
    if name is not None:
        claims["name"] = name
    if groups is not None:
        claims["groups"] = groups
    return jwt.encode(
        claims, rsa_keypair["private_pem"], algorithm="RS256",
        headers={"kid": rsa_keypair["kid"]},
    )


def _make_provider(
    rsa_keypair,
    *,
    scopes: str | None = None,
    idp_hint: str | None = None,
    prompt: str | None = None,
):
    """Construct a provider with realm discovery + JWKS stubbed (no network)."""
    kwargs: dict[str, Any] = {"issuer": _ISSUER, "client_id": _CLIENT_ID}
    if scopes is not None:
        kwargs["scopes"] = scopes
    if idp_hint is not None:
        kwargs["idp_hint"] = idp_hint
    if prompt is not None:
        kwargs["prompt"] = prompt
    p = kc_plugin.KeycloakDashboardAuthProvider(**kwargs)
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


def _authorize_params(redirect_url: str) -> dict[str, str]:
    query = urllib.parse.urlparse(redirect_url).query
    return dict(urllib.parse.parse_qsl(query))


# ---------------------------------------------------------------------------
# Identity + protocol
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_protocol_compliance(self):
        assert_protocol_compliance(kc_plugin.KeycloakDashboardAuthProvider)

    def test_provider_name_is_keycloak(self):
        assert kc_plugin.KeycloakDashboardAuthProvider.name == "keycloak"

    def test_requires_issuer(self):
        with pytest.raises(ValueError, match="issuer"):
            kc_plugin.KeycloakDashboardAuthProvider(issuer="", client_id=_CLIENT_ID)

    def test_rejects_cleartext_issuer(self):
        # Inherited from the OIDC base: a non-https issuer would ship codes in the clear.
        with pytest.raises(ProviderError, match="https"):
            kc_plugin.KeycloakDashboardAuthProvider(
                issuer="http://idp.example.com/realms/x", client_id=_CLIENT_ID
            )


# ---------------------------------------------------------------------------
# start_login: PKCE base + kc_idp_hint / prompt
# ---------------------------------------------------------------------------


class TestStartLogin:
    def test_plain_oidc_url_has_no_keycloak_extras(self, rsa_keypair):
        p = _make_provider(rsa_keypair)
        start = p.start_login(redirect_uri="https://hermes.example/auth/callback")
        params = _authorize_params(start.redirect_url)
        assert params["response_type"] == "code"
        assert params["client_id"] == _CLIENT_ID
        assert params["code_challenge_method"] == "S256"
        assert start.redirect_url.startswith(_DISCOVERY_DOC["authorization_endpoint"])
        # No hints configured → neither Keycloak param is present.
        assert "kc_idp_hint" not in params
        assert "prompt" not in params

    def test_idp_hint_and_prompt_appended(self, rsa_keypair):
        p = _make_provider(rsa_keypair, idp_hint="google", prompt="login")
        params = _authorize_params(
            p.start_login(redirect_uri="https://hermes.example/auth/callback").redirect_url
        )
        assert params["kc_idp_hint"] == "google"
        assert params["prompt"] == "login"
        # The shared PKCE params survive the rebuild.
        assert params["code_challenge_method"] == "S256"
        assert params["state"]

    def test_extras_do_not_clobber_pkce_state(self, rsa_keypair):
        p = _make_provider(rsa_keypair, idp_hint="kerberos")
        start = p.start_login(redirect_uri="https://hermes.example/auth/callback")
        params = _authorize_params(start.redirect_url)
        cookie = dict(
            seg.split("=", 1)
            for seg in start.cookie_payload["hermes_session_pkce"].split(";")
            if "=" in seg
        )
        # state echoed in the URL must equal the one in the cookie, exactly once.
        assert params["state"] == cookie["state"]
        pairs = urllib.parse.parse_qsl(urllib.parse.urlparse(start.redirect_url).query)
        assert [k for k, _ in pairs].count("state") == 1
        assert [k for k, _ in pairs].count("kc_idp_hint") == 1

    def test_empty_hint_and_prompt_are_omitted(self, rsa_keypair):
        p = _make_provider(rsa_keypair, idp_hint="", prompt="")
        params = _authorize_params(
            p.start_login(redirect_uri="https://hermes.example/auth/callback").redirect_url
        )
        assert "kc_idp_hint" not in params
        assert "prompt" not in params


# ---------------------------------------------------------------------------
# complete_login / verify_session: Keycloak claim mapping
# ---------------------------------------------------------------------------


class TestSessionMapping:
    def test_complete_login_provider_is_keycloak(self, rsa_keypair):
        p = _make_provider(rsa_keypair)
        id_token = _mint_id_token(rsa_keypair)
        # A distinct lifetime keeps the two tokens byte-different, so the assertion below can
        # tell which one was stored.
        access_token = _mint_id_token(rsa_keypair, ttl_seconds=901)
        resp = _mock_post(
            200,
            {"access_token": access_token, "id_token": id_token, "token_type": "Bearer",
             "refresh_token": "rt_1"},
        )
        # httpx is shared with the base module; patch it there where exchange_token reads it.
        with patch("plugins.dashboard_auth.self_hosted.httpx.post", return_value=resp):
            session = p.complete_login(
                code="abc", state="s", code_verifier="vfy",
                redirect_uri="https://hermes.example/auth/callback",
            )
        assert isinstance(session, Session)
        assert session.provider == "keycloak"
        assert session.user_id == "keycloak-subj-1"
        # Keycloak's preferred_username surfaces as display_name when name is absent.
        assert session.display_name == "alice"
        # The OAuth *access* token is the session credential. It is also the one minted with the
        # resource-server audience, so this single field serves both the dashboard cookie and
        # the downstream MCP call; the ID token is verified for flow integrity but not stored.
        assert session.access_token == access_token
        assert session.access_token != id_token

    def test_verify_session_round_trips(self, rsa_keypair):
        p = _make_provider(rsa_keypair)
        id_token = _mint_id_token(rsa_keypair, name="Alice Example")
        session = p.verify_session(access_token=id_token)
        assert session is not None
        assert session.provider == "keycloak"
        assert session.display_name == "Alice Example"

    def test_expired_token_returns_none(self, rsa_keypair):
        p = _make_provider(rsa_keypair)
        # Past the 60s clock-skew leeway, so exp really has lapsed.
        id_token = _mint_id_token(rsa_keypair, ttl_seconds=-120)
        assert p.verify_session(access_token=id_token) is None

    def test_wrong_audience_returns_none(self, rsa_keypair):
        # Shared JwtOAuthProvider contract (see test_self_hosted_provider): a wrong-audience
        # token is a confirmed denial, not an outage. The IdP answered (JWKS fetched, token
        # parsed) but the token is not ours, so ``verify_session`` returns None and the
        # middleware refreshes / re-logins instead of answering a misleading 503
        # "provider unreachable" (#94558, #47815).
        p = _make_provider(rsa_keypair)
        id_token = _mint_id_token(rsa_keypair, aud="someone-else")
        assert p.verify_session(access_token=id_token) is None

    def test_foreign_issuer_returns_none(self, rsa_keypair):
        # A token minted by a different realm must not authenticate against this one, and the
        # iss mismatch is likewise a denial (None), not config drift surfaced as 503.
        p = _make_provider(rsa_keypair)
        id_token = _mint_id_token(rsa_keypair, iss="https://evil.example/realms/x")
        assert p.verify_session(access_token=id_token) is None


# ---------------------------------------------------------------------------
# Plugin entry point: env + config.yaml precedence (Keycloak's own surface)
# ---------------------------------------------------------------------------


class TestPluginRegister:
    @pytest.fixture(autouse=True)
    def clear_env(self, monkeypatch):
        for var in (
            "HERMES_DASHBOARD_KEYCLOAK_ISSUER",
            "HERMES_DASHBOARD_KEYCLOAK_CLIENT_ID",
            "HERMES_DASHBOARD_KEYCLOAK_SCOPES",
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
        kc_plugin.register(ctx)
        ctx.register_dashboard_auth_provider.assert_not_called()
        assert "HERMES_DASHBOARD_KEYCLOAK_ISSUER" in kc_plugin.LAST_SKIP_REASON

    def test_registers_from_config(self, patch_config):
        patch_config({"keycloak": {"issuer": _ISSUER, "client_id": _CLIENT_ID}})
        ctx = MagicMock()
        kc_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert isinstance(registered, kc_plugin.KeycloakDashboardAuthProvider)
        assert registered._issuer == _ISSUER
        assert registered.name == "keycloak"
        assert kc_plugin.LAST_SKIP_REASON == ""

    def test_registers_from_env(self, patch_config, monkeypatch):
        patch_config(None)
        monkeypatch.setenv("HERMES_DASHBOARD_KEYCLOAK_ISSUER", _ISSUER)
        monkeypatch.setenv("HERMES_DASHBOARD_KEYCLOAK_CLIENT_ID", _CLIENT_ID)
        ctx = MagicMock()
        kc_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._client_id == _CLIENT_ID
        assert registered._scopes == "openid profile email"

    def test_env_overrides_config(self, patch_config, monkeypatch):
        patch_config({"keycloak": {"issuer": "https://cfg.example", "client_id": "cfg-client"}})
        monkeypatch.setenv("HERMES_DASHBOARD_KEYCLOAK_ISSUER", _ISSUER)
        monkeypatch.setenv("HERMES_DASHBOARD_KEYCLOAK_CLIENT_ID", _CLIENT_ID)
        ctx = MagicMock()
        kc_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._issuer == _ISSUER
        assert registered._client_id == _CLIENT_ID

    def test_idp_hint_and_prompt_from_config(self, patch_config):
        patch_config({
            "keycloak": {
                "issuer": _ISSUER, "client_id": _CLIENT_ID,
                "idp_hint": "google", "prompt": "login",
            }
        })
        ctx = MagicMock()
        kc_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._idp_hint == "google"
        assert registered._prompt == "login"

    def test_public_client_has_no_secret(self, patch_config):
        # Keycloak is a public (PKCE-only) client by design: the provider never
        # carries a client secret, whatever the config surface looks like.
        patch_config({"keycloak": {"issuer": _ISSUER, "client_id": _CLIENT_ID}})
        ctx = MagicMock()
        kc_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._client_secret == ""

    def test_config_load_failure_falls_through(self, monkeypatch):
        def _broken():
            raise OSError("unreadable")

        monkeypatch.setattr("hermes_cli.config.load_config", _broken)
        ctx = MagicMock()
        kc_plugin.register(ctx)  # must not raise
        ctx.register_dashboard_auth_provider.assert_not_called()

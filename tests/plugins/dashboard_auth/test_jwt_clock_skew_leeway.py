"""Clock-skew leeway for dashboard-auth JWT verification (#47815).

PyJWT's default ``leeway=0`` while ``iat`` is required means any host whose clock
lags the issuer's — different VM/container, NTP jitter — fails verification with
``ImmatureSignatureError``. Invariants covered here, one per acceptance case:

1. A future-``iat`` token (clock skew) verifies with the default leeway — for the
   shared ``verify_jwt``, the self-hosted OIDC provider and the Nous provider.
2. ``ImmatureSignatureError`` beyond the leeway is a *validation* failure
   (``InvalidCodeError`` → ``verify_session`` returns ``None`` → the middleware
   retries/refreshes), NOT a ``ProviderError`` (which surfaces as 503
   "provider unreachable" and historically discarded rotated refresh tokens).
3. The leeway is configurable via config.yaml for both providers, and a typo
   (unparseable / negative / non-finite) fails closed to 0 — it can never widen
   the window.

All HTTP and JWKS are stubbed; nothing here talks to a real IdP or Portal.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict
from unittest.mock import MagicMock

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import plugins.dashboard_auth._shared as shared
import plugins.dashboard_auth.nous as nous_plugin
import plugins.dashboard_auth.self_hosted as oidc_plugin
from hermes_cli.dashboard_auth import InvalidCodeError, ProviderError

_ISSUER = "https://auth.example.com/application/o/hermes"
_CLIENT_ID = "hermes-dashboard"
_PORTAL = "https://portal.example.com"
_DISCOVERY_DOC = {
    "issuer": _ISSUER,
    "authorization_endpoint": f"{_ISSUER}/authorize",
    "token_endpoint": f"{_ISSUER}/token",
    "jwks_uri": f"{_ISSUER}/jwks",
}


@pytest.fixture(scope="module")
def rsa_keypair() -> dict[str, Any]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_key = serialization.load_pem_private_key(
        private_pem.encode(), password=None
    ).public_key()
    return {"private_pem": private_pem, "public_key": public_key, "kid": "test-key-1"}


def _fake_jwks_client(rsa_keypair) -> MagicMock:
    fake_key = MagicMock()
    fake_key.key = rsa_keypair["public_key"]
    client = MagicMock()
    client.get_signing_key_from_jwt.return_value = fake_key
    return client


def _mint_token(
    rsa_keypair: dict[str, Any],
    *,
    iss: str,
    aud: str,
    iat_offset: int = 0,
    ttl_seconds: int = 900,
) -> str:
    """A minimal signed JWT whose ``iat`` is ``iat_offset`` seconds in the future."""
    now = int(time.time())
    claims = {"iss": iss, "aud": aud, "sub": "usr_abc", "iat": now + iat_offset, "exp": now + ttl_seconds}
    return pyjwt.encode(claims, rsa_keypair["private_pem"], algorithm="RS256", headers={"kid": rsa_keypair["kid"]})


def _self_hosted_provider(rsa_keypair, *, leeway: float | None = None) -> Any:
    kwargs: dict[str, Any] = {"issuer": _ISSUER, "client_id": _CLIENT_ID}
    if leeway is not None:
        kwargs["id_token_leeway"] = leeway
    p = oidc_plugin.SelfHostedOIDCProvider(**kwargs)
    p._discovery = dict(_DISCOVERY_DOC)
    p._discovery_fetched_at = time.time()
    p._jwks_client = _fake_jwks_client(rsa_keypair)
    return p


def _nous_provider(rsa_keypair, *, leeway: float | None = None) -> Any:
    kwargs: dict[str, Any] = {"client_id": "agent:inst123", "portal_url": _PORTAL}
    if leeway is not None:
        kwargs["token_leeway"] = leeway
    p = nous_plugin.NousDashboardAuthProvider(**kwargs)
    p._jwks_client = _fake_jwks_client(rsa_keypair)
    return p


# ---------------------------------------------------------------------------
# 1. Future-iat tokens verify with the default leeway (clock skew tolerated)
# ---------------------------------------------------------------------------


class TestDefaultLeewayAbsorbsClockSkew:
    """A token whose iat is seconds ahead of the verifier (issuer clock skew)
    is the normal case for hosts without tight time-sync — RFC 7519 §4.1.4-4.1.6
    allow "some small leeway, usually no more than a few minutes"."""

    SKEW = 30  # seconds of issuer-ahead skew, well inside the 60s default

    def test_shared_verify_jwt(self, rsa_keypair):
        token = _mint_token(rsa_keypair, iss=_ISSUER, aud=_CLIENT_ID, iat_offset=self.SKEW)
        claims = shared.verify_jwt(
            token, _fake_jwks_client(rsa_keypair), algorithms=["RS256"],
            audience=_CLIENT_ID, issuer=_ISSUER, label="ID token")
        assert claims["sub"] == "usr_abc"

    def test_self_hosted_verify_session(self, rsa_keypair):
        provider = _self_hosted_provider(rsa_keypair)
        token = _mint_token(rsa_keypair, iss=_ISSUER, aud=_CLIENT_ID, iat_offset=self.SKEW)
        session = provider.verify_session(access_token=token)
        assert session is not None
        assert session.user_id == "usr_abc"

    def test_nous_verify_session(self, rsa_keypair):
        provider = _nous_provider(rsa_keypair)
        token = _mint_token(rsa_keypair, iss=_PORTAL, aud="agent:inst123", iat_offset=self.SKEW)
        session = provider.verify_session(access_token=token)
        assert session is not None
        assert session.user_id == "usr_abc"


# ---------------------------------------------------------------------------
# 2. ImmatureSignatureError beyond the leeway is a validation failure, not 503
# ---------------------------------------------------------------------------


class TestImmatureSignatureClassification:
    """Skew larger than the leeway still fails — but as a *token* failure
    (InvalidCodeError → verify_session returns None → middleware refreshes /
    401s), never as ProviderError ("provider unreachable" / 503), which used to
    discard already-rotated refresh tokens."""

    HUGE_SKEW = 3600  # an hour ahead: far beyond any sane leeway

    def test_shared_verify_jwt_raises_invalid_code(self, rsa_keypair):
        token = _mint_token(rsa_keypair, iss=_ISSUER, aud=_CLIENT_ID, iat_offset=self.HUGE_SKEW)
        with pytest.raises(InvalidCodeError, match="not yet valid"):
            shared.verify_jwt(
                token, _fake_jwks_client(rsa_keypair), algorithms=["RS256"],
                audience=_CLIENT_ID, issuer=_ISSUER, label="ID token")

    def test_self_hosted_verify_session_returns_none(self, rsa_keypair):
        provider = _self_hosted_provider(rsa_keypair)
        token = _mint_token(rsa_keypair, iss=_ISSUER, aud=_CLIENT_ID, iat_offset=self.HUGE_SKEW)
        assert provider.verify_session(access_token=token) is None

    def test_nous_verify_session_returns_none(self, rsa_keypair):
        provider = _nous_provider(rsa_keypair)
        token = _mint_token(rsa_keypair, iss=_PORTAL, aud="agent:inst123", iat_offset=self.HUGE_SKEW)
        assert provider.verify_session(access_token=token) is None


# ---------------------------------------------------------------------------
# 3. parse_leeway fails closed
# ---------------------------------------------------------------------------


class TestParseLeeway:
    def test_empty_uses_default(self):
        assert shared.parse_leeway(None) == shared.DEFAULT_TOKEN_LEEWAY_SECONDS
        assert shared.parse_leeway("") == shared.DEFAULT_TOKEN_LEEWAY_SECONDS

    @pytest.mark.parametrize("bad", ["not-a-number", "-5", "nan", "inf", "-inf", float("nan"), float("inf")])
    def test_invalid_fails_closed_to_zero(self, bad):
        assert shared.parse_leeway(bad) == 0.0

    def test_valid_values(self):
        assert shared.parse_leeway("0") == 0.0
        assert shared.parse_leeway(30) == 30.0
        assert shared.parse_leeway("120.5") == 120.5

    def test_explicit_zero_is_respected(self):
        # leeway=0 keeps the strict pre-fix behaviour for operators who want it
        assert shared.parse_leeway(0) == 0.0

    def test_finite_check(self):
        assert not math.isinf(shared.parse_leeway("60"))


# ---------------------------------------------------------------------------
# 4. Config plumbing (config.yaml only — no new env vars)
# ---------------------------------------------------------------------------


def _patch_oauth_config(monkeypatch, oauth_block):
    cfg = {} if oauth_block is None else {"dashboard": {"oauth": oauth_block}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)


class TestSelfHostedConfig:
    @pytest.fixture(autouse=True)
    def clear_env(self, monkeypatch):
        for var in ("HERMES_DASHBOARD_OIDC_ISSUER", "HERMES_DASHBOARD_OIDC_CLIENT_ID"):
            monkeypatch.delenv(var, raising=False)

    def test_leeway_from_config_yaml(self, monkeypatch, rsa_keypair):
        _patch_oauth_config(monkeypatch, {"self_hosted": {
            "issuer": _ISSUER, "client_id": _CLIENT_ID, "id_token_leeway": 300}})
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._id_token_leeway == 300.0

    def test_unset_leeway_uses_default(self, monkeypatch):
        _patch_oauth_config(monkeypatch, {"self_hosted": {"issuer": _ISSUER, "client_id": _CLIENT_ID}})
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._id_token_leeway == shared.DEFAULT_TOKEN_LEEWAY_SECONDS

    def test_invalid_config_leeway_fails_closed(self, monkeypatch, rsa_keypair):
        _patch_oauth_config(monkeypatch, {"self_hosted": {
            "issuer": _ISSUER, "client_id": _CLIENT_ID, "id_token_leeway": "garbage"}})
        ctx = MagicMock()
        oidc_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._id_token_leeway == 0.0

    def test_explicit_leeway_bounds_tolerance(self, rsa_keypair):
        # leeway=0: a clearly-future iat is rejected (strict mode still available)
        provider = _self_hosted_provider(rsa_keypair, leeway=0)
        token = _mint_token(rsa_keypair, iss=_ISSUER, aud=_CLIENT_ID, iat_offset=60)
        assert provider.verify_session(access_token=token) is None


class TestNousConfig:
    @pytest.fixture(autouse=True)
    def clear_env(self, monkeypatch):
        monkeypatch.delenv("HERMES_DASHBOARD_OAUTH_CLIENT_ID", raising=False)

    def test_leeway_from_config_yaml(self, monkeypatch):
        _patch_oauth_config(monkeypatch, {"client_id": "agent:inst123", "token_leeway": 120})
        ctx = MagicMock()
        nous_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._token_leeway == 120.0

    def test_unset_leeway_uses_default(self, monkeypatch):
        _patch_oauth_config(monkeypatch, {"client_id": "agent:inst123"})
        ctx = MagicMock()
        nous_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._token_leeway == shared.DEFAULT_TOKEN_LEEWAY_SECONDS

    def test_invalid_config_leeway_fails_closed(self, monkeypatch):
        _patch_oauth_config(monkeypatch, {"client_id": "agent:inst123", "token_leeway": -3})
        ctx = MagicMock()
        nous_plugin.register(ctx)
        registered = ctx.register_dashboard_auth_provider.call_args.args[0]
        assert registered._token_leeway == 0.0

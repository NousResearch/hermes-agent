"""#94558 — a non-JWT bearer must not be reported as "Auth provider unreachable".

Hosted agents answered every opaque/peer bearer on the gated API with a fast
HTTP 503 ``{"detail": "Auth provider 'nous' unreachable"}`` while Portal was
perfectly healthy: ``NousDashboardAuthProvider._verify_jwt`` folded *every*
``PyJWKClient`` failure — including ``DecodeError('Not enough segments')`` for
a token that is not a JWT at all — into ``ProviderError``. Only a transport
failure fetching the JWKS is "unreachable"; anything else means "not my
token" (``verify_session`` -> None -> 401 / next provider).

The same 503 later came back for well-signed Portal JWTs this instance does not accept: the
user's Portal CLI token (``aud=hermes-cli:hermes-cli``), another instance's ``aud`` or
``agent_instance_id``, and a ``kid`` the reachable JWKS does not hold.

Real ``NousDashboardAuthProvider`` + real ``SelfHostedOIDCProvider`` JWKS path,
a real local HTTP JWKS server (reachable case) or a closed port (unreachable),
and the real gated web_server app for the HTTP-level assertion.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import (
    InvalidCodeError,
    ProviderError,
    classify_jwks_lookup_error,
    clear_providers,
    register_provider,
)
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE, SESSION_RT_COOKIE
import plugins.dashboard_auth.nous as nous_plugin

OPAQUE_PEER_KEY = "hk_live_opaque_peer_key_0123456789abcdef"
# Well-formed RS256 JWT header with an unknown kid, bogus payload/signature.
FOREIGN_KID_JWT = "eyJhbGciOiJSUzI1NiIsImtpZCI6Inp6eiJ9.e30.sig"


_INSTANCE = "test-instance"
_KID = "portal-k1"


@contextmanager
def _serve_jwks(keys: list[dict]):
    """A reachable JWKS endpoint on loopback publishing ``keys``; yields its base URL."""

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"keys": keys}).encode())

        def log_message(self, *a):  # silence
            pass

    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture(scope="module")
def empty_jwks_server():
    """A reachable JWKS endpoint that knows no keys."""
    with _serve_jwks([]) as url:
        yield url


@pytest.fixture(scope="module")
def portal():
    """A reachable Portal JWKS publishing one signing key; yields ``(url, private_key)``."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = {**json.loads(RSAAlgorithm.to_jwk(private.public_key())),
           "kid": _KID, "use": "sig", "alg": "RS256"}
    with _serve_jwks([jwk]) as url:
        yield url, private


def _mint(portal, *, aud=f"agent:{_INSTANCE}", kid=_KID, **extra) -> str:
    url, private = portal
    now = int(time.time())
    claims = {"iss": url, "aud": aud, "sub": "usr_abc", "iat": now, "exp": now + 900,
              "oauth_contract_version": 1, "scope": "agent_dashboard:access", **extra}
    return jwt.encode(claims, private, algorithm="RS256", headers={"kid": kid})


def _nous(portal_url: str) -> nous_plugin.NousDashboardAuthProvider:
    return nous_plugin.NousDashboardAuthProvider(client_id=f"agent:{_INSTANCE}", portal_url=portal_url)


# ── classifier ────────────────────────────────────────────────────────────

def test_classifier_maps_transport_failure_to_provider_error():
    exc = jwt.PyJWKClientConnectionError("Fail to fetch data from the url")
    assert isinstance(classify_jwks_lookup_error(exc), ProviderError)


@pytest.mark.parametrize(
    "exc",
    [
        jwt.DecodeError("Not enough segments"),
        jwt.PyJWKSetError("The JWK Set did not contain any keys"),
        jwt.InvalidTokenError("bad"),
    ],
)
def test_classifier_maps_unverifiable_token_to_invalid_code(exc):
    assert isinstance(classify_jwks_lookup_error(exc), InvalidCodeError)


def test_classifier_keeps_bare_jwk_client_error_as_provider_fault():
    assert isinstance(classify_jwks_lookup_error(jwt.PyJWKClientError("weird JWKS shape")), ProviderError)


# ── Nous provider ─────────────────────────────────────────────────────────

def test_opaque_bearer_with_healthy_portal_is_not_unreachable(empty_jwks_server):
    provider = _nous(empty_jwks_server)
    assert provider.verify_session(access_token=OPAQUE_PEER_KEY) is None


def test_foreign_kid_jwt_with_healthy_portal_is_not_unreachable(empty_jwks_server):
    provider = _nous(empty_jwks_server)
    assert provider.verify_session(access_token=FOREIGN_KID_JWT) is None


def test_unknown_kid_against_populated_jwks_is_not_unreachable(portal):
    """The JWKS was fetched fine; it just has no key for this kid (rotated out, or a forged
    header). That is an unverifiable token, not an unreachable provider."""
    provider = _nous(portal[0])
    assert provider.verify_session(access_token=_mint(portal, kid="rotated-out")) is None


def test_real_jwt_with_unreachable_portal_still_raises_provider_error():
    provider = _nous("http://127.0.0.1:9")  # discard port: connection refused
    with pytest.raises(ProviderError):
        provider.verify_session(access_token=FOREIGN_KID_JWT)


def test_opaque_bearer_with_unreachable_portal_is_still_just_not_ours():
    """No network call is even needed to know an opaque string is not our JWT."""
    provider = _nous("http://127.0.0.1:9")
    assert provider.verify_session(access_token=OPAQUE_PEER_KEY) is None


# ── self-hosted OIDC provider (sibling site of the same hunk) ──────────────

def test_self_hosted_provider_shares_the_classification(empty_jwks_server, monkeypatch):
    import plugins.dashboard_auth.self_hosted as sh

    provider = object.__new__(sh.SelfHostedOIDCProvider)
    provider._jwks_client = None
    provider._client_id = "hermes"
    monkeypatch.setattr(
        provider, "_get_discovery",
        lambda: {"jwks_uri": f"{empty_jwks_server}/jwks", "issuer": empty_jwks_server},
    )
    with pytest.raises(InvalidCodeError):
        provider._verify_id_token(OPAQUE_PEER_KEY)


# ── HTTP level: the gated API answers 401, not 503 ────────────────────────

@contextmanager
def _gated_app(portal_url: str):
    """The real gated dashboard app with only the Nous provider, pointed at ``portal_url``."""
    clear_providers()
    keys = ("bound_host", "bound_port", "auth_required")
    prev = {k: getattr(web_server.app.state, k, None) for k in keys}
    web_server.app.state.bound_host = "agent.example.test"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    register_provider(_nous(portal_url))
    try:
        yield TestClient(web_server.app, base_url="https://agent.example.test")
    finally:
        clear_providers()
        for k, v in prev.items():
            setattr(web_server.app.state, k, v)


@pytest.fixture
def _gated_nous(empty_jwks_server):
    with _gated_app(empty_jwks_server) as client:
        yield client


@pytest.fixture
def gated_portal(portal):
    with _gated_app(portal[0]) as client:
        yield client


def test_gated_api_rejects_opaque_bearer_with_401_not_503(_gated_nous):
    r = _gated_nous.get("/api/auth/me", headers={"Authorization": f"Bearer {OPAQUE_PEER_KEY}"})
    assert r.status_code != 503, r.text
    assert r.status_code == 401
    assert "unreachable" not in r.text.lower()


def test_gated_api_rejects_opaque_cookie_with_401_not_503(_gated_nous):
    _gated_nous.cookies.set(SESSION_AT_COOKIE, OPAQUE_PEER_KEY)
    r = _gated_nous.get("/api/auth/me")
    assert r.status_code != 503, r.text
    assert "unreachable" not in r.text.lower()


def _bearer(client, token):
    return client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})


def test_gated_api_accepts_own_portal_token(gated_portal, portal):
    assert _bearer(gated_portal, _mint(portal)).status_code == 200


@pytest.mark.parametrize("token_kwargs", [
    pytest.param({"aud": "hermes-cli:hermes-cli"}, id="portal-cli-token"),
    pytest.param({"aud": "agent:other-instance"}, id="other-instance-aud"),
    pytest.param({"agent_instance_id": "other-instance"}, id="other-instance-claim"),
    pytest.param({"kid": "rotated-out"}, id="unknown-kid"),
])
def test_gated_api_rejects_foreign_portal_jwt_bearer_with_401_not_503(gated_portal, portal, token_kwargs):
    r = _bearer(gated_portal, _mint(portal, **token_kwargs))
    assert r.status_code == 401, r.text
    assert r.json()["reason"] == "invalid_or_expired_session"


def test_gated_api_foreign_portal_jwt_cookie_is_401_not_503(gated_portal, portal):
    """Cookie path: a foreign access-token cookie with no refresh token is a signed-out 401."""
    gated_portal.cookies.set(f"__Host-{SESSION_AT_COOKIE}", _mint(portal, aud="hermes-cli:hermes-cli"))
    r = gated_portal.get("/api/auth/me")
    assert r.status_code == 401, r.text
    assert r.json()["reason"] == "invalid_or_expired_session"


def test_gated_api_refresh_minting_foreign_token_is_401_not_503(gated_portal, portal):
    """Cookie path with a refresh token: the rotated token Portal hands back does not verify
    here, so the provider rejects the RT (signed-out 401), never a 503 or an uncaught 500."""
    foreign = _mint(portal, aud="hermes-cli:hermes-cli")
    grant = MagicMock(status_code=200, headers={"content-type": "application/json"})
    grant.json.return_value = {"access_token": foreign, "token_type": "Bearer", "refresh_token": "rt2"}
    # The __Host- names the server sets over HTTPS (bare names are not read there, #136772).
    gated_portal.cookies.set(f"__Host-{SESSION_AT_COOKIE}", foreign)
    gated_portal.cookies.set(f"__Host-{SESSION_RT_COOKIE}", "rt-old")
    with patch("plugins.dashboard_auth._shared.httpx.post", return_value=grant) as post:
        r = gated_portal.get("/api/auth/me")
    assert post.called
    assert r.status_code == 401, r.text
    assert r.json()["reason"] == "invalid_or_expired_session"


def test_forged_alg_with_published_kid_is_rejected_without_a_drift_warning(portal, caplog):
    """The drift WARNING is reserved for aud/iss mismatches, which PyJWT reports only after the
    signature verified. A forged token naming the published kid with another alg is rejected
    before any signature check: a 401, and no attacker-sized log line."""
    forged = jwt.encode({"iss": "x" * 4000, "aud": "y"}, "k" * 32, algorithm="HS256", headers={"kid": _KID})
    provider = _nous(portal[0])
    with caplog.at_level(logging.WARNING, logger="plugins.dashboard_auth._shared"):
        assert provider.verify_session(access_token=forged) is None
    assert "verification failed" not in caplog.text


def test_gated_api_unreachable_jwks_is_still_503(portal):
    """The outage contract is kept: a JWKS that cannot be fetched stays a 503."""
    with _gated_app("http://127.0.0.1:9") as client:  # discard port: connection refused
        assert _bearer(client, _mint(portal)).status_code == 503

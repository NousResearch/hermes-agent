"""#94558 — a non-JWT bearer must not be reported as "Auth provider unreachable".

Hosted agents answered every opaque/peer bearer on the gated API with a fast
HTTP 503 ``{"detail": "Auth provider 'nous' unreachable"}`` while Portal was
perfectly healthy: ``NousDashboardAuthProvider._verify_jwt`` folded *every*
``PyJWKClient`` failure — including ``DecodeError('Not enough segments')`` for
a token that is not a JWT at all — into ``ProviderError``. Only a transport
failure fetching the JWKS is "unreachable"; anything else means "not my
token" (``verify_session`` -> None -> 401 / next provider).

Real ``NousDashboardAuthProvider`` + real ``SelfHostedOIDCProvider`` JWKS path,
a real local HTTP JWKS server (reachable case) or a closed port (unreachable),
and the real gated web_server app for the HTTP-level assertion.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import jwt
import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import (
    InvalidCodeError,
    ProviderError,
    classify_jwks_lookup_error,
    clear_providers,
    register_provider,
)
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE
import plugins.dashboard_auth.nous as nous_plugin

OPAQUE_PEER_KEY = "hk_live_opaque_peer_key_0123456789abcdef"
# Well-formed RS256 JWT header with an unknown kid, bogus payload/signature.
FOREIGN_KID_JWT = "eyJhbGciOiJSUzI1NiIsImtpZCI6Inp6eiJ9.e30.sig"


@pytest.fixture(scope="module")
def empty_jwks_server():
    """A reachable JWKS endpoint that knows no keys."""

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"keys": []}).encode())

        def log_message(self, *a):  # silence
            pass

    srv = HTTPServer(("127.0.0.1", 0), _H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _nous(portal_url: str) -> nous_plugin.NousDashboardAuthProvider:
    return nous_plugin.NousDashboardAuthProvider(client_id="agent:test-instance", portal_url=portal_url)


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

@pytest.fixture
def _gated_nous(empty_jwks_server):
    clear_providers()
    prev = {k: getattr(web_server.app.state, k, None) for k in ("bound_host", "bound_port", "auth_required")}
    web_server.app.state.bound_host = "agent.example.test"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    register_provider(_nous(empty_jwks_server))
    yield TestClient(web_server.app, base_url="https://agent.example.test")
    clear_providers()
    for k, v in prev.items():
        setattr(web_server.app.state, k, v)


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

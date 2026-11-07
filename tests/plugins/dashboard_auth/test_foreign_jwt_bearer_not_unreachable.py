"""A well-signed Portal JWT minted for another client is a 401, not ``503 unreachable``.

Hosted dashboards answered ``Authorization: Bearer <the user's Nous Portal CLI token>`` (a real,
unexpired Portal RS256 JWT with ``aud=hermes-cli:hermes-cli``) with ``503 {"detail": "Auth
provider 'nous' unreachable"}`` on every gated route while Portal was up: ``verify_jwt`` mapped
``InvalidAudienceError`` (and every other claim failure but expiry) to ``ProviderError``. The
same 503 came back for a JWT whose ``kid`` the published JWKS does not hold, and for a token
carrying another instance's ``agent_instance_id``. Only an unreachable JWKS is an outage.

Real ``NousDashboardAuthProvider``, a real local JWKS server publishing the signing key, and the
real gated ``web_server`` app.
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
import plugins.dashboard_auth.nous as nous_plugin

_KID = "portal-k1"
_INSTANCE = "inst123"


@pytest.fixture(scope="module")
def portal():
    """A reachable Portal JWKS that publishes one signing key; yields ``(url, private_key)``."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = {**json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private.public_key())),
           "kid": _KID, "use": "sig", "alg": "RS256"}

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"keys": [jwk]}).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", private
    srv.shutdown()


def _mint(portal, *, aud=f"agent:{_INSTANCE}", kid=_KID, **extra) -> str:
    url, private = portal
    now = int(time.time())
    claims = {"iss": url, "aud": aud, "sub": "usr_abc", "iat": now, "exp": now + 900,
              "oauth_contract_version": 1, **extra}
    return jwt.encode(claims, private, algorithm="RS256", headers={"kid": kid})


@contextmanager
def _gated_app(portal_url: str):
    """The real gated dashboard app with only the Nous provider, pointed at ``portal_url``."""
    clear_providers()
    keys = ("bound_host", "bound_port", "auth_required")
    prev = {k: getattr(web_server.app.state, k, None) for k in keys}
    web_server.app.state.bound_host = "agent.example.test"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    register_provider(nous_plugin.NousDashboardAuthProvider(
        client_id=f"agent:{_INSTANCE}", portal_url=portal_url))
    try:
        yield TestClient(web_server.app, base_url="https://agent.example.test")
    finally:
        clear_providers()
        for k, v in prev.items():
            setattr(web_server.app.state, k, v)


@pytest.fixture
def gated(portal):
    with _gated_app(portal[0]) as client:
        yield client


def _get(client, token):
    return client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})


def test_own_token_is_accepted(gated, portal):
    r = _get(gated, _mint(portal))
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("token_kwargs", [
    pytest.param({"aud": "hermes-cli:hermes-cli"}, id="portal-cli-token"),
    pytest.param({"aud": "agent:other-instance"}, id="other-instance-aud"),
    pytest.param({"agent_instance_id": "other-instance"}, id="other-instance-claim"),
    pytest.param({"kid": "rotated-out"}, id="unknown-kid"),
])
def test_foreign_portal_jwt_bearer_is_401_not_503(gated, portal, token_kwargs):
    r = _get(gated, _mint(portal, **token_kwargs))
    assert r.status_code == 401, r.text
    assert r.json()["reason"] == "invalid_or_expired_session"


def test_unreachable_jwks_is_still_503(portal):
    """The outage contract the fix keeps: a JWKS that cannot be fetched stays a 503."""
    with _gated_app("http://127.0.0.1:9") as client:  # discard port: connection refused
        r = _get(client, _mint(portal))
    assert r.status_code == 503, r.text

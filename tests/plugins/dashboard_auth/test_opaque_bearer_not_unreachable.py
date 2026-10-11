"""KEN-333 JWKS verification semantics using the real PyJWKClient HTTP path."""
from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.testclient import TestClient

import plugins.dashboard_auth.nous as nous_plugin
import plugins.dashboard_auth.self_hosted as self_hosted_plugin
from hermes_cli import web_server
from hermes_cli.dashboard_auth import ProviderError, clear_providers, register_provider
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE

_CLIENT_ID = "agent:ken333"


def _b64url_uint(value: int) -> str:
    width = (value.bit_length() + 7) // 8
    return base64.urlsafe_b64encode(value.to_bytes(width, "big")).rstrip(b"=").decode()


@pytest.fixture(scope="module")
def signing_material() -> dict[str, Any]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = key.public_key().public_numbers()
    return {
        "private_pem": key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        "jwk": {
            "kty": "RSA", "use": "sig", "alg": "RS256", "kid": "portal-key",
            "n": _b64url_uint(public.n), "e": _b64url_uint(public.e),
        },
    }


class _JWKSHandler(BaseHTTPRequestHandler):
    response: object = {"keys": []}

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        payload = self.response
        if isinstance(payload, str):
            self.wfile.write(payload.encode())
        else:
            self.wfile.write(json.dumps(payload).encode())

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def jwks_server(signing_material):
    _JWKSHandler.response = {"keys": [signing_material["jwk"]]}
    server = HTTPServer(("127.0.0.1", 0), _JWKSHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", _JWKSHandler
    finally:
        server.shutdown()
        thread.join()


def _token(
    material: dict[str, Any], *, kid: object = "portal-key", issuer: str,
    audience: str = _CLIENT_ID, expired: bool = False,
) -> str:
    now = int(time.time())
    headers = {} if kid == "absent" else {"kid": "portal-key" if kid is None else kid}
    token = jwt.encode(
        {
            "iss": issuer, "aud": audience, "sub": "user-1", "iat": now,
            "exp": now - 1 if expired else now + 300,
            "agent_instance_id": "ken333", "oauth_contract_version": 1,
        },
        material["private_pem"], algorithm="RS256", headers=headers,
    )
    if kid is None:
        # PyJWT correctly refuses to sign a null kid. The signature is
        # deliberately stale after this header-only negative probe; lookup
        # must reject it before signature verification.
        header = base64.urlsafe_b64encode(
            json.dumps({"alg": "RS256", "typ": "JWT", "kid": None}).encode()
        ).rstrip(b"=").decode()
        return f"{header}.{token.split('.', 1)[1]}"
    return token


def _nous(portal_url: str) -> nous_plugin.NousDashboardAuthProvider:
    return nous_plugin.NousDashboardAuthProvider(client_id=_CLIENT_ID, portal_url=portal_url)


def _self_hosted(issuer: str) -> self_hosted_plugin.SelfHostedOIDCProvider:
    provider = self_hosted_plugin.SelfHostedOIDCProvider(issuer=issuer, client_id=_CLIENT_ID)
    provider._discovery = {"issuer": issuer, "jwks_uri": f"{issuer}/jwks"}
    provider._discovery_fetched_at = time.time()
    return provider


def test_nous_valid_jwt_from_populated_http_jwks_verifies(jwks_server, signing_material):
    portal, _ = jwks_server
    session = _nous(portal).verify_session(access_token=_token(signing_material, issuer=portal))
    assert session is not None
    assert session.user_id == "user-1"


@pytest.mark.parametrize("kid", ["unknown-key", "absent", None])
def test_nous_unknown_or_absent_kid_against_usable_jwks_is_invalid_not_outage(
    jwks_server, signing_material, kid,
):
    portal, _ = jwks_server
    assert _nous(portal).verify_session(access_token=_token(signing_material, kid=kid, issuer=portal)) is None


def test_self_hosted_unknown_kid_against_usable_jwks_is_invalid_not_outage(jwks_server, signing_material):
    issuer, _ = jwks_server
    assert _self_hosted(issuer).verify_session(
        access_token=_token(signing_material, kid="unknown-key", issuer=issuer)
    ) is None


@pytest.mark.parametrize("document", [{}, {"keys": []}, {"keys": "not-a-list"}, "not-json"])
def test_self_hosted_unusable_or_malformed_jwks_is_provider_error(
    jwks_server, signing_material, document,
):
    issuer, handler = jwks_server
    handler.response = document
    with pytest.raises(ProviderError):
        _self_hosted(issuer).verify_session(
            access_token=_token(signing_material, issuer=issuer)
        )


def test_self_hosted_jwks_transport_outage_is_provider_error(signing_material):
    issuer = "http://127.0.0.1:9"
    with pytest.raises(ProviderError):
        _self_hosted(issuer).verify_session(
            access_token=_token(signing_material, issuer=issuer)
        )


@pytest.mark.parametrize("document", [{}, {"keys": []}, {"keys": "not-a-list"}, "not-json"])
def test_nous_unusable_or_malformed_jwks_is_provider_error(jwks_server, signing_material, document):
    portal, handler = jwks_server
    handler.response = document
    with pytest.raises(ProviderError):
        _nous(portal).verify_session(access_token=_token(signing_material, issuer=portal))


def test_nous_jwks_transport_outage_is_provider_error(signing_material):
    with pytest.raises(ProviderError):
        _nous("http://127.0.0.1:9").verify_session(
            access_token=_token(signing_material, issuer="http://127.0.0.1:9")
        )


@pytest.fixture
def gated_nous(jwks_server):
    portal, _ = jwks_server
    clear_providers()
    previous = {name: getattr(web_server.app.state, name, None) for name in ("bound_host", "bound_port", "auth_required")}
    web_server.app.state.bound_host = "agent.example.test"
    web_server.app.state.bound_port = 443
    web_server.app.state.auth_required = True
    register_provider(_nous(portal))
    try:
        yield TestClient(web_server.app, base_url="https://agent.example.test"), portal
    finally:
        clear_providers()
        for name, value in previous.items():
            setattr(web_server.app.state, name, value)


def test_gated_api_rejects_unknown_kid_bearer_with_401(gated_nous, signing_material):
    client, portal = gated_nous
    response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {_token(signing_material, kid='unknown-key', issuer=portal)}"},
    )
    assert response.status_code == 401


def test_gated_api_rejects_absent_kid_cookie_with_401(gated_nous, signing_material):
    client, portal = gated_nous
    client.cookies.set(SESSION_AT_COOKIE, _token(signing_material, kid=None, issuer=portal))
    response = client.get("/api/auth/me")
    assert response.status_code == 401

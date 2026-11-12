"""Nous Portal agent passes on the peer route.

The Portal mints short-lived passes for one agent (``aud`` = the agent's client_id). A pass that
carries ``agent_dashboard:access`` is a dashboard sign-in; a peer pass carries only the peer
permissions (``agents:message`` / ``agents:control``) and must open ``/api/peer`` and nothing else.
"""
from __future__ import annotations

import httpx
import jwt
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

import plugins.dashboard_auth.nous as nous_plugin
from hermes_agent.hermes_cli import web_server
from hermes_agent.hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_agent.hermes_cli.web_routers import peer as peer_routes
from tests.plugins.dashboard_auth.test_nous_provider import _mint_token, _patched_jwks
from tests.plugins.dashboard_auth.test_nous_provider import rsa_keypair as rsa_keypair  # noqa: PLC0414 -- the self-alias
# is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import

PORTAL = "https://portal.example.com"
PEER_SCOPES = "agents:message agents:control"


@pytest.fixture
def provider(rsa_keypair):
    p = nous_plugin.NousDashboardAuthProvider(client_id="agent:inst123", portal_url=PORTAL)
    _patched_jwks(p, rsa_keypair)
    return p


def test_a_peer_pass_is_not_a_dashboard_session(provider, rsa_keypair):
    token = _mint_token(rsa_keypair, scope=PEER_SCOPES)
    assert provider.verify_session(access_token=token) is None

    principal = provider.verify_peer_pass(token=token)
    assert principal is not None
    assert (principal.principal, principal.provider) == ("usr_abc", "nous")
    assert set(principal.scopes) == {"agents:message", "agents:control"}


def test_a_dashboard_sign_in_still_works_and_carries_no_peer_permission(provider, rsa_keypair):
    token = _mint_token(rsa_keypair)  # scope agent_dashboard:access, what every dashboard grant mints
    assert provider.verify_session(access_token=token).user_id == "usr_abc"
    principal = provider.verify_peer_pass(token=token)
    assert principal is not None and not {"agents:message", "agents:control"} & set(principal.scopes)


def test_a_token_without_a_scope_claim_opens_nothing(provider, rsa_keypair):
    """Every dashboard grant and every pass writes ``scope``. A Portal token for this agent without
    one was minted for some other purpose, so it is neither a dashboard session nor a peer pass."""
    claims = jwt.decode(_mint_token(rsa_keypair), options={"verify_signature": False})
    del claims["scope"]
    token = jwt.encode(claims, rsa_keypair["private_pem"], algorithm="RS256", headers={"kid": rsa_keypair["kid"]})
    assert provider.verify_session(access_token=token) is None
    principal = provider.verify_peer_pass(token=token)
    assert principal is not None and principal.scopes == ()


def test_a_pass_for_another_agent_is_not_recognised(provider, rsa_keypair):
    assert provider.verify_peer_pass(token=_mint_token(rsa_keypair, aud="agent:other", scope=PEER_SCOPES)) is None
    assert provider.verify_peer_pass(token="not-a-jwt") is None


def test_on_the_dashboard_a_peer_pass_opens_the_peer_route_and_nothing_else(provider, rsa_keypair, monkeypatch):
    seen: list[str] = []

    async def gateway(request):
        seen.append(request.url.path)
        return JSONResponse({"status": "ok"})
    transport = httpx.ASGITransport(app=Starlette(routes=[Route("/health", gateway)]))
    monkeypatch.setattr(peer_routes, "_gateway_client", lambda _state: httpx.AsyncClient(transport=transport))
    monkeypatch.setattr(peer_routes, "_upstream", lambda profile: ("http://gateway.test", "agent-key-0123456789"))
    clear_providers()
    register_provider(provider)
    state = web_server.app.state
    prev = {k: getattr(state, k, None) for k in ("bound_host", "bound_port", "auth_required")}
    state.bound_host, state.bound_port, state.auth_required = "agent.example.com", 443, True
    peer_pass = {"Authorization": f"Bearer {_mint_token(rsa_keypair, scope=PEER_SCOPES)}"}
    try:
        with TestClient(web_server.app, base_url="https://agent.example.com") as client:
            assert client.get("/api/sessions", headers=peer_pass).status_code == 401
            assert client.get("/api/peer/health", headers=peer_pass).status_code == 200
    finally:
        clear_providers()
        for key, value in prev.items():
            setattr(state, key, value)
    assert seen == ["/health"]

"""The gateway-mounted Webapp receives chat OAuth callbacks without browser authentication."""

import asyncio

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server, web_server_mcp
from hermes_constants import get_hermes_home
from tools.mcp_dashboard_oauth import DashboardOAuthFlow
from tui_gateway import mcp_oauth_sessions as sessions


@pytest.fixture
def client(monkeypatch):
    web_server_mcp._mcp_oauth_flows.clear()
    sessions._sessions.clear()
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    yield TestClient(web_server.app)
    web_server_mcp._mcp_oauth_flows.clear()
    sessions._sessions.clear()


def _flow(flow_id="connector", state="fixture-state"):
    flow = DashboardOAuthFlow(flow_id, "reports", None, str(get_hermes_home()), "")
    asyncio.run(flow.publish_authorization_url(f"https://idp.example/authorize?state={state}"))
    return flow


def test_public_web_callback_delivers_connector_and_rejects_replay(client):
    flow = _flow()
    sessions.register_flow(flow)
    params = {"code": "fixture-code", "state": "fixture-state", "iss": "https://idp.example"}
    response = client.get("/api/mcp/oauth/callback/reports", params=params)
    assert response.status_code == 200
    assert "Authorization received" in response.text
    assert asyncio.run(flow.wait_for_callback(timeout=0)) == ("fixture-code", "fixture-state", "https://idp.example")
    assert client.get("/api/mcp/oauth/callback/reports", params=params).status_code == 409


@pytest.mark.parametrize("params,expected", [
    ({"state": "wrong", "code": "fixture-code"}, 404),
    ({"code": "fixture-code"}, 404),
    ({"state": "fixture-state", "error": "access_denied"}, 400),
])
def test_public_web_callback_rejects_bad_or_denied_callback(client, params, expected):
    sessions.register_flow(_flow())
    assert client.get("/api/mcp/oauth/callback/reports", params=params).status_code == expected


def test_public_web_callback_without_either_registry_is_expired(client):
    assert client.get("/api/mcp/oauth/callback/reports", params={"state": "fixture-state"}).status_code == 404


def test_dashboard_flow_wins_when_both_registries_match(client):
    dashboard = _flow("dashboard")
    connector = _flow()
    web_server_mcp._mcp_oauth_flows[dashboard.flow_id] = dashboard
    sessions.register_flow(connector)
    response = client.get("/api/mcp/oauth/callback/reports", params={"state": "fixture-state", "code": "fixture-code"})
    assert response.status_code == 200
    assert dashboard._callback_ready.is_set() and not connector._callback_ready.is_set()

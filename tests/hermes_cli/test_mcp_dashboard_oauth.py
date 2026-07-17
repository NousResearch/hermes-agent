"""Dashboard HTTP contract for hosted MCP OAuth."""

from unittest.mock import patch

import pytest


def _client():
    from starlette.testclient import TestClient

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


@pytest.fixture(autouse=True)
def _clear_flows():
    from hermes_cli import web_server

    web_server._mcp_oauth_flows.clear()
    yield
    web_server._mcp_oauth_flows.clear()


def test_hosted_auth_start_returns_public_authorization_url(monkeypatch):
    from hermes_cli import web_server

    client = _client()
    client.post(
        "/api/mcp/servers",
        json={"name": "reports", "url": "https://mcp.example/mcp", "auth": "oauth"},
    )

    def fake_worker(flow, cfg):
        import asyncio

        asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=s1"))

    monkeypatch.setattr(web_server, "_run_dashboard_mcp_oauth", fake_worker)
    with patch(
        "hermes_cli.dashboard_auth.prefix.resolve_public_url",
        return_value="https://agent.example",
    ):
        response = client.post("/api/mcp/servers/reports/auth")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "authorization_required"
    assert body["authorization_url"] == "https://idp.example/authorize?state=s1"
    flow = web_server._mcp_oauth_flows[body["flow_id"]]
    assert flow.redirect_uri == f"https://agent.example/api/mcp/oauth/callback/{body['flow_id']}"


def test_hosted_callback_is_public_and_delivers_code():
    import asyncio

    from hermes_cli import web_server
    from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-public",
        server_name="reports",
        profile=None,
        redirect_uri="https://agent.example/api/mcp/oauth/callback/flow-public",
    )
    asyncio.run(
        flow.publish_authorization_url(
            "https://idp.example/authorize?state=expected"
        )
    )
    web_server._mcp_oauth_flows[flow.flow_id] = flow

    assert "/api/mcp/oauth/callback" not in PUBLIC_API_PATHS
    response = _client().get(
        "/api/mcp/oauth/callback/flow-public?code=abc&state=expected"
    )
    assert response.status_code == 200
    assert flow._callback == ("abc", "expected")


def test_hosted_callback_bypasses_gated_cookie_auth(monkeypatch):
    import asyncio

    from starlette.testclient import TestClient

    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-gated",
        server_name="reports",
        profile=None,
        redirect_uri="https://agent.example/api/mcp/oauth/callback/flow-gated",
    )
    asyncio.run(
        flow.publish_authorization_url(
            "https://idp.example/authorize?state=expected"
        )
    )
    web_server._mcp_oauth_flows[flow.flow_id] = flow
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)

    response = TestClient(web_server.app).get(
        "/api/mcp/oauth/callback/flow-gated?code=abc&state=expected"
    )

    assert response.status_code == 200
    assert flow._callback == ("abc", "expected")


def test_hosted_callback_rejects_wrong_state_before_waking_sdk():
    import asyncio

    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-state-route",
        server_name="reports",
        profile=None,
        redirect_uri="https://agent.example/api/mcp/oauth/callback/flow-state-route",
    )
    asyncio.run(
        flow.publish_authorization_url(
            "https://idp.example/authorize?state=expected-state"
        )
    )
    web_server._mcp_oauth_flows[flow.flow_id] = flow

    response = _client().get(
        "/api/mcp/oauth/callback/flow-state-route?code=attacker&state=wrong"
    )
    assert response.status_code == 400
    assert flow._callback is None


def test_hosted_auth_start_bounds_pending_flow_registry():
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    client = _client()
    client.post(
        "/api/mcp/servers",
        json={"name": "reports", "url": "https://mcp.example/mcp", "auth": "oauth"},
    )
    for index in range(web_server._MAX_PENDING_MCP_OAUTH_FLOWS):
        flow = DashboardOAuthFlow(
            flow_id=f"existing-{index}",
            server_name="reports",
            profile=None,
            redirect_uri=f"https://agent.example/callback/{index}",
        )
        web_server._mcp_oauth_flows[flow.flow_id] = flow

    response = client.post("/api/mcp/servers/reports/auth")
    assert response.status_code == 429


def test_hosted_auth_rejects_overlapping_flow_for_same_server():
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    client = _client()
    client.post(
        "/api/mcp/servers",
        json={"name": "reports", "url": "https://mcp.example/mcp", "auth": "oauth"},
    )
    existing = DashboardOAuthFlow(
        flow_id="existing-reports",
        server_name="reports",
        profile="other-profile",
        redirect_uri="https://agent.example/callback/existing",
    )
    web_server._mcp_oauth_flows[existing.flow_id] = existing

    response = client.post("/api/mcp/servers/reports/auth")

    assert response.status_code == 409
    assert "already in progress" in response.text


def test_flow_status_does_not_expose_authorization_code():
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-status",
        server_name="reports",
        profile=None,
        redirect_uri="https://agent.example/api/mcp/oauth/callback/flow-status",
    )
    flow.authorization_url = "https://idp.example/authorize"
    flow.status = "approved"
    flow._callback = ("secret-code", "secret-state")
    web_server._mcp_oauth_flows[flow.flow_id] = flow

    response = _client().get("/api/mcp/oauth/flows/flow-status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert "secret-code" not in response.text
    assert "secret-state" not in response.text

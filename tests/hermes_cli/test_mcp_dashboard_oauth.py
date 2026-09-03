"""Dashboard HTTP contract for hosted MCP OAuth."""

from contextlib import nullcontext
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
    web_server.app.state.auth_required = False
    yield
    web_server._mcp_oauth_flows.clear()
    web_server.app.state.auth_required = False


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
    assert flow.redirect_uri == "https://agent.example/api/mcp/oauth/callback/reports"


def test_hosted_callback_bypasses_gated_cookie_auth(monkeypatch):
    import asyncio

    from starlette.testclient import TestClient

    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-gated",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/api/mcp/oauth/callback/reports",
    )
    asyncio.run(
        flow.publish_authorization_url(
            "https://idp.example/authorize?state=expected"
        )
    )
    web_server._mcp_oauth_flows[flow.flow_id] = flow
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)

    response = TestClient(web_server.app).get(
        "/api/mcp/oauth/callback/reports?code=abc&state=expected"
    )

    assert response.status_code == 200
    assert flow._callback == ("abc", "expected")


def test_hosted_auth_allows_same_server_name_in_different_profiles(tmp_path, monkeypatch):
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(web_server, "_resolve_profile_dir", lambda _name: profile_home)

    existing = DashboardOAuthFlow(
        flow_id="existing-default",
        server_name="reports",
        profile=None,
        hermes_home=str(tmp_path / "default"),
        redirect_uri="https://agent.example/callback/existing",
    )
    web_server._mcp_oauth_flows[existing.flow_id] = existing

    def fake_worker(flow, cfg):
        import asyncio

        asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=work"))

    with patch("hermes_cli.mcp_config._get_mcp_servers", return_value={"reports": {"url": "https://mcp.example"}}), \
         patch.object(web_server, "_run_dashboard_mcp_oauth", fake_worker):
        response = _client().post("/api/mcp/servers/reports/auth?profile=work")

    assert response.status_code != 409




def test_flow_status_does_not_expose_authorization_code():
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-status",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
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


def test_oauth_worker_error_redacts_server_env_file_values(tmp_path):
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    env_file = tmp_path / "server.env"
    env_file.write_text(
        "MCP_PRIVATE_TOKEN=server-secret-value\n", encoding="utf-8"
    )
    flow = DashboardOAuthFlow(
        flow_id="flow-private",
        server_name="private",
        profile=None,
        hermes_home=str(tmp_path),
        redirect_uri="https://agent.example/api/mcp/oauth/callback/private",
    )
    cfg = {
        "url": "https://example.invalid/${MCP_PRIVATE_TOKEN}",
        "env_file": str(env_file),
    }

    with patch(
        "hermes_cli.mcp_config._probe_single_server",
        side_effect=RuntimeError("request failed at /server-secret-value"),
    ), patch(
        "tools.mcp_oauth_manager.get_manager"
    ) as get_manager, patch(
        "tools.mcp_oauth.HermesTokenStorage"
    ) as token_storage, patch(
        "tools.mcp_oauth.force_interactive_oauth",
        return_value=nullcontext(),
    ), patch(
        "tools.mcp_dashboard_oauth.dashboard_oauth_flow",
        return_value=nullcontext(),
    ), patch(
        "tools.mcp_oauth.humanize_oauth_registration_error",
        return_value=None,
    ):
        token_storage.return_value.snapshot.return_value = object()
        get_manager.return_value.remove.return_value = None
        web_server._run_dashboard_mcp_oauth(flow, cfg)

    assert flow.status == "error"
    assert flow.error is not None
    assert "server-secret-value" not in flow.error
    assert "[REDACTED]" in flow.error

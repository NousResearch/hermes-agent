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
        "&iss=https%3A%2F%2Fidp.example"
    )

    assert response.status_code == 200
    assert flow._callback == ("abc", "expected", "https://idp.example")


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


def test_dashboard_worker_initiates_oauth_before_public_probe(tmp_path, monkeypatch):
    """Desktop Authenticate must not rely on an anonymous tools/list probe."""
    import asyncio

    from hermes_cli import mcp_config, web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, get_dashboard_oauth_flow

    events = []

    def fake_initiate(name, cfg, connect_timeout):
        events.append("authorize")
        assert name == "hugging_face"
        assert connect_timeout >= 315
        flow = get_dashboard_oauth_flow()
        assert flow is not None
        asyncio.run(
            flow.publish_authorization_url(
                "https://huggingface.co/oauth/authorize?state=expected"
            )
        )

    def fake_probe(name, cfg, connect_timeout=30):
        assert events == ["authorize"]
        events.append("probe")
        return [("hf_fs", "Browse Hugging Face")]

    monkeypatch.setattr(mcp_config, "_initiate_explicit_oauth", fake_initiate)
    monkeypatch.setattr(mcp_config, "_probe_single_server", fake_probe)
    monkeypatch.setattr(mcp_config, "_oauth_tokens_present", lambda _name: True)
    monkeypatch.setattr(mcp_config, "_save_mcp_server", lambda _name, _cfg: True)

    flow = DashboardOAuthFlow(
        flow_id="flow-explicit-auth",
        server_name="hugging_face",
        profile=None,
        hermes_home=str(tmp_path),
        redirect_uri="http://127.0.0.1:43123/callback",
    )
    web_server._run_dashboard_mcp_oauth(
        flow,
        {
            "url": "https://huggingface.co/mcp",
            "auth": "oauth",
            "oauth": {"scope": "openid profile read-mcp"},
        },
    )

    assert events == ["authorize", "probe"]
    assert flow.status == "approved"
    assert flow.authorization_url.startswith("https://huggingface.co/oauth/authorize")
    assert flow.tools == [{"name": "hf_fs", "description": "Browse Hugging Face"}]
    assert flow.worker_done is True




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

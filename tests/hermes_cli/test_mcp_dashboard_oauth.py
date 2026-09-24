"""Dashboard HTTP contract for hosted MCP OAuth."""

import asyncio
from unittest.mock import patch

import pytest
import hermes_cli.web_server_mcp as _web_server_mcp
import hermes_cli.web_server_profiles as _web_server_profiles


def _client():
    from starlette.testclient import TestClient

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client


@pytest.fixture(autouse=True)
def _clear_flows():
    from hermes_cli import web_server

    _web_server_mcp._mcp_oauth_flows.clear()
    web_server.app.state.auth_required = False
    yield
    _web_server_mcp._mcp_oauth_flows.clear()
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

    monkeypatch.setattr(_web_server_mcp, "_run_dashboard_mcp_oauth", fake_worker)
    with patch(
        "hermes_cli.dashboard_auth.prefix.resolve_public_url",
        return_value="https://agent.example",
    ):
        response = client.post("/api/mcp/servers/reports/auth")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "authorization_required"
    assert body["authorization_url"] == "https://idp.example/authorize?state=s1"
    flow = _web_server_mcp._mcp_oauth_flows[body["flow_id"]]
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
    _web_server_mcp._mcp_oauth_flows[flow.flow_id] = flow
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)

    response = TestClient(web_server.app).get(
        "/api/mcp/oauth/callback/reports?code=abc&state=expected"
    )

    assert response.status_code == 200
    assert flow._callback == ("abc", "expected", None)


def test_hosted_auth_allows_same_server_name_in_different_profiles(tmp_path, monkeypatch):
    from hermes_cli import web_server
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    monkeypatch.setattr(_web_server_profiles, "_resolve_profile_dir", lambda _name: profile_home)

    existing = DashboardOAuthFlow(
        flow_id="existing-default",
        server_name="reports",
        profile=None,
        hermes_home=str(tmp_path / "default"),
        redirect_uri="https://agent.example/callback/existing",
    )
    _web_server_mcp._mcp_oauth_flows[existing.flow_id] = existing

    def fake_worker(flow, cfg):
        import asyncio

        asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=work"))

    with patch("hermes_cli.mcp_config._get_mcp_servers", return_value={"reports": {"url": "https://mcp.example"}}), \
         patch.object(_web_server_mcp, "_run_dashboard_mcp_oauth", fake_worker):
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
    _web_server_mcp._mcp_oauth_flows[flow.flow_id] = flow

    response = _client().get("/api/mcp/oauth/flows/flow-status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert "secret-code" not in response.text
    assert "secret-state" not in response.text


def test_cancel_marks_the_flow_terminal_for_a_retrying_worker():
    """DELETE /flows/{id} is a user cancellation: the worker's next attempt must not re-mint the
    flow behind the dashboard's back (it would reopen a URL the user just dismissed)."""
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    flow = DashboardOAuthFlow(
        flow_id="flow-cancel-retry",
        server_name="reports",
        profile=None,
        hermes_home="/tmp/hermes-test",
        redirect_uri="https://agent.example/api/mcp/oauth/callback/reports",
    )
    asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=first"))
    _web_server_mcp._mcp_oauth_flows[flow.flow_id] = flow

    assert _client().delete("/api/mcp/oauth/flows/flow-cancel-retry").json()["ok"] is True
    with pytest.raises(RuntimeError, match="cancelled"):
        asyncio.run(flow.publish_authorization_url("https://idp.example/authorize?state=second"))
    assert flow.snapshot()["status"] == "error"


def test_dashboard_reauth_marks_server_for_forced_oauth(monkeypatch, tmp_path):
    """``_run_dashboard_mcp_oauth`` sets the one-shot force flag after ``manager.remove`` and
    before the probe — the dashboard's Authenticate button must behave like ``hermes mcp login``
    for servers that answer ``initialize`` with 200 and never challenge (#89412)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.mcp_config as mc
    import tools.mcp_oauth_manager as mgr_mod
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    forced = []
    real_manager = mgr_mod.get_manager()

    class _RecordingManager:
        def remove(self, name, hermes_home=None):
            return real_manager.remove(name, hermes_home=hermes_home)

        def set_force_oauth(self, name):
            forced.append(name)

        def restore_entry(self, *args, **kwargs):
            pass

    monkeypatch.setattr(mgr_mod, "get_manager", lambda: _RecordingManager())
    monkeypatch.setattr(mc, "_probe_single_server", lambda name, cfg, connect_timeout=None: [("t", "d")])
    monkeypatch.setattr(mc, "_oauth_tokens_present", lambda name: True)
    monkeypatch.setattr(mc, "_save_mcp_server", lambda name, cfg: True)

    flow = DashboardOAuthFlow(
        flow_id="flow-force-oauth",
        server_name="gmail",
        profile=None,
        hermes_home=str(tmp_path),
        redirect_uri="https://agent.example/api/mcp/oauth/callback/gmail",
        reconnect_live=False,
    )
    _web_server_mcp._run_dashboard_mcp_oauth(flow, {"url": "https://gmailmcp.example.test/mcp/v1", "auth": "oauth"})

    assert forced == ["gmail"]
    assert flow.snapshot()["status"] == "approved"

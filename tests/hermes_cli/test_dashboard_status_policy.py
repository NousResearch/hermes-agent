"""Status auth policy must reach both HTTP gates through the real config loader."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.config import atomic_config_write
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.cookies import SESSION_AT_COOKIE
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


@pytest.fixture
def status_policy_app(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in (
        "bound_host", "bound_port", "auth_required", "status_auth_required",
        "trusted_public_hosts", "trusted_public_origins", "listener_public_url", "ui_surface",
    ):
        monkeypatch.setattr(web_server.app.state, name, None, raising=False)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "dashboard")
    clear_providers()
    register_provider(StubAuthProvider())
    yield tmp_path
    clear_providers()


@pytest.mark.parametrize("enabled", [None, False, True], ids=["default", "public", "private"])
@pytest.mark.parametrize("gated", [False, True], ids=["loopback-token", "provider"])
def test_config_controls_status_without_gating_health(status_policy_app, enabled, gated):
    dashboard = {} if enabled is None else {"require_auth_for_status": enabled}
    if gated:
        # The real reverse-proxy startup path must keep its external Host/Origin trust.
        dashboard["public_url"] = "https://status.example.test"
    atomic_config_write(status_policy_app / "config.yaml", {"dashboard": dashboard})
    web_server.app.state.bound_host = "127.0.0.1"
    web_server._configure_auth_gate("127.0.0.1", False, None, None)
    assert web_server.app.state.auth_required is gated
    url = "https://status.example.test" if gated else "http://127.0.0.1"
    client = TestClient(web_server.app, base_url=url)

    anonymous = client.get("/api/status")
    assert anonymous.status_code == (401 if enabled else 200), anonymous.text
    if gated and enabled:
        assert anonymous.json()["error"] == "unauthenticated"
        assert anonymous.json()["login_url"] == "/login"
    assert client.get("/api/health").status_code == 200
    # Removing the exact /api/status bypass must not broaden adjacent paths.
    assert client.get("/api/status/extension").status_code == 401
    assert client.get("/api/status", headers={"Authorization": "Bearer invalid"}).status_code == (
        401 if enabled else 200
    )

    if gated and enabled:
        assert client.get("/api/status", headers={
            "X-Hermes-Session-Token": web_server._SESSION_TOKEN,
        }).status_code == 401  # Loopback's token must not replace the active provider.
        assert client.get("/api/status", params={"token": web_server._SESSION_TOKEN}).status_code == 401
    if gated:
        start = client.get("/auth/login?provider=stub", follow_redirects=False)
        state = start.headers["location"].split("state=")[1]
        callback = client.get(f"/auth/callback?code=stub_code&state={state}", follow_redirects=False)
        assert callback.status_code == 302
        bearer = client.cookies[f"__Host-{SESSION_AT_COOKIE}"]
        assert client.get("/api/status").status_code == 200
        client.cookies.clear()
        headers = {"Authorization": f"Bearer {bearer}"}
    else:
        headers = {"X-Hermes-Session-Token": web_server._SESSION_TOKEN}
    response = client.get("/api/status", headers=headers)
    assert response.status_code == 200, response.text
    assert "version" in response.json() and "gateway_state" in response.json()


def test_private_status_uses_provider_refresh(status_policy_app):
    atomic_config_write(status_policy_app / "config.yaml", {
        "dashboard": {"require_auth_for_status": True, "public_url": "https://status.example.test"},
    })
    web_server.app.state.bound_host = "127.0.0.1"
    web_server._configure_auth_gate("127.0.0.1", False, None, None)
    clear_providers()
    register_provider(StubAuthProvider(default_ttl=0))
    client = TestClient(web_server.app, base_url="https://status.example.test")
    start = client.get("/auth/login?provider=stub", follow_redirects=False)
    state = start.headers["location"].split("state=")[1]
    client.get(f"/auth/callback?code=stub_code&state={state}", follow_redirects=False)
    # Browser evicted the expired AT; refresh must run, never treat status as public.
    for cookie in list(client.cookies.jar):
        if cookie.name.endswith(SESSION_AT_COOKIE):
            client.cookies.delete(cookie.name, domain=cookie.domain, path=cookie.path)
    refreshed = client.get("/api/status")
    assert refreshed.status_code == 200, refreshed.text
    assert any(SESSION_AT_COOKIE in value for value in refreshed.headers.get_list("set-cookie"))
    client.cookies.clear()
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/health").status_code == 200

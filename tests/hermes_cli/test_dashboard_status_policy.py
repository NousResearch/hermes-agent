"""Status auth policy must reach both HTTP gates through the real config loader."""
from __future__ import annotations

import asyncio
import types

import httpx
import pytest
from fastapi.testclient import TestClient

import hermes_cli.config as config_module
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


def _configure_dashboard(home, dashboard, *, gated=False) -> TestClient:
    """Write *dashboard* to config.yaml and resolve the gate exactly as ``hermes dashboard`` does."""
    if gated:
        # The real reverse-proxy startup path must keep its external Host/Origin trust.
        dashboard = {**dashboard, "public_url": "https://status.example.test"}
    atomic_config_write(home / "config.yaml", {"dashboard": dashboard})
    web_server.app.state.bound_host = "127.0.0.1"
    web_server._configure_auth_gate("127.0.0.1", False, None, None)
    assert web_server.app.state.auth_required is gated
    return TestClient(web_server.app, base_url="https://status.example.test" if gated else "http://127.0.0.1")


@pytest.mark.parametrize(("raw", "private"), [
    ("false", False), ("False", False), ("0", False), ("off", False),
    ("true", True), ("TRUE", True), ("1", True), ("yes", True),
])
def test_quoted_yaml_flag_uses_the_project_truthy_parser(status_policy_app, raw, private):
    """``require_auth_for_status: "false"`` is a string in YAML; ``bool("false")`` would lock status."""
    client = _configure_dashboard(status_policy_app, {"require_auth_for_status": raw})
    assert web_server.app.state.status_auth_required is private
    assert client.get("/api/status").status_code == (401 if private else 200)


@pytest.mark.parametrize("dashboard", ["oops", ["require_auth_for_status"], 7], ids=["string", "list", "int"])
def test_non_mapping_dashboard_section_keeps_status_public(status_policy_app, dashboard):
    """A malformed ``dashboard:`` block falls back to defaults, as ``dashboard.public_url`` does."""
    client = _configure_dashboard(status_policy_app, dashboard)
    assert web_server.app.state.status_auth_required is False
    assert client.get("/api/status").status_code == 200


def test_unloadable_config_keeps_status_public(status_policy_app, monkeypatch):
    def unloadable():
        raise RuntimeError("config.yaml unreadable")

    web_server.app.state.bound_host = "127.0.0.1"
    with monkeypatch.context() as broken:
        broken.setattr(config_module, "load_config", unloadable)
        broken.setattr(web_server, "load_config", unloadable)
        web_server._configure_auth_gate("127.0.0.1", False, None, None)
    assert web_server.app.state.status_auth_required is False
    assert web_server.app.state.auth_required is False
    client = TestClient(web_server.app, base_url="http://127.0.0.1")
    assert client.get("/api/status").status_code == 200


@pytest.mark.parametrize(("method", "path", "when_public"), [
    ("HEAD", "/api/status", 405),  # public: reaches the GET-only route, never the gate's 401
    ("GET", "/api/status/", 401),  # exact match: the trailing-slash variant is never public
    ("HEAD", "/api/status/", 401),
], ids=["head", "trailing-slash", "head-trailing-slash"])
@pytest.mark.parametrize("enabled", [False, True], ids=["public", "private"])
@pytest.mark.parametrize("gated", [False, True], ids=["loopback-token", "provider"])
def test_status_method_and_slash_variants_follow_the_exact_path_policy(
    status_policy_app, gated, enabled, method, path, when_public,
):
    client = _configure_dashboard(status_policy_app, {"require_auth_for_status": enabled}, gated=gated)
    assert client.request(method, path).status_code == (401 if enabled else when_public)


@pytest.fixture
def webapp_policy_home(tmp_path, monkeypatch):
    """Stub sign-in for the gateway mount. The mount itself sets and removes ``listener_public_url``,
    ``ui_surface``, ``web_dist`` and ``webapp_home``; only what it leaves behind is restored here."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in ("bound_host", "bound_port", "auth_required", "status_auth_required", "trusted_public_hosts",
                "trusted_public_origins", "loopback_companion", "withhold_session_token", "gateway_runner",
                "session_authority"):
        monkeypatch.setattr(web_server.app.state, key, getattr(web_server.app.state, key, None), raising=False)
    clear_providers()
    register_provider(StubAuthProvider())
    yield tmp_path
    clear_providers()


@pytest.mark.parametrize("enabled", [None, False, True], ids=["default", "public", "private"])
def test_gateway_webapp_listener_applies_the_status_policy(webapp_policy_home, enabled):
    """``gateway/run_api.py`` mounts the same app and resolves its gate through ``_configure_auth_gate``
    with the Webapp's own ``listener_public_url``; the opt-in must reach that listener too."""
    from gateway.run_api import GatewayDashboardMount
    from gateway.webapp_listener import WebappMount

    atomic_config_write(webapp_policy_home / "config.yaml", {
        "dashboard": {} if enabled is None else {"require_auth_for_status": enabled}})
    dist = webapp_policy_home / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><html><body>WEBAPP</body></html>", encoding="utf-8")
    state = types.SimpleNamespace(bound_host="0.0.0.0", bound_port=9120, loopback_companion=None,
                                  webapp=WebappMount(dist, webapp_policy_home, ""))
    mount = GatewayDashboardMount(types.SimpleNamespace(session_authority=None), state)

    async def scenario() -> dict:
        try:
            web = await mount.ensure()
            assert (web.app.state.ui_surface, web.app.state.auth_required) == ("webapp", True)
            transport = httpx.ASGITransport(app=mount, client=("192.168.1.50", 50000))
            async with httpx.AsyncClient(transport=transport, base_url="http://192.168.1.20:9120") as browser:
                answers = {
                    "status": (await browser.get("/api/status")).status_code,
                    "head": (await browser.head("/api/status")).status_code,
                    "slash": (await browser.get("/api/status/")).status_code,
                    "health": (await browser.get("/api/health")).status_code,
                }
                await browser.get("/auth/login?provider=stub", follow_redirects=True)
                answers["signed_in"] = (await browser.get("/api/status")).status_code
            return answers
        finally:
            await mount.close()

    private = bool(enabled)
    assert asyncio.run(scenario()) == {
        "status": 401 if private else 200, "head": 401 if private else 405, "slash": 401,
        "health": 200, "signed_in": 200,
    }


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

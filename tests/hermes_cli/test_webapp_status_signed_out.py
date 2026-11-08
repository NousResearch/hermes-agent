"""The gateway's Webapp answers a signed-out ``/api/status`` with liveness and the sign-in shape only.

``/api/status`` is public (``PUBLIC_API_PATHS``): the native-app login probe and uptime checks read
it without a session. A Webapp on a wildcard bind is reachable by everyone on the network, so its
profile names, install id and host figures wait for sign-in. The standalone dashboard keeps its
public body (Hermes Cloud lists profiles from it without a session, #60585).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import hermes_cli.web_server_gateway as _web_server_gateway
import plugins.dashboard_auth.basic as basic_plugin
from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_cli.dashboard_auth.routes import _reset_password_rate_limit

LAN = "http://192.168.50.220:9130"
IDENTITY = ("profiles", "parked_profiles", "gateway_shared_with", "install_id", "memory", "disk",
            "nous_session_valid", "active_sessions", "gateway_mode")


@pytest.fixture(scope="module")
def password_hash():
    return basic_plugin.hash_password("hunter2")


@pytest.fixture(params=["dashboard", "webapp"])
def wildcard_gate(request, password_hash, monkeypatch):
    """An all-interfaces listener behind the bundled username/password sign-in: the standalone
    dashboard or the gateway's Webapp, with two profiles installed."""
    monkeypatch.delenv("HERMES_DASHBOARD_PUBLIC_URL", raising=False)
    for key, value in (("bound_host", "0.0.0.0"), ("bound_port", 9130), ("auth_required", True),
                       ("trusted_public_hosts", frozenset())):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    for key in ("ui_surface", "listener_public_url", "webapp_home"):
        if hasattr(web_server.app.state, key):
            monkeypatch.delattr(web_server.app.state, key)
    if request.param == "webapp":
        monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
        monkeypatch.setattr(web_server.app.state, "listener_public_url", "", raising=False)
    monkeypatch.setattr(_web_server_gateway, "_collect_profile_gateway_topology_cached", lambda: {
        "profiles": ["default", "coder"], "parked_profiles": [], "gateway_mode": "multiplex",
        "gateways": []})
    clear_providers()
    register_provider(basic_plugin.BasicAuthProvider(
        username="admin", password_hash=password_hash, secret=b"s" * 32))
    _reset_password_rate_limit()
    yield request.param
    clear_providers()


def test_signed_out_webapp_status_carries_no_profile_or_install_identity(wildcard_gate):
    client = TestClient(web_server.app, base_url=LAN)
    signed_out = client.get("/api/status")
    assert signed_out.status_code == 200
    body = signed_out.json()
    # What the native-app login probe and uptime checks read survives on every surface.
    assert body["auth_required"] is True and body["auth_providers"] == ["basic"]
    assert body["gateway_running"] in (True, False) and body["overall"] in ("ok", "degraded")
    assert body["version"]
    if wildcard_gate == "dashboard":
        assert body["profiles"] == ["default", "coder"]  # Hermes Cloud's contract (#60585)
        return
    assert [key for key in IDENTITY if key in body] == []
    # ``?profile=`` is no profile-name oracle either: an unknown name answers like a real one.
    assert client.get("/api/status?profile=nope").status_code == 200
    assert "profiles" not in client.get("/api/status?profile=coder").json()

    login = client.post("/auth/password-login",
                        json={"provider": "basic", "username": "admin", "password": "hunter2"})
    assert login.status_code == 200, login.text
    signed_in = client.get("/api/status").json()
    assert signed_in["profiles"] == ["default", "coder"]
    assert signed_in["install_id"]
    assert client.get("/api/status?profile=nope").status_code == 404


def test_a_cookie_the_gate_would_refuse_reads_as_signed_out_on_the_webapp(wildcard_gate):
    if wildcard_gate != "webapp":
        pytest.skip("the dashboard's public body does not depend on the caller")
    client = TestClient(web_server.app, base_url=LAN)
    login = client.post("/auth/password-login",
                        json={"provider": "basic", "username": "admin", "password": "hunter2"})
    assert login.status_code == 200, login.text
    forged = TestClient(web_server.app, base_url=LAN)
    for cookie in client.cookies.jar:
        value = (cookie.value or "") if "signin" in cookie.name else "forged"
        forged.cookies.set(cookie.name, value, domain=cookie.domain, path=cookie.path)
    assert "profiles" not in forged.get("/api/status").json()
    assert "profiles" not in client.get("/api/status", headers={"Authorization": "Bearer forged"}).json()

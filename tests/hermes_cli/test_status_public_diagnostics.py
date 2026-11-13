"""``/api/status`` is public (``PUBLIC_API_PATHS``), so an unverified caller gets the gateway's
diagnostics only as bounded codes: free-text adapter ``error_message``, any adapter-specific
platform field, the raw ``gateway_exit_reason`` and the multiplex remediation prose stay for a
caller the auth gate would accept (#90700).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import hermes_agent.gateway.status as _gw_status
import hermes_agent.hermes_cli.web_server_gateway as _web_server_gateway
import hermes_agent.plugins.dashboard_auth.basic as basic_plugin
from hermes_agent.hermes_cli import web_server
from hermes_agent.hermes_cli.dashboard_auth import clear_providers, register_provider
from hermes_agent.hermes_cli.dashboard_auth.routes import _reset_password_rate_limit

LAN = "http://192.168.50.220:9130"
SECRET = "hunter2-bot-token"
EXIT_TEXT = f"TRACEBACK password={SECRET} endpoint=https://internal.example"
STANDALONE = f"Profiles 'default' and 'coder' both hold WEIXIN_TOKEN={SECRET}"
RUNTIME = {
    "gateway_state": "startup_failed", "exit_reason": EXIT_TEXT,
    "updated_at": "2026-01-01T00:00:00+00:00",
    "platforms": {"telegram": {
        "state": "fatal", "error_code": "telegram_polling_conflict",
        "error_message": f"POST https://api.telegram.org/bot{SECRET}/getMe failed: 409",
        "needs_attention": True, "updated_at": "2026-01-01T00:00:00+00:00",
        "ingress_url": f"https://hooks.internal.example/{SECRET}",
        "writer_pid": 4242, "writer_start_time": 1.0}}}


@pytest.fixture(scope="module")
def password_hash():
    return basic_plugin.hash_password("pw")


@pytest.fixture
def gated(password_hash, monkeypatch):
    monkeypatch.delenv("HERMES_DASHBOARD_PUBLIC_URL", raising=False)
    for key, value in (("bound_host", "0.0.0.0"), ("bound_port", 9130), ("auth_required", True),
                       ("trusted_public_hosts", frozenset())):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    for key in ("ui_surface", "listener_public_url", "webapp_home"):
        if hasattr(web_server.app.state, key):
            monkeypatch.delattr(web_server.app.state, key)
    monkeypatch.setattr(_gw_status, "get_running_pid_cached", lambda *a, **k: None)
    monkeypatch.setattr(_gw_status, "read_runtime_status", lambda *a, **k: dict(RUNTIME))
    monkeypatch.setattr(_web_server_gateway, "_load_configured_gateway_platforms", lambda: {"telegram"})
    monkeypatch.setattr(_web_server_gateway, "_collect_profile_gateway_topology_cached", lambda: {
        "profiles": ["default", "coder"], "parked_profiles": [], "gateway_mode": "multiplex",
        "gateways": [], "multiplex_standalone_reason": STANDALONE})
    clear_providers()
    register_provider(basic_plugin.BasicAuthProvider(
        username="admin", password_hash=password_hash, secret=b"s" * 32))
    _reset_password_rate_limit()
    yield TestClient(web_server.app, base_url=LAN)
    clear_providers()


def test_anonymous_status_carries_codes_not_diagnostic_text(gated):
    resp = gated.get("/api/status")
    assert resp.status_code == 200
    assert SECRET not in resp.text and "internal.example" not in resp.text
    body = resp.json()
    assert body["gateway_state"] == "startup_failed"
    # Still enough for a probe or the pre-login SPA to say what is wrong, as codes.
    assert body["gateway_platforms"] == {"telegram": {
        "state": "fatal", "error_code": "telegram_polling_conflict", "needs_attention": True,
        "updated_at": "2026-01-01T00:00:00+00:00"}}
    assert body["gateway_exit_reason"] == "auth_failed"
    assert body["multiplex_standalone_reason"] is None


def test_signed_in_operator_keeps_full_diagnostics(gated):
    login = gated.post("/auth/password-login",
                       json={"provider": "basic", "username": "admin", "password": "pw"})
    assert login.status_code == 200, login.text
    body = gated.get("/api/status").json()
    entry = body["gateway_platforms"]["telegram"]
    assert SECRET in entry["error_message"]
    assert body["gateway_exit_reason"] == EXIT_TEXT
    assert body["multiplex_standalone_reason"] == STANDALONE


def test_forged_credentials_read_as_anonymous(gated):
    resp = gated.get("/api/status", headers={"Authorization": "Bearer forged"})
    assert resp.status_code == 200
    assert SECRET not in resp.text


def test_hostile_platform_values_are_bounded(gated, monkeypatch):
    monkeypatch.setattr(_gw_status, "read_runtime_status", lambda *a, **k: {
        **RUNTIME, "exit_reason": "loop_liveness_watchdog",
        "platforms": {"telegram": {"state": "fatal", "error_code": f"x {SECRET}",
                                   "needs_attention": SECRET, "updated_at": SECRET,
                                   "retrying_since": {"nested": SECRET}}}})
    resp = gated.get("/api/status")
    assert resp.status_code == 200
    assert SECRET not in resp.text
    assert resp.json()["gateway_platforms"] == {"telegram": {"state": "fatal"}}
    # A source-controlled code already is the bounded form and passes through unchanged.
    assert resp.json()["gateway_exit_reason"] == "loop_liveness_watchdog"


def test_loopback_dashboard_gates_detail_on_the_session_token(gated, monkeypatch):
    # No sign-in gate on a loopback bind: the SPA's injected session token is the credential.
    for key, value in (("bound_host", "127.0.0.1"), ("auth_required", False)):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    client = TestClient(web_server.app, base_url="http://127.0.0.1:9130")
    anonymous = client.get("/api/status")
    assert anonymous.status_code == 200
    assert SECRET not in anonymous.text
    body = client.get("/api/status", headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}).json()
    assert SECRET in body["gateway_platforms"]["telegram"]["error_message"]


def test_unlisted_platform_state_reads_as_unknown():
    # A live gateway's entries reach the projection whatever their state; only the enum passes.
    from hermes_agent.hermes_cli.web_routers.status_public import redact_status_for_unverified
    status = {"gateway_platforms": {"telegram": {"state": f"fatal: {SECRET}"}}}
    redact_status_for_unverified(status)
    assert status["gateway_platforms"] == {"telegram": {"state": "unknown"}}

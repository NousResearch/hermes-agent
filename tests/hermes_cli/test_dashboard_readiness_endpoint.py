"""Public dashboard health/readiness probes.

These routes are intentionally unauthenticated so uptime/smoke checks can verify
that a remote dashboard/QG is alive without a dashboard session, token, or any
secret-bearing header.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth import clear_providers, register_provider
from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider


_HOST_DETAIL_FIELDS = frozenset({
    "hermes_home",
    "config_path",
    "env_path",
    "gateway_pid",
    "gateway_health_url",
})


def test_healthz_is_public_under_gated_dashboard():
    clear_providers()
    register_provider(StubAuthProvider())
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = True
    try:
        client = TestClient(web_server.app, base_url="https://qg.example.test")
        response = client.get("/healthz")
    finally:
        clear_providers()
        web_server.app.state.auth_required = prev_required

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "hermes-dashboard"
    assert "version" in body
    assert not (_HOST_DETAIL_FIELDS & set(body.keys()))


def test_readyz_is_public_and_minimal_under_gated_dashboard(monkeypatch):
    clear_providers()
    register_provider(StubAuthProvider())
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = True
    monkeypatch.setattr(
        web_server,
        "_kanban_readiness_check",
        lambda: {"status": "ok"},
    )
    try:
        client = TestClient(web_server.app, base_url="https://qg.example.test")
        response = client.get("/readyz")
    finally:
        clear_providers()
        web_server.app.state.auth_required = prev_required

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "ok",
        "service": "hermes-dashboard",
        "version": web_server.__version__,
        "checks": {
            "app": {"status": "ok"},
            "kanban_db": {"status": "ok"},
        },
    }
    assert not (_HOST_DETAIL_FIELDS & set(body.keys()))


def test_readyz_degrades_when_kanban_db_unavailable(monkeypatch):
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False
    monkeypatch.setattr(
        web_server,
        "_kanban_readiness_check",
        lambda: {"status": "unavailable"},
    )
    try:
        client = TestClient(web_server.app, base_url="http://127.0.0.1:9119")
        response = client.get("/readyz")
    finally:
        web_server.app.state.auth_required = prev_required

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["kanban_db"] == {"status": "unavailable"}

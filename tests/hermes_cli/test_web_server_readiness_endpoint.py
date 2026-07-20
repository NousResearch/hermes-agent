"""Regression coverage for the lightweight Desktop readiness handshake."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS


@pytest.fixture
def loopback_client():
    previous_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.auth_required = False

    with TestClient(web_server.app, base_url="http://127.0.0.1:9119") as client:
        yield client

    web_server.app.state.auth_required = previous_required


def test_ready_endpoint_is_authenticated_and_minimal(loopback_client):
    assert "/api/ready" not in PUBLIC_API_PATHS
    assert loopback_client.get("/api/ready").status_code == 401

    response = loopback_client.get(
        "/api/ready",
        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
    )

    assert response.status_code == 200
    assert response.json() == {
        "ready": True,
        "version": web_server.__version__,
    }


def test_ready_endpoint_skips_expensive_status_work(loopback_client, monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("readiness must not execute rich /api/status work")

    monkeypatch.setattr(web_server, "check_config_version", unexpected)
    monkeypatch.setattr(web_server, "get_running_pid_cached", unexpected)
    monkeypatch.setattr(web_server, "_status_active_sessions", unexpected)
    monkeypatch.setattr(web_server, "_resolve_restart_drain_timeout", unexpected)
    monkeypatch.setattr(web_server, "_collect_profile_gateway_topology", unexpected)

    response = loopback_client.get(
        "/api/ready",
        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
    )

    assert response.status_code == 200
    assert response.json()["ready"] is True

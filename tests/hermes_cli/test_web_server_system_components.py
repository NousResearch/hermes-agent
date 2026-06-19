"""Tests for the read-only system-components analytics endpoint.

The endpoint serves ``~/.hermes/system_components_status.json`` (produced
out-of-band by ``scripts/publish_system_components_status.py``) annotated with
staleness. It must never 500 — a missing or corrupt snapshot returns a safe
default so the dashboard can render an actionable message.
"""

import json
import time

import pytest
from starlette.testclient import TestClient

from hermes_cli import web_server


def _client_with_app_state():
    prev_auth_required = getattr(web_server.app.state, "auth_required", None)
    prev_bound_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None

    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client, prev_auth_required, prev_bound_host


def _restore_app_state(prev_auth_required, prev_bound_host):
    if prev_auth_required is None:
        if hasattr(web_server.app.state, "auth_required"):
            delattr(web_server.app.state, "auth_required")
    else:
        web_server.app.state.auth_required = prev_auth_required
    if prev_bound_host is None:
        if hasattr(web_server.app.state, "bound_host"):
            delattr(web_server.app.state, "bound_host")
    else:
        web_server.app.state.bound_host = prev_bound_host


@pytest.fixture
def syscomp_client(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(web_server, "get_hermes_home", lambda: home)

    client, prev_auth_required, prev_bound_host = _client_with_app_state()
    try:
        yield client, home
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            close()
        _restore_app_state(prev_auth_required, prev_bound_host)


def _get(client):
    resp = client.get("/api/analytics/system-components")
    assert resp.status_code == 200
    return resp.json()


def test_missing_snapshot_returns_safe_default(syscomp_client):
    client, _home = syscomp_client
    body = _get(client)
    assert body["missing"] is True
    assert body["stale"] is True
    assert body["age_seconds"] is None
    assert body["components"] == []
    assert "publish_system_components_status" in body["message"]


def test_fresh_snapshot_is_not_stale(syscomp_client):
    client, home = syscomp_client
    snapshot = {
        "timestamp": time.time(),
        "source": "publish_system_components_status.py",
        "summary": {"total": 1, "up": 1, "degraded": 0, "down": 0, "absent": 0},
        "components": [
            {
                "name": "Hermes Gateway",
                "endpoint": "http://127.0.0.1:8765/api/status",
                "status": "up",
                "http_status": 200,
                "latency_ms": 4.2,
                "error": None,
            }
        ],
        "auxiliary_routes": {
            "compression_provider": "openrouter",
            "compression_model": "openrouter/auto:fusion",
        },
    }
    (home / "system_components_status.json").write_text(json.dumps(snapshot))

    body = _get(client)
    assert body["missing"] is False
    assert body["stale"] is False
    assert body["age_seconds"] is not None and body["age_seconds"] >= 0
    assert len(body["components"]) == 1
    assert body["components"][0]["name"] == "Hermes Gateway"
    assert body["auxiliary_routes"]["compression_model"] == "openrouter/auto:fusion"


def test_old_snapshot_is_stale(syscomp_client):
    client, home = syscomp_client
    old_ts = time.time() - (3 * 60 * 60)  # 3h old, past the 2h threshold
    (home / "system_components_status.json").write_text(
        json.dumps({"timestamp": old_ts, "components": [], "auxiliary_routes": None})
    )

    body = _get(client)
    assert body["stale"] is True
    assert body["missing"] is False
    assert body["age_seconds"] >= 2 * 60 * 60


def test_corrupt_snapshot_does_not_500(syscomp_client):
    client, home = syscomp_client
    (home / "system_components_status.json").write_text("{ not valid json ")

    body = _get(client)
    assert body["source"] == "error"
    assert body["stale"] is True
    assert body["components"] == []


def test_secret_like_keys_are_redacted(syscomp_client):
    client, home = syscomp_client
    (home / "system_components_status.json").write_text(
        json.dumps(
            {
                "timestamp": time.time(),
                "components": [
                    {
                        "name": "LiteLLM",
                        "endpoint": "http://127.0.0.1:4000/health",
                        "status": "up",
                        "api_key": "sk-should-never-surface",
                    }
                ],
                "auxiliary_routes": None,
            }
        )
    )

    body = _get(client)
    assert body["components"][0]["api_key"] == "***redacted***"

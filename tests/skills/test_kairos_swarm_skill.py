"""Tests for Kairos Swarm skill and dashboard API.

Verifies:
1. Public endpoints (/api/status, /health, /).
2. Authorization enforcement on /api/trigger_goal and /api/update_agent (401 without auth, 200 with auth).
3. Import safety and error logging without UnboundLocalError.
4. Behavioral configuration from config.yaml / env.
5. SKILL.md documentation and port consistency.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Load dashboard_api dynamically from optional-skills path
_API_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "optional-skills"
    / "autonomous-ai-agents"
    / "kairos-swarm"
    / "backend"
    / "dashboard_api.py"
)
_spec = importlib.util.spec_from_file_location("dashboard_api", _API_PATH)
dashboard_api = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dashboard_api)


@pytest.fixture
def client():
    return TestClient(dashboard_api.app)


def test_public_status_and_health_endpoints(client):
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json().get("status") == "ok"

    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    data = res_status.json()
    assert "agents" in data
    assert "active_task" in data


def test_embedded_dashboard_html_fallback(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "KAIROS SWARM DASHBOARD" in res.text


def test_trigger_goal_requires_authorization(client, monkeypatch):
    monkeypatch.setenv("KAIROS_API_KEY", "secret-test-key")

    # Unauthorized request (no headers)
    res_unauth = client.post("/api/trigger_goal", json={"goal": "Test goal"})
    assert res_unauth.status_code == 401
    assert "Unauthorized" in res_unauth.json().get("detail", "")

    # Authorized request with Bearer header
    res_auth = client.post(
        "/api/trigger_goal",
        json={"goal": "Test goal"},
        headers={"Authorization": "Bearer secret-test-key"},
    )
    assert res_auth.status_code == 200
    assert res_auth.json().get("goal") == "Test goal"


def test_update_agent_requires_authorization(client, monkeypatch):
    monkeypatch.setenv("KAIROS_API_KEY", "secret-test-key")

    agent_payload = {
        "name": "Coder",
        "status": "working",
        "current_task": "Writing tests",
        "progress": 50.0,
        "last_update": "2026-07-29T00:00:00",
        "color": "#22c55e",
    }

    # Unauthorized
    res_unauth = client.post("/api/update_agent", json=agent_payload)
    assert res_unauth.status_code == 401

    # Authorized with X-API-Key
    res_auth = client.post(
        "/api/update_agent",
        json=agent_payload,
        headers={"X-API-Key": "secret-test-key"},
    )
    assert res_auth.status_code == 200
    assert res_auth.json().get("success") is True


def test_emit_log_and_agent_update_safety():
    dashboard_api.emit_log("Safety check log message", level="info", sender="TestRunner")
    assert any("Safety check log message" in log for log in dashboard_api.current_state.logs)

    dashboard_api.emit_log("Safety error log message", level="error", sender="TestRunner")
    assert any("Safety error log message" in log for log in dashboard_api.current_state.logs)

    dashboard_api.emit_agent_update("Coder", "working", "Refactoring", 75.0)
    coder = next(a for a in dashboard_api.current_state.agents if a.name == "Coder")
    assert coder.status == "working"
    assert coder.current_task == "Refactoring"
    assert coder.progress == 75.0


def test_skill_md_port_and_content_alignment():
    skill_path = (
        Path(dashboard_api.__file__).resolve().parent.parent / "SKILL.md"
    )
    assert skill_path.exists()
    content = skill_path.read_text(encoding="utf-8")

    assert "--port 8001" in content
    assert "http://localhost:8001" in content
    assert "http://localhost:3000" not in content
    assert "config.yaml" in content

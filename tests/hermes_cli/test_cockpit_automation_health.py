"""Tests for BIF-522 read-only Automation Health Cockpit summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


class ForbiddenMutation:
    def __getattr__(self, name: str):  # pragma: no cover - failure path only
        if any(word in name for word in ("add", "append", "create", "delete", "enqueue", "insert", "repair", "restart", "retry", "run", "save", "send", "set", "trigger", "update", "write")):
            raise AssertionError(f"automation health attempted mutation method {name}")
        raise AttributeError(name)


@pytest.fixture()
def client(monkeypatch, _isolate_hermes_home, tmp_path):
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover
        pytest.skip("fastapi/starlette not installed")

    from hermes_cli import web_server
    import hermes_cli.cockpit_automation_health as health

    from hermes_cli.config import get_hermes_home

    home = Path(get_hermes_home())
    (home / "cron").mkdir(parents=True, exist_ok=True)
    (home / "cron" / "jobs.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "a82830911bcd",
                        "name": "Hermes Daily Ops Radar",
                        "enabled": True,
                        "last_status": "success",
                        "last_run_at": "2026-05-14T08:30:00Z",
                        "next_run_at": "2026-05-15T08:30:00Z",
                        "prompt": "secret prompt should not leak",
                    },
                    {
                        "id": "raw-job-id-should-not-leak",
                        "name": "Nightly check",
                        "enabled": False,
                        "last_status": "failed token=secret123",
                        "last_run_at": "2026-05-12T08:30:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        health,
        "get_n8n_daily_checks_local_payload",
        lambda: {
            "schema_version": 1,
            "read_only": True,
            "source": "n8n_live_latest_execution",
            "live": True,
            "stale": False,
            "actions_enabled": False,
            "external_delivery_enabled": False,
            "generated_at": 1778750000.0,
            "checks": [
                {"name": "Morning Briefing", "status": "observed", "execution_status": "success", "last_completed": "2026-05-14T08:00:00Z", "output_summary": "ok"},
                {"name": "Workflow Health Daily Report", "status": "observed", "execution_status": "error", "last_completed": "2026-05-14T08:05:00Z", "output_summary": "API_KEY=secret should redact"},
            ],
        },
    )
    monkeypatch.setattr(
        health,
        "get_daily_ops_radar_payload",
        lambda: {
            "schema_version": 1,
            "read_only": True,
            "actions_enabled": False,
            "external_delivery_enabled": False,
            "source": "cron_output_latest_markdown",
            "generated_at": 1778750001.0,
            "job": {"enabled": True, "status": "success", "last_run_at": "2026-05-14T08:30:00Z", "next_run_at": "2026-05-15T08:30:00Z", "id": "a82830911bcd"},
            "summary": {"behind_count": 2, "relevant_change_count": 1, "last_run": "2026-05-14T08:30:00Z"},
            "raw_excerpt": "raw logs should not leak",
        },
    )
    monkeypatch.setattr(health, "read_runtime_status", lambda: {"updated_at": "2026-05-14T08:31:00Z", "platforms": {"discord": {"state": "connected", "chat_id": "C1234567890"}}})
    monkeypatch.setattr(health, "get_running_pid", lambda cleanup_stale=False: 4242)
    monkeypatch.setattr(health, "_MUTATION_SENTINEL", ForbiddenMutation())

    return TestClient(web_server.app)


@pytest.fixture()
def auth_headers():
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN

    return {_SESSION_HEADER_NAME: _SESSION_TOKEN}


def _assert_no_sensitive_content(value):
    if isinstance(value, dict):
        forbidden_keys = {"id", "job_id", "prompt", "raw_excerpt", "chat_id", "user_id", "thread_id", "execution_id", "workflow_id", "pid"}
        assert not (set(value) & forbidden_keys)
        for nested in value.values():
            _assert_no_sensitive_content(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_sensitive_content(nested)
    elif isinstance(value, str):
        for forbidden in ["secret", "C1234567890", "raw-job-id", "a82830911bcd", "raw logs", "API_KEY", "token="]:
            assert forbidden not in value


def test_automation_health_requires_dashboard_session_token(client):
    response = client.get("/api/cockpit/automation-health")

    assert response.status_code == 401


def test_automation_health_returns_simplified_read_only_sanitized_summary(client, auth_headers):
    response = client.get("/api/cockpit/automation-health", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 1
    assert body["read_only"] is True
    assert body["actions_enabled"] is False
    assert body["mutation_enabled"] is False
    assert body["external_delivery_enabled"] is False
    assert set(body["reading_model"]) == {"attention", "healthy", "stale_or_failing", "last_checked_source"}
    assert set(body["summary"]) >= {"attention", "healthy", "stale_or_failing", "last_checked", "source"}
    assert 3 <= len(body["cards"]) <= 6
    assert {card["bucket"] for card in body["cards"]} & {"attention", "healthy", "stale_or_failing"}
    assert all(card["last_checked"] and card["source"] for card in body["cards"])
    assert not any(card["last_checked"] == "1778750000.0" for card in body["cards"])
    assert all(card["title"] and card["summary"] for card in body["cards"])
    daily_ops_cards = [card for card in body["cards"] if card["title"] == "Daily Ops Radar"]
    assert daily_ops_cards
    assert daily_ops_cards[0]["bucket"] == "attention"
    assert "upgrade-review signal" in daily_ops_cards[0]["summary"]
    assert "not an automation failure" in daily_ops_cards[0]["summary"]
    assert any("external-send" in detail for detail in daily_ops_cards[0].get("details", []))
    assert all("details" not in card or len(card["details"]) <= 3 for card in body["cards"])
    _assert_no_sensitive_content(body)


def test_automation_health_capability_is_authenticated_read_only(client, auth_headers):
    response = client.get("/api/cockpit/capabilities", headers=auth_headers)

    assert response.status_code == 200
    endpoint = response.json()["endpoints"]["/api/cockpit/automation-health"]
    assert endpoint["methods"] == ["GET"]
    assert endpoint["actions_enabled"] is False
    assert endpoint["external_delivery_enabled"] is False
    assert endpoint["mutation_enabled"] is False


def test_automation_health_endpoint_is_get_only(client, auth_headers):
    for method in ("post", "put", "patch", "delete"):
        response = getattr(client, method)("/api/cockpit/automation-health", headers=auth_headers)
        assert response.status_code == 405


def test_automation_health_n8n_fixture_fallback_is_stale_not_attention(monkeypatch):
    import hermes_cli.cockpit_automation_health as health

    monkeypatch.setattr(
        health,
        "get_n8n_daily_checks_local_payload",
        lambda: {
            "schema_version": 1,
            "read_only": True,
            "source": "fixture_bif_525_inventory",
            "fallback": True,
            "stale": True,
            "generated_at": 1778750000.0,
            "checks": [
                {"name": "Morning Briefing", "execution_status": "unknown", "status": "observed"},
                {"name": "Workflow Health Daily Report", "execution_status": "unknown", "status": "observed"},
            ],
        },
    )

    card = health._n8n_card(1778750001.0)

    assert card["bucket"] == "stale_or_failing"
    assert "fallback-only" in card["summary"]
    assert "needs review" not in card["summary"]
    assert "No repair, retry, or workflow trigger controls" in card["details"][0]


def test_automation_health_stale_local_n8n_errors_are_not_current_attention(monkeypatch):
    import hermes_cli.cockpit_automation_health as health

    monkeypatch.setattr(
        health,
        "get_n8n_daily_checks_local_payload",
        lambda: {
            "schema_version": 1,
            "read_only": True,
            "source": "local_n8n_inventory_summary",
            "fallback": False,
            "stale": True,
            "inventory_checked_at": "2026-05-08T14:31:20Z",
            "generated_at": "2026-05-14T16:30:00Z",
            "checks": [
                {
                    "name": "Alexa Bring Sync",
                    "status": "error",
                    "execution_status": "error",
                    "last_completed": "2026-05-08T14:31:20Z",
                },
                {
                    "name": "Morning Briefing",
                    "status": "success",
                    "execution_status": "success",
                    "last_completed": "2026-05-08T14:31:20Z",
                },
            ],
        },
    )

    card = health._n8n_card(1778758200.0)

    assert card["bucket"] == "stale_or_failing"
    assert "local snapshot still contains" in card["summary"]
    assert "current live failure state" in card["summary"]
    assert "needs review" not in card["summary"]
    assert "Alexa Bring Sync" not in card.get("details", [])
    assert "No repair, retry, or workflow trigger controls" in card["details"][0]


def test_automation_health_uses_only_local_sources_and_never_live_n8n(monkeypatch, _isolate_hermes_home, auth_headers):
    try:
        from starlette.testclient import TestClient
    except ImportError:  # pragma: no cover
        pytest.skip("fastapi/starlette not installed")

    import socket
    import subprocess
    import urllib.request

    from hermes_cli import web_server
    import hermes_cli.cockpit_automation_health as health
    import hermes_cli.cockpit_n8n as n8n
    from hermes_cli.config import get_hermes_home

    home = Path(get_hermes_home())
    (home / "cron").mkdir(parents=True, exist_ok=True)
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": []}), encoding="utf-8")

    class ForbiddenLiveRead(BaseException):
        pass

    def forbidden(*args, **kwargs):  # pragma: no cover - failure path only
        raise ForbiddenLiveRead("Automation Health attempted a live read or subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(n8n, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(n8n.N8nApiClient, "__init__", forbidden)
    monkeypatch.setattr(n8n.N8nApiClient, "_get_json", forbidden)
    monkeypatch.setattr(n8n.N8nApiClient, "list_workflows", forbidden)
    monkeypatch.setattr(n8n.N8nApiClient, "latest_execution", forbidden)
    monkeypatch.setattr(n8n, "_live_check_rows", forbidden)
    monkeypatch.setattr(health, "read_runtime_status", lambda: {})
    monkeypatch.setattr(health, "get_running_pid", lambda cleanup_stale=False: None)

    response = TestClient(web_server.app).get("/api/cockpit/automation-health", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    n8n_cards = [card for card in body["cards"] if card["title"] == "Daily automation checks"]
    assert n8n_cards
    assert body["read_only"] is True
    assert body["actions_enabled"] is False

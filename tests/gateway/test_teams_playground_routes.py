"""Dashboard contract tests for the optional Teams Playground target."""

import pytest

import hermes_cli.web_routers.messaging as messaging
from hermes_cli.web_server_messaging import _messaging_platform_catalog


def test_teams_catalog_exposes_only_playground_settings_not_production_secrets():
    teams = next(entry for entry in _messaging_platform_catalog() if entry["id"] == "teams")
    assert "TEAMS_PLAYGROUND_URL" in teams["env_vars"]
    assert "TEAMS_PLAYGROUND_ALLOW_PRIVATE" in teams["env_vars"]
    assert "TEAMS_CLIENT_SECRET" in teams["required_env"]
    assert "TEAMS_PLAYGROUND_URL" not in teams["required_env"]


@pytest.mark.anyio
async def test_playground_health_route_returns_generated_urls_without_restart(monkeypatch):
    async def fake_handshake(config):
        return type("Result", (), {"ok": True, "message": "Playground health check passed."})()

    monkeypatch.setattr(messaging, "_playground_handshake", fake_handshake)
    monkeypatch.setattr(messaging, "_playground_payload", lambda _profile: {
        "enabled": True, "base_url": "http://localhost:3979",
        "test_url": "http://localhost:3979/hermes/teams-playground",
        "callback_url": "http://localhost:3979/api/messages",
    })
    result = await messaging.test_teams_playground()
    assert result == {
        "ok": True,
        "message": "Playground health check passed.",
        "test_url": "http://localhost:3979/hermes/teams-playground",
        "callback_url": "http://localhost:3979/api/messages",
    }


@pytest.mark.anyio
async def test_playground_health_route_does_not_restart_or_expose_secrets(monkeypatch):
    async def fake_handshake(config):
        return type("Result", (), {"ok": False, "message": "Playground handshake timed out."})()

    monkeypatch.setattr(messaging, "_playground_handshake", fake_handshake)
    monkeypatch.setattr(messaging, "_playground_payload", lambda _profile: {
        "enabled": True, "base_url": "http://localhost:3979",
        "test_url": "http://localhost:3979/hermes/teams-playground",
        "callback_url": "http://localhost:3979/api/messages",
    })
    result = await messaging.test_teams_playground()
    assert result["ok"] is False
    assert "secret" not in str(result).lower()
    assert "restart" not in str(result).lower()

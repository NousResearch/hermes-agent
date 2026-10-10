"""The messaging list loads config once per request. Regression for #131658."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway import config as gateway_config
from hermes_cli.web_routers import messaging
from hermes_constants import get_hermes_home
from hermes_cli.web_server_profiles import _hermes_home_scope


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(messaging, "resolve_gateway_liveness", lambda **kw: SimpleNamespace(running=False))
    monkeypatch.setattr(messaging, "multiplexer_liveness_for_profile", lambda *a: None)
    monkeypatch.setattr(messaging, "get_runtime_status_running_pid", lambda *a, **kw: None)
    monkeypatch.setattr(messaging, "read_runtime_status", lambda **kw: None)
    monkeypatch.setattr(messaging, "_GATEWAY_HEALTH_URL", None)
    catalog = [{"id": name, "name": name, "description": "", "docs_url": "",
                "required_env": ["DISCORD_BOT_TOKEN"], "env_vars": ["DISCORD_BOT_TOKEN"]}
               for name in ("discord", "telegram")]
    monkeypatch.setattr(messaging, "_messaging_platform_catalog", lambda: catalog)
    app = FastAPI()
    app.include_router(messaging.router)
    return TestClient(app)


def test_list_loads_gateway_config_once_and_refreshes(client, monkeypatch):
    real_load = gateway_config.load_gateway_config
    calls = []
    def load():
        calls.append(get_hermes_home())
        return real_load()
    monkeypatch.setattr(gateway_config, "load_gateway_config", load)
    for expected in (1, 2):
        response = client.get("/api/messaging/platforms")
        assert response.status_code == 200
        assert len(response.json()["platforms"]) == 2
        assert len(calls) == expected


def test_failed_load_is_not_retried_per_row_and_next_request_recovers(client, monkeypatch):
    calls = []
    def load():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("fixture unavailable")
        return gateway_config.GatewayConfig()
    monkeypatch.setattr(gateway_config, "load_gateway_config", load)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "fixture-token")
    response = client.get("/api/messaging/platforms")
    assert response.status_code == 200
    assert all(row["configured"] and not row["enabled"] for row in response.json()["platforms"])
    assert len(calls) == 1
    assert client.get("/api/messaging/platforms").status_code == 200
    assert len(calls) == 2


def test_scoped_real_config_is_request_local_a_b_a(client, tmp_path, monkeypatch):
    real_load = messaging.load_config
    calls = []
    def load():
        calls.append(get_hermes_home())
        return real_load()
    monkeypatch.setattr(messaging, "load_config", load)
    monkeypatch.setattr(gateway_config, "load_gateway_config", lambda: pytest.fail("scoped read used process config"))
    homes = [tmp_path / name for name in ("a", "b")]
    for home, enabled in zip(homes, ("true", "false")):
        home.mkdir()
        (home / "config.yaml").write_text(f"platforms:\n  discord:\n    enabled: {enabled}\n", encoding="utf8")
        (home / ".env").write_text("DISCORD_BOT_TOKEN=fixture\n", encoding="utf8")
    for home, enabled in ((homes[0], True), (homes[1], False), (homes[0], True)):
        with _hermes_home_scope(home):
            assert real_load()["platforms"]["discord"]["enabled"] is enabled
            rows = messaging._platform_payloads(home, messaging._messaging_platform_catalog())
        assert rows[0]["enabled"] is enabled
        assert rows[0]["configured"] is True
    assert calls == [homes[0], homes[1], homes[0]]


def test_scoped_load_failure_preserves_env_fallback(client, tmp_path, monkeypatch):
    calls = []
    def load():
        calls.append(1)
        raise ValueError("fixture corrupt")
    monkeypatch.setattr(messaging, "load_config", load)
    monkeypatch.setattr(messaging, "load_env", lambda: {"DISCORD_BOT_TOKEN": "fixture"})
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "must-not-borrow")
    entries = messaging._messaging_platform_catalog()
    rows = messaging._platform_payloads(tmp_path, entries)
    assert len(calls) == 1
    assert all(row["configured"] and not row["enabled"] for row in rows)

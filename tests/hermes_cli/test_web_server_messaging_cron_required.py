"""Channels page: api_server cannot be disabled while a loopback-firing cron provider needs it.

The gateway force-starts the loopback api_server for such a provider
(``gateway/cron_loopback_listener.py``), so a stored disable would only make the page lie.
The PUT is rejected before any write; the builtin ticker keeps the toggle working.
"""
import pytest

import hermes_yaml as yaml


class _LoopbackProvider:
    fires_over_loopback = True
    name = "loopback-fake"

    def is_available(self):
        return True


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    import plugins.cron_providers as pc
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(
        pc, "load_cron_scheduler",
        lambda name: _LoopbackProvider() if name == "loopback-fake" else None)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _set_provider(provider):
    from hermes_constants import get_hermes_home

    cfg = {"cron": {"provider": provider}} if provider else {}
    path = get_hermes_home() / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _api_server(client):
    platforms = client.get("/api/messaging/platforms").json()["platforms"]
    return next(p for p in platforms if p["id"] == "api_server")


def test_disable_rejected_under_loopback_provider(client):
    config_path = _set_provider("loopback-fake")
    before = config_path.read_text(encoding="utf-8")

    resp = client.put("/api/messaging/platforms/api_server", json={"enabled": False})

    assert resp.status_code == 409
    assert "cron" in resp.json()["detail"]
    assert config_path.read_text(encoding="utf-8") == before  # nothing persisted

    payload = _api_server(client)
    assert payload["required_reason"]
    assert payload["enabled"] is True


def test_enable_and_env_edits_still_allowed_under_loopback_provider(client):
    _set_provider("loopback-fake")
    resp = client.put(
        "/api/messaging/platforms/api_server",
        json={"enabled": True, "env": {"API_SERVER_PORT": "8650"}})
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize("provider", [None, "builtin"])
def test_disable_allowed_under_builtin_ticker(client, provider):
    config_path = _set_provider(provider)

    resp = client.put("/api/messaging/platforms/api_server", json={"enabled": False})

    assert resp.status_code == 200, resp.text
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["platforms"]["api_server"]["enabled"] is False
    assert _api_server(client)["required_reason"] is None


def test_other_platforms_unaffected_by_loopback_provider(client):
    _set_provider("loopback-fake")
    resp = client.put("/api/messaging/platforms/webhook", json={"enabled": False})
    assert resp.status_code == 200, resp.text

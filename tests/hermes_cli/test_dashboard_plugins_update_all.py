"""The dashboard's "Update all" and "Auto-update" plugin controls, through the real app and routes."""

from __future__ import annotations

import pytest


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    starlette = pytest.importorskip("starlette.testclient")
    from hermes_agent import hermes_state
    from hermes_agent.hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from hermes_agent.hermes_constants import get_hermes_home

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    return starlette.TestClient(app, raise_server_exceptions=False,
                                headers={_SESSION_HEADER_NAME: _SESSION_TOKEN})


def test_auto_update_switch_round_trips_through_the_profile_config(client):
    from hermes_agent.hermes_cli.config import load_config

    assert client.get("/api/dashboard/plugins/auto-update").json()["auto_apply"] is False
    on = client.put("/api/dashboard/plugins/auto-update", json={"enabled": True})
    assert on.status_code == 200 and on.json()["auto_apply"] is True, on.text
    assert load_config()["plugins"]["auto_apply"] is True
    assert client.get("/api/dashboard/plugins/auto-update").json()["auto_apply"] is True


def test_auto_update_refused_on_a_managed_install_is_a_400_with_the_reason(client, monkeypatch):
    from hermes_agent.hermes_cli import config

    monkeypatch.setattr(config, "is_managed", lambda: True)
    resp = client.put("/api/dashboard/plugins/auto-update", json={"enabled": True})
    assert resp.status_code == 400 and "package manager" in resp.json()["detail"]


def test_update_all_is_not_captured_by_the_named_plugin_routes(client):
    resp = client.post("/api/dashboard/agent-plugins/update-all")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"results": [], "skipped": [], "current": 0}

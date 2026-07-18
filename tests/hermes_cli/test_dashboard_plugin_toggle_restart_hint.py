"""Dashboard plugin toggle responses surface ``restart_required`` (#54941).

Plugin enable/disable writes config only — running gateway/TUI processes
scan plugins once at start and never see the change, so the dashboard must
tell the user a restart is needed (the CLI already prints "Takes effect on
next session."). These tests pin the endpoint-layer contract: a toggle that
changed state reports ``restart_required: true``; a no-op toggle reports
``restart_required: false``.

The underlying config write (``dashboard_set_agent_plugin_enabled``) is
stubbed — its own behaviour is covered by tests/hermes_cli/test_plugins_cmd
tests; here we only assert the HTTP layer's annotation.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hermes_cli import plugins_cmd, web_server
from hermes_cli.dashboard_auth import clear_providers


@pytest.fixture
def loopback_client():
    clear_providers()
    prev_host = getattr(web_server.app.state, "bound_host", None)
    prev_port = getattr(web_server.app.state, "bound_port", None)
    prev_required = getattr(web_server.app.state, "auth_required", None)
    web_server.app.state.bound_host = "127.0.0.1"
    web_server.app.state.bound_port = 8080
    web_server.app.state.auth_required = False
    client = TestClient(
        web_server.app,
        base_url="http://127.0.0.1:8080",
        headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN},
    )
    yield client
    web_server.app.state.bound_host = prev_host
    web_server.app.state.bound_port = prev_port
    web_server.app.state.auth_required = prev_required


def _stub_toggle(monkeypatch, *, unchanged: bool):
    calls = []

    def fake(name, *, enabled):
        calls.append((name, enabled))
        return {"ok": True, "name": "web/exa", "unchanged": unchanged}

    monkeypatch.setattr(
        plugins_cmd, "dashboard_set_agent_plugin_enabled", fake
    )
    return calls


def test_enable_change_reports_restart_required(loopback_client, monkeypatch):
    calls = _stub_toggle(monkeypatch, unchanged=False)
    resp = loopback_client.post("/api/dashboard/agent-plugins/web/exa/enable")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["restart_required"] is True
    assert calls == [("web/exa", True)]


def test_disable_change_reports_restart_required(loopback_client, monkeypatch):
    calls = _stub_toggle(monkeypatch, unchanged=False)
    resp = loopback_client.post("/api/dashboard/agent-plugins/web/exa/disable")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["restart_required"] is True
    assert calls == [("web/exa", False)]


def test_noop_toggle_reports_no_restart(loopback_client, monkeypatch):
    _stub_toggle(monkeypatch, unchanged=True)
    resp = loopback_client.post("/api/dashboard/agent-plugins/web/exa/enable")
    assert resp.status_code == 200
    assert resp.json()["restart_required"] is False


def test_failed_toggle_has_no_restart_field(loopback_client, monkeypatch):
    def fake(name, *, enabled):
        return {"ok": False, "error": "Plugin 'nope' is not installed or bundled."}

    monkeypatch.setattr(
        plugins_cmd, "dashboard_set_agent_plugin_enabled", fake
    )
    resp = loopback_client.post("/api/dashboard/agent-plugins/nope/disable")
    assert resp.status_code == 400

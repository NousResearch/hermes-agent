from __future__ import annotations

from fastapi.testclient import TestClient


def test_plugins_inventory_api_requires_session_token(monkeypatch):
    from hermes_cli import web_server

    monkeypatch.setattr(
        "hermes_cli.plugins.list_plugin_inventory",
        lambda: {
            "schema_version": 1,
            "metadata": {},
            "summary": {"total": 0},
            "plugins": [],
            "warnings": [],
        },
    )

    client = TestClient(web_server.app)

    unauthorized = client.get("/api/plugins/inventory")
    assert unauthorized.status_code == 401

    authorized = client.get(
        "/api/plugins/inventory",
        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
    )
    assert authorized.status_code == 200
    assert authorized.json()["schema_version"] == 1
    assert authorized.json()["plugins"] == []

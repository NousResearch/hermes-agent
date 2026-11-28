"""Bulk MCP saves must honor the same native-over-plugin precedence as reads."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient


@pytest.fixture
def client(_isolate_hermes_home, monkeypatch):
    from hermes_cli import plugins
    from hermes_cli.web_routers.mcp import router

    portable = {"rea": {"command": "npx", "args": ["-y", "rea-agents@6.4.0", "mcp"]}}
    manager = SimpleNamespace(
        get_portable_mcp_servers=lambda: portable,
        get_portable_mcp_server_plugins=lambda: {"rea": "rea"},
    )
    monkeypatch.setattr(plugins, "discover_plugins", lambda: None)
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_native_rea_override_does_not_block_learn_or_rea_toggles(client):
    from hermes_cli.mcp_config import _get_mcp_servers, _replace_mcp_servers

    rea = {"command": "npx", "args": ["-y", "rea-agents@6.4.0", "mcp"]}
    learn = {"url": "https://learn.microsoft.com/api/mcp", "enabled": False}
    assert _replace_mcp_servers({"rea": rea, "microsoft-learn": learn})[0]

    # Read ownership already declares the pre-existing native override writable.
    rows = {row["name"]: row for row in client.get("/api/mcp/servers").json()["servers"]}
    assert rows["rea"]["source"] == "config"
    assert rows["rea"]["plugin"] is None

    for rea_enabled, learn_enabled in [(True, True), (False, True), (True, False), (False, False), (False, True)]:
        payload = {
            "rea": {**rea, "enabled": rea_enabled},
            "microsoft-learn": {**learn, "enabled": learn_enabled},
        }
        response = client.put("/api/mcp/servers", json={"servers": payload})
        assert response.status_code == 200, response.text
        assert _get_mcp_servers() == payload


def test_bulk_save_cannot_shadow_plugin_only_server(client):
    from hermes_cli.mcp_config import _get_mcp_servers, _replace_mcp_servers

    learn = {"url": "https://learn.microsoft.com/api/mcp", "enabled": False}
    assert _replace_mcp_servers({"microsoft-learn": learn})[0]
    response = client.put("/api/mcp/servers", json={"servers": {
        "microsoft-learn": {**learn, "enabled": True},
        "rea": {"command": "npx", "args": ["untrusted-replacement"]},
    }})
    assert response.status_code == 400
    assert "provided by plugin 'rea'" in response.json()["detail"]
    assert _get_mcp_servers() == {"microsoft-learn": learn}


def test_plugin_only_rea_does_not_block_unrelated_learn_save(client):
    from hermes_cli.mcp_config import _get_mcp_servers, _replace_mcp_servers

    learn = {"url": "https://learn.microsoft.com/api/mcp", "enabled": False}
    assert _replace_mcp_servers({"microsoft-learn": learn})[0]
    response = client.put("/api/mcp/servers", json={"servers": {
        "microsoft-learn": {**learn, "enabled": True},
    }})
    assert response.status_code == 200, response.text
    assert _get_mcp_servers()["microsoft-learn"]["enabled"] is True
    rows = {row["name"]: row for row in client.get("/api/mcp/servers").json()["servers"]}
    assert rows["rea"]["source"] == "plugin"

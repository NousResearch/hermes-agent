"""Protocol and confirmation contracts for live dashboard route remounting."""

from tui_gateway import server


def test_reload_routes_requires_supported_protocol_version(monkeypatch):
    import hermes_cli.web_server as web_server

    monkeypatch.setattr(
        web_server,
        "remount_dashboard_plugin_api_routes",
        lambda: {"ok": True, "mounted": ["fleet-graph"], "count": 1},
    )
    response = server.handle_request(
        {
            "id": "reload-unsupported-version",
            "method": "plugins.manage",
            "params": {
                "action": "reload_dashboard_routes",
                "confirm": True,
                "protocol_version": 999,
            },
        }
    )
    assert isinstance(response, dict)
    assert "error" in response


def test_reload_routes_returns_versioned_receipt(monkeypatch):
    import hermes_cli.web_server as web_server

    monkeypatch.setattr(
        web_server,
        "remount_dashboard_plugin_api_routes",
        lambda: {"ok": True, "mounted": ["fleet-graph"], "count": 1},
    )
    response = server.handle_request(
        {
            "id": "reload-v1",
            "method": "plugins.manage",
            "params": {
                "action": "reload_dashboard_routes",
                "confirm": True,
                "protocol_version": 1,
            },
        }
    )
    assert isinstance(response, dict)
    result = response.get("result")
    assert isinstance(result, dict)
    assert result["protocol"] == "fleet-graph.dashboard-routes"
    assert result["protocol_version"] == 1


def test_reload_routes_rejects_boolean_protocol_version(monkeypatch):
    import hermes_cli.web_server as web_server

    called = []
    monkeypatch.setattr(
        web_server,
        "remount_dashboard_plugin_api_routes",
        lambda: called.append(True) or {"ok": True, "mounted": []},
    )
    response = server.handle_request(
        {
            "id": "reload-bool-version",
            "method": "plugins.manage",
            "params": {
                "action": "reload_dashboard_routes",
                "confirm": True,
                "protocol_version": True,
            },
        }
    )
    assert isinstance(response, dict)
    assert response["error"]["code"] == 4020
    assert not called

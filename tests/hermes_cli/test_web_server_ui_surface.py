"""/api/status and the authenticated /api/host/identity advertise the exact UI surface served."""

import gateway.status as _gw_status


def test_status_advertises_exact_ui_surface(monkeypatch):
    from starlette.testclient import TestClient

    import hermes_cli.web_server as ws

    client = TestClient(ws.app)
    client.headers[ws._SESSION_HEADER_NAME] = ws._SESSION_TOKEN
    surface = "webapp"
    monkeypatch.setattr(ws.app.state, "ui_surface", surface, raising=False)
    monkeypatch.setattr(_gw_status, "get_running_pid_cached", lambda: None)
    monkeypatch.setattr(_gw_status, "read_runtime_status", lambda: None)

    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["ui_surface"] == surface

    identity = client.get("/api/host/identity")
    assert identity.status_code == 200
    assert identity.json()["ui_surface"] == surface
    client.headers.pop(ws._SESSION_HEADER_NAME)
    assert client.get("/api/host/identity").status_code == 401

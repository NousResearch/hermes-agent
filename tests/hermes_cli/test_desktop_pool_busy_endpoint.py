from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli import web_server_idle_exit


def test_pool_busy_endpoint_requires_token(monkeypatch):
    token = "t" * 64
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    web_server.app.state.auth_required = False
    client = TestClient(web_server.app)

    assert client.get("/api/desktop/pool-busy").status_code == 401


def test_pool_busy_endpoint_reports_running_cron_job(monkeypatch):
    """#108863: the Electron pool idle reaper's only signal (`lastActiveAt`) can't
    see a cron job running with no chat window attached. This endpoint must report
    busy so the reaper does not reap the backend mid-run."""
    token = "t" * 64
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    web_server.app.state.auth_required = False
    monkeypatch.setattr(web_server_idle_exit, "turn_in_flight", lambda: True)
    client = TestClient(web_server.app)

    response = client.get("/api/desktop/pool-busy", headers={"X-Hermes-Session-Token": token})

    assert response.status_code == 200
    assert response.json() == {"busy": True}


def test_pool_busy_endpoint_reports_idle_when_no_turn_or_cron_running(monkeypatch):
    token = "t" * 64
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    web_server.app.state.auth_required = False
    monkeypatch.setattr(web_server_idle_exit, "turn_in_flight", lambda: False)
    client = TestClient(web_server.app)

    response = client.get("/api/desktop/pool-busy", headers={"X-Hermes-Session-Token": token})

    assert response.status_code == 200
    assert response.json() == {"busy": False}


def test_pool_busy_endpoint_fails_closed_when_probe_is_indeterminate(monkeypatch):
    """An indeterminate probe (e.g. the gateway session table is unreadable) must
    report busy=null, never busy=false — the caller (main.ts's isPoolBackendBusy)
    treats anything but an explicit `false` as busy."""
    token = "t" * 64
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", token)
    web_server.app.state.auth_required = False
    monkeypatch.setattr(web_server_idle_exit, "turn_in_flight", lambda: None)
    client = TestClient(web_server.app)

    response = client.get("/api/desktop/pool-busy", headers={"X-Hermes-Session-Token": token})

    assert response.status_code == 200
    assert response.json() == {"busy": None}

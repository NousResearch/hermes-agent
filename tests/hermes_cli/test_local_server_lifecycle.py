"""Server on/off lifecycle route (round-9 feedback: 'we should be able to
completely turn off the local engine'). Contract: stop tears the server
down AND persists enabled=false (durable, unlike eject); start persists
enabled=true and boots."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    from hermes_cli import web_server

    test_client = TestClient(web_server.app)
    token = getattr(web_server, "_SESSION_TOKEN", "")
    if token:
        test_client.headers["Authorization"] = f"Bearer {token}"
    return test_client


def test_stop_disables_and_tears_down(client, monkeypatch):
    stopped = {"called": False}

    def _shutdown():
        stopped["called"] = True

    class _FakeSup:
        pass

    monkeypatch.setattr("hermes_cli.local_runtime.bootstrap.get_supervisor",
                        lambda: _FakeSup())
    monkeypatch.setattr("hermes_cli.local_runtime.bootstrap.shutdown_local_runtime",
                        _shutdown)

    r = client.post("/api/local-models/server", json={"action": "stop"})
    assert r.status_code == 200
    assert stopped["called"] is True

    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["enabled"] is False


def test_start_enables_and_boots(client, monkeypatch):
    booted = {"called": False}

    class _FakeSup:
        base_url = "http://127.0.0.1:18434/v1"

    def _ensure(config, force=False):
        booted["called"] = True
        assert force is True
        return _FakeSup()

    monkeypatch.setattr("hermes_cli.local_runtime.bootstrap.ensure_local_runtime",
                        _ensure)

    r = client.post("/api/local-models/server", json={"action": "start"})
    assert r.status_code == 200
    assert booted["called"] is True

    from hermes_cli.config import load_config

    assert load_config()["local_runtime"]["enabled"] is True


def test_bogus_action_rejected(client):
    r = client.post("/api/local-models/server", json={"action": "reboot"})
    assert r.status_code == 400


def test_status_reports_loaded_models_from_live_router(client, monkeypatch):
    """Round-11 regression: the loaded-models read inside the status route
    raised NameError (missing json import), the blanket except swallowed it,
    and {} shipped as truth — 'Not in memory' on a machine with 30 GB of
    VRAM in use. This test exercises the REAL route against a stub router
    and demands the loaded set comes through."""
    import http.server
    import json as _json
    import threading

    class _Router(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = _json.dumps({"data": [
                {"id": "m-loaded", "status": {"value": "loaded"}},
                {"id": "m-loading", "status": {"value": "loading"}},
                {"id": "m-cold", "status": {"value": "unloaded"}},
            ]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Router)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        # Patch the ROUTE's binding: local_models binds _state_endpoint via
        # from-import at module load, so patching the endpoint module's
        # attribute never reaches the name the route actually calls.
        monkeypatch.setattr(
            "hermes_cli.web_routers.local_models._state_endpoint",
            lambda: {"base_url": f"http://127.0.0.1:{port}/v1", "api_key": "k"})
        payload = client.get("/api/local-models/status").json()
        assert payload["server_running"] is True
        assert payload["loaded_models"] == {"m-loaded": "loaded", "m-loading": "loading"}
    finally:
        server.shutdown()

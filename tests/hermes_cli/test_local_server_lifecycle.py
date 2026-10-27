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
    booted = {"called": False, "force": None}

    class _FakeSup:
        base_url = "http://127.0.0.1:18434/v1"

    def _ensure(config, force=False):
        booted["called"] = True
        booted["force"] = force
        return _FakeSup()

    monkeypatch.setattr("hermes_cli.local_runtime.bootstrap.ensure_local_runtime",
                        _ensure)

    r = client.post("/api/local-models/server", json={"action": "start"})
    assert r.status_code == 200
    assert booted["called"] is True
    # force=False: the route flips ``enabled`` via ``_set_runtime_enabled(True)`` first, and
    # ``_start_local_server`` then lets the gate inside ensure_local_runtime take over instead
    # of force-bypassing it. Otherwise any route that reused the helper would undo a user
    # "Turn off" click by force-spawning the server (regression test below).
    assert booted["force"] is False

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


@pytest.mark.parametrize("refuse", [False, True])
def test_stop_forwards_recovery_and_preserves_conflict(client, tmp_path, monkeypatch, refuse):
    from fastapi import HTTPException
    from hermes_cli.web_routers import local_models
    from hermes_cli.local_runtime import supervisor

    monkeypatch.setattr(supervisor, "runtimes_root", lambda: tmp_path)
    supervisor.state_path().write_text("{}")
    monkeypatch.setattr(local_models.bootstrap, "get_supervisor", lambda: None)
    monkeypatch.setattr(local_models, "_state_endpoint", lambda: None)
    called, disabled = [], []
    def recover():
        called.append(True)
        if refuse:
            raise HTTPException(409, "Another Hermes process owns this server, or its ownership could not be verified")
    monkeypatch.setattr(local_models, "_terminate_state_pid", recover)
    monkeypatch.setattr(local_models, "_set_runtime_enabled", lambda value: disabled.append(value))
    response = client.post("/api/local-models/server", json={"action": "stop"})
    assert response.status_code == (409 if refuse else 200), response.text
    assert called == [True]
    assert disabled == ([] if refuse else [False])


def test_start_local_server_honors_disabled_toggle(client, monkeypatch, tmp_path):
    """Regression for the "I turned it off and it came back" bug: when ``local_runtime.enabled``
    is False, ``_start_local_server`` (used by activate and quickstart) must NOT bypass the gate
    via ``force=True`` and must NOT silently flip the flag back to True. It raises a clear error
    telling the user to re-enable in the UI; only the explicit /server start route does that flip.
    """
    from hermes_cli.web_routers import local_models

    # User has turned the runtime off.
    monkeypatch.setattr(local_models, "_runtime_section",
                        lambda: {"enabled": False, "backend": "metal"})

    # _start_local_server should refuse BEFORE consulting ensure_local_runtime — ensure_local_runtime
    # itself is gated, but a force=True helper call would bypass it. With the fix, force is gone and
    # the section is checked up front.
    called_ensure = {"force": None, "config": None}

    def _ensure(config, force=False):
        called_ensure["force"] = force
        called_ensure["config"] = config
        return None

    monkeypatch.setattr(local_models.bootstrap, "ensure_local_runtime", _ensure)

    with pytest.raises(RuntimeError, match="turned off"):
        local_models._start_local_server(
            {"local_runtime": {"enabled": False, "backend": "metal"}},
            "server unreachable",
        )

    # ensure_local_runtime must NOT be called when disabled — the early raise must short-circuit.
    assert called_ensure["force"] is None, (
        "ensure_local_runtime was called even though the user disabled the runtime; the fix "
        "should reject up front instead of relying on the gate inside ensure_local_runtime")


def test_activate_does_not_reenable_when_disabled(client, monkeypatch):
    """Regression: ``POST /api/local-models/activate`` used to call ``_set_runtime_enabled(True)``
    in its job, undoing the user's "Turn off" click on the next mount or auto-activate. The flag
    is now only flipped by the explicit ``/api/local-models/server`` start action.
    """
    from hermes_cli.web_routers import local_models

    flipped = []

    def _set_enabled(value):
        flipped.append(value)
        return {"local_runtime": {"enabled": value}}

    monkeypatch.setattr(local_models, "_set_runtime_enabled", _set_enabled)
    # Don't actually start a server or hit a router — short-circuit the job's I/O.
    monkeypatch.setattr(local_models, "_ensure_server", lambda *a, **k: None)
    monkeypatch.setattr(local_models, "_assign_default", lambda *a, **k: None)
    monkeypatch.setattr(local_models, "_spawn_job",
                        lambda job, name, fn, **kw: fn() or True)
    monkeypatch.setattr(local_models, "_job", lambda *a, **k: {"started_at": 0})
    monkeypatch.setattr(local_models, "_step", lambda *a, **k: None)
    monkeypatch.setattr(local_models, "_finish", lambda *a, **k: None)

    response = client.post(
        "/api/local-models/activate",
        json={"model_id": "Qwen3.6-35B-A3B-UD-Q4_K_M"})

    # 200 because the job is mocked, but the meaningful assertion is the side-effect:
    # no True flip leaked through. Stop is the only path that sets False, start is the only
    # path that sets True; activate must not touch the flag.
    assert response.status_code in (200, 404), response.text
    assert True not in flipped, (
        "activate route flipped local_runtime.enabled to True; the explicit user Start button "
        "is the only path that should set the flag, otherwise the UI 'Turn off' toggle is undone")

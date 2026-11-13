"""Webapp SSH Test reaches the authenticated gateway without blocking the listener (#150863)."""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hermes_agent.hermes_cli import gateway_connections as store
from hermes_agent.hermes_cli import gateway_connections_probe as probe
from hermes_agent.hermes_cli import gateway_connections_ssh as ssh
from hermes_agent.hermes_cli import web_server


@pytest.fixture
def connection(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return store.save_connection({"kind": "ssh", "label": "Fixture SSH", "host": "fixture.example",
                                  "user": "fixture", "remoteProfile": "research"})[0]


@pytest.fixture
def status_server():
    seen = []
    state = {"status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("X-Hermes-Session-Token")))
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"version": "fixture-version"}).encode())

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield SimpleNamespace(base_url=f"http://127.0.0.1:{server.server_port}",
                                  token="fixture-tunnel-token", seen=seen, state=state)
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.asyncio
async def test_ssh_test_reaches_status_off_loop_and_preserves_selection(connection, status_server, monkeypatch):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    before = store.read_registry()

    def ensure(entry):
        assert entry["remoteProfile"] == "research"
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "SSH startup blocked the Webapp event loop"
        return status_server

    monkeypatch.setattr(ssh, "ensure_tunnel", ensure)
    task = asyncio.create_task(probe.probe_connection(connection["id"]))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert (await probe.probe_connection("local"))["ok"] is True
    finally:
        release.set()
        result = await task
    assert result == {"ok": True, "baseUrl": status_server.base_url, "version": "fixture-version"}
    assert status_server.seen == [("/api/status", status_server.token)]
    assert status_server.token not in json.dumps(result)
    assert store.read_registry() == before


@pytest.mark.parametrize("failure", ["auth-failed", "unreachable", "host-key-changed", "credential"])
def test_ssh_test_reports_reach_failures_and_keeps_the_webapp_gate(connection, status_server, monkeypatch, failure):
    for key in ("gateway_runner", "listener_public_url", "webapp_home"):
        monkeypatch.delattr(web_server.app.state, key, raising=False)
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "ui_surface", "webapp", raising=False)
    attempts = []
    message = "SSH authentication failed. Load the key into ssh-agent and retry."

    def ensure(entry):
        attempts.append(entry["id"])
        if failure != "credential":
            raise ssh.SshReachError(failure, message)
        status_server.state["status"] = 401
        return status_server

    monkeypatch.setattr(ssh, "ensure_tunnel", ensure)
    client = TestClient(web_server.app, raise_server_exceptions=False)
    path = f"/api/gateway-connections/{connection['id']}/test"
    assert client.post(path).status_code == 401
    assert attempts == []
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    response = client.post(path)
    assert response.status_code == (400 if failure == "credential" else 502)
    assert ("refused the saved credential" if failure == "credential" else message) in response.text
    assert status_server.token not in response.text
    assert attempts == [connection["id"]]
    assert client.post("/api/gateway-connections/missing/test").status_code == 404
    monkeypatch.setattr(web_server.app.state, "ui_surface", "serve")
    assert client.post(path).status_code == 404
    assert attempts == [connection["id"]]

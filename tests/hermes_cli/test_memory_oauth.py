"""Memory OAuth routes: the declared surface reports support, statuses carry only state/connected/auth."""

import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import memory_oauth


@pytest.fixture
def client(memory_homes, monkeypatch):
    (memory_homes["default"] / "plugins/silent").mkdir(parents=True)
    (memory_homes["default"] / "plugins/silent/__init__.py").write_text("# MemoryProvider\n")
    app = FastAPI()
    app.include_router(memory_oauth.router)
    with TestClient(app) as http:
        yield http


def _flow(status):
    return types.SimpleNamespace(start_loopback_flow_background=lambda: status, get_flow_status=lambda: status)


@pytest.mark.parametrize("method, endpoint", [("post", "start"), ("get", "status")])
def test_unknown_and_unsupported_providers(client, method, endpoint):
    call = getattr(client, method)
    assert call(f"/api/memory/providers/nope/oauth/{endpoint}").status_code == 404
    assert call(f"/api/memory/providers/nope/oauth/{endpoint}?surface=declared").status_code == 404
    assert call(f"/api/memory/providers/silent/oauth/{endpoint}").status_code == 404
    assert call(f"/api/memory/providers/silent/oauth/{endpoint}?surface=declared").json() == {
        "supported": False, "state": "unsupported", "connected": False, "auth": None, "detail": ""}


def test_status_keeps_only_state_connected_and_auth(client, monkeypatch):
    raw = {"state": "connected", "connected": True, "auth": "oauth", "detail": "token=secret", "workspace": "acme"}
    monkeypatch.setattr(memory_oauth, "_resolve_flow", lambda provider: _flow(raw))
    assert client.get("/api/memory/providers/silent/oauth/status").json() == {
        "state": "connected", "connected": True, "auth": "oauth", "detail": "Connected"}
    assert client.post("/api/memory/providers/silent/oauth/start?surface=declared").json() == {
        "state": "connected", "connected": True, "auth": "oauth", "detail": "Connected", "supported": True}
    monkeypatch.setattr(memory_oauth, "_resolve_flow", lambda provider: _flow({"state": "weird", "auth": "magic", "connected": "yes"}))
    assert client.get("/api/memory/providers/silent/oauth/status").json() == {
        "state": "error", "detail": "Authorization did not complete", "auth": None}

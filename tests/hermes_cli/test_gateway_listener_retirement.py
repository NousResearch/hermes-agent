"""The dashboard idle/retirement routes must not retire a gateway through its own listener.

The gateway mounts the dashboard app (``app.state.gateway_runner`` set). The idle proof reads a
standalone ``hermes serve`` child's ledgers only, so with an agent running on the gateway runner it
used to answer ``idle: true`` and ``prepare``/``commit`` took the fence: every later dashboard RPC,
cron tick and admission was refused while the gateway kept running. On the gateway listener the
routes now refuse and the fence stays open; a standalone serve keeps its permit protocol.
"""

from fastapi.testclient import TestClient

from tests.hermes_cli.native_gateway_client import native_desktop_client


def _fresh_fence(monkeypatch):
    from hermes_cli import backend_retirement
    from tui_gateway import server

    fence = backend_retirement.RetirementFence()
    monkeypatch.setattr(backend_retirement, "retirement", fence)
    monkeypatch.setitem(server._methods, "test.retirement", lambda rid, params: server._ok(rid, {}))
    return fence, server


RPC = {"id": "test", "method": "test.retirement", "params": {}}


def test_gateway_listener_is_never_idle_and_refuses_retirement(monkeypatch):
    fence, server = _fresh_fence(monkeypatch)
    client = native_desktop_client(monkeypatch)

    idle = client.get("/api/health/idle")
    assert idle.status_code == 200, idle.text
    assert idle.json() == {"ok": True, "idle": False, "reason": "gateway_owns_lifecycle"}

    for body in ({"action": "prepare"}, {"action": "commit", "token": "guessed"},
                 {"action": "cancel", "token": "guessed"}):
        refused = client.post("/api/health/retirement", json=body)
        assert refused.status_code == 200, refused.text
        assert refused.json() == {"ok": False, "idle": False, "reason": "gateway_owns_lifecycle"}

    # The fence was never taken: no permit, nothing committed, new work is still admitted.
    assert fence._token is None and not fence._committed
    with fence.work() as admitted:
        assert admitted
    assert "result" in server.handle_request(RPC)


def test_standalone_serve_keeps_the_permit_protocol(monkeypatch):
    fence, server = _fresh_fence(monkeypatch)
    from hermes_cli import web_server

    monkeypatch.setattr(web_server.app.state, "gateway_runner", None, raising=False)
    monkeypatch.setattr(web_server, "_SESSION_TOKEN", "standalone-retirement-token")
    client = TestClient(web_server.app)
    headers = {"X-Hermes-Session-Token": "standalone-retirement-token"}

    assert client.get("/api/health/idle", headers=headers).json() == {"ok": True, "idle": True, "reason": None}
    permit = client.post("/api/health/retirement", json={"action": "prepare"}, headers=headers).json()
    assert permit["ok"] is True and permit["idle"] is True and permit["token"]
    assert client.post("/api/health/retirement", json={"action": "commit", "token": permit["token"]},
                       headers=headers).json() == {"ok": True}
    assert "error" in server.handle_request(RPC)
    assert fence._committed


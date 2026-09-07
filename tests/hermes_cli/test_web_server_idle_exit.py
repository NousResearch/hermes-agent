"""A Desktop-owned ``serve --isolated`` backend over SSH must retire itself once no client has been
connected for the grace window and no turn is running (#101626): it is detached from any parent on
purpose, so the client count IS its liveness signal."""

from fastapi import FastAPI, WebSocket
from starlette.testclient import TestClient

import hermes_cli.web_server as ws_mod
from hermes_cli.web_server_idle_exit import (
    IdleClientTracker, should_exit_idle, start_idle_watchdog, wrap_asgi_with_ws_tracking)


def test_ws_sessions_are_counted_at_the_asgi_boundary_for_any_route():
    app = FastAPI()

    @app.websocket("/api/anything")
    async def anything(ws: WebSocket):
        await ws.accept()
        await ws.receive_text()
        await ws.close()

    @app.websocket("/api/refused")
    async def refused(ws: WebSocket):
        await ws.close(code=4401)  # never accepted: must not count

    tracker = IdleClientTracker(now=lambda: 0.0)
    client = TestClient(wrap_asgi_with_ws_tracking(app, tracker))
    with client.websocket_connect("/api/anything") as a:
        with client.websocket_connect("/api/anything") as b:
            assert tracker.live_count() == 2
            b.send_text("bye")
        assert tracker.live_count() == 1
        a.send_text("bye")
    assert tracker.live_count() == 0
    try:
        with client.websocket_connect("/api/refused"):
            pass
    except Exception:
        pass
    assert tracker.live_count() == 0  # a refused (never accepted) upgrade is not a client


def test_exit_only_after_grace_with_no_client_and_no_running_turn():
    clock = {"t": 0.0}
    tracker = IdleClientTracker(now=lambda: clock["t"])
    grace = 900.0
    assert should_exit_idle(tracker, grace, probe=lambda: False) is False  # just started
    clock["t"] = 901.0
    assert should_exit_idle(tracker, grace, probe=lambda: False) is True
    assert should_exit_idle(tracker, grace, probe=lambda: True) is False   # turn running
    assert should_exit_idle(tracker, grace, probe=lambda: None) is False   # indeterminate: fail closed
    tracker.on_open()
    clock["t"] = 5000.0
    assert should_exit_idle(tracker, grace, probe=lambda: False) is False  # a client is connected
    tracker.on_close()
    assert should_exit_idle(tracker, grace, probe=lambda: False) is False  # grace restarts on close
    clock["t"] = 5000.0 + grace + 1
    assert should_exit_idle(tracker, grace, probe=lambda: False) is True


def test_watchdog_sets_should_exit_and_only_arms_for_ssh_isolated_backends(monkeypatch):
    class _Server:
        should_exit = False

    server = _Server()
    clock = iter([0.0, 0.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0])
    tracker = IdleClientTracker(now=lambda: next(clock, 10_000.0))
    start_idle_watchdog(server, tracker, grace_s=1.0, poll_s=0.01, probe=lambda: False).join(timeout=5)
    assert server.should_exit is True

    # The uvicorn builder wraps the app + arms tunnel pings ONLY when a session token was handed over.
    monkeypatch.setattr(ws_mod.app.state, "auth_required", False, raising=False)
    plain, _ = ws_mod._build_uvicorn_server("127.0.0.1", 0)
    assert plain.ws_ping_interval is None and getattr(ws_mod.app.state, "ssh_isolated_clients", None) is None
    try:
        isolated, _ = ws_mod._build_uvicorn_server("127.0.0.1", 0, ssh_isolated=True)
        assert isolated.ws_ping_interval and isolated.ws_ping_timeout > isolated.ws_ping_interval
        assert isinstance(ws_mod.app.state.ssh_isolated_clients, IdleClientTracker)
    finally:
        ws_mod.app.state._state.pop("ssh_isolated_clients", None)  # process-global app: never leak the tracker

"""The Webapp hosts no embedded dashboard chat; it refuses that chat's sockets instead of half-starting it.

The dashboard's ``/chat`` tab runs the TUI in a PTY (``/api/pty``) that dials this process's legacy
``/api/ws``, and feeds a sidebar through ``/api/pub`` + ``/api/events``. A gateway-attached Webapp
relays ``/api/ws`` to the profile's gateway and closes a socket without the gateway subprotocol
(4426), so a TUI it spawned could never connect. It answers 4404 up front, as a dashboard with
embedded chat disabled does; the dashboard itself keeps the chat.
"""
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hermes_cli import web_server, web_server_chat
from hermes_cli.web_routers import chat_ws


@pytest.fixture
def surface(monkeypatch):
    spawned = []

    class Bridge:
        alive = True

        def read(self, timeout):
            return b""

        async def write(self, data):
            return True

        def resize(self, cols, rows):
            pass

        def is_alive(self):
            return self.alive

        def close(self):
            self.alive = False

    def spawn(argv, cwd=None, env=None):
        spawned.append(argv)
        return Bridge()

    async def argv(**_):
        return (["hermes", "--tui"], "/tmp", {})

    monkeypatch.setattr(web_server_chat.PtyBridge, "spawn", staticmethod(spawn))
    monkeypatch.setattr(web_server_chat, "_resolve_chat_argv_async", argv)
    for name in ("_ws_host_origin_reason", "_ws_client_reason"):
        monkeypatch.setattr(web_server_chat, name, lambda ws: None)
    monkeypatch.setattr(web_server_chat, "_ws_auth_reason", lambda ws, **_: (None, "test"))
    monkeypatch.setattr(chat_ws, "_ws_auth_ok", lambda ws, **_: True)
    monkeypatch.setattr(chat_ws, "_ws_request_is_allowed", lambda ws: True)
    monkeypatch.setattr(web_server_chat.PTY_REGISTRY, "_closed", False)

    def use(name):
        monkeypatch.setattr(web_server.app.state, "ui_surface", name, raising=False)
        return TestClient(web_server.app)

    yield use, spawned
    web_server_chat.PTY_REGISTRY._sessions.clear()


@pytest.mark.parametrize("route", ["/api/pty?channel=chat1", "/api/pub?channel=chat1", "/api/events?channel=chat1"])
def test_the_webapp_refuses_the_embedded_chat_sockets(surface, route):
    use, spawned = surface
    client = use("webapp")
    # Refused before the upgrade is accepted: the connect itself fails.
    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect(route):
            pass
    assert closed.value.code == 4404
    assert spawned == []


def test_the_dashboard_still_starts_its_embedded_chat(surface):
    use, spawned = surface
    client = use("dashboard")
    with client.websocket_connect("/api/pty?channel=chat1"):
        pass
    assert spawned and spawned[0][:2] == ["hermes", "--tui"]

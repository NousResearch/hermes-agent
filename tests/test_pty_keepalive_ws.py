import pytest

from hermes_cli import web_server


@pytest.mark.asyncio
async def test_attach_token_reuses_same_session(monkeypatch):
    """Two connects with the same ?attach= token hit one spawned bridge."""
    spawned = []

    class FakeBridge:
        def __init__(self):
            self.alive = True

        def read(self, timeout):
            return b""        # idle forever

        def write(self, data):
            pass

        def resize(self, cols, rows):
            pass

        def close(self):
            self.alive = False

    def fake_spawn(argv, cwd=None, env=None):
        b = FakeBridge()
        spawned.append(b)
        return b

    monkeypatch.setattr(web_server.PtyBridge, "spawn", staticmethod(fake_spawn))
    # bypass auth + argv resolution for the test
    monkeypatch.setattr(web_server, "_ws_auth_reason", lambda ws: (None, "test"))
    monkeypatch.setattr(web_server, "_ws_host_origin_reason", lambda ws: None)
    monkeypatch.setattr(web_server, "_ws_client_reason", lambda ws: None)

    async def fake_argv(**kw):
        return (["x"], "/tmp", {})

    monkeypatch.setattr(web_server, "_resolve_chat_argv_async", fake_argv)

    from starlette.testclient import TestClient

    try:
        client = TestClient(web_server.app)
        with client.websocket_connect("/api/pty?attach=TOK1") as ws1:
            ws1.send_bytes(b"hi")
        with client.websocket_connect("/api/pty?attach=TOK1") as ws2:
            ws2.send_bytes(b"again")
        assert len(spawned) == 1                # reattached, did not respawn
    finally:
        web_server.PTY_REGISTRY._sessions.clear()


@pytest.mark.asyncio
async def test_disconnect_during_snapshot_detaches_session(monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    class FakeWS:
        client = None
        query_params = {"attach": "TOK"}

        async def accept(self):
            pass

    class FakeSession:
        async def attach(self, ws):
            raise WebSocketDisconnect()

    class FakeRegistry:
        detached = None

        async def attach_or_spawn(self, key, *, spawn):
            return FakeSession(), False

        def detach(self, key, ws):
            self.detached = (key, ws)

    registry = FakeRegistry()
    ws = FakeWS()
    monkeypatch.setattr(web_server, "PTY_REGISTRY", registry)
    monkeypatch.setattr(web_server, "_ws_auth_reason", lambda ws: (None, "test"))
    monkeypatch.setattr(web_server, "_ws_host_origin_reason", lambda ws: None)
    monkeypatch.setattr(web_server, "_ws_client_reason", lambda ws: None)
    monkeypatch.setattr(web_server, "_channel_or_close_code", lambda ws: None)

    async def fake_argv(**kw):
        return (["x"], "/tmp", {})

    monkeypatch.setattr(web_server, "_resolve_chat_argv_async", fake_argv)

    await web_server.pty_ws(ws)  # type: ignore[arg-type]

    assert registry.detached == ("TOK", ws)

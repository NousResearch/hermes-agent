import json

import pytest

from hermes_cli import web_server


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


@pytest.fixture
def pty_keepalive_harness(monkeypatch):
    spawned = []

    def fake_spawn(argv, cwd=None, env=None):
        b = FakeBridge()
        spawned.append(argv)
        return b

    monkeypatch.setattr(web_server.PtyBridge, "spawn", staticmethod(fake_spawn))
    monkeypatch.setattr(web_server, "_ws_auth_reason", lambda ws: (None, "test"))
    monkeypatch.setattr(web_server, "_ws_host_origin_reason", lambda ws: None)
    monkeypatch.setattr(web_server, "_ws_client_reason", lambda ws: None)

    async def fake_argv(**kw):
        return (["x", kw.get("resume") or "fresh"], "/tmp", {})

    monkeypatch.setattr(web_server, "_resolve_chat_argv_async", fake_argv)

    try:
        yield spawned
    finally:
        web_server.PTY_REGISTRY._sessions.clear()


@pytest.mark.asyncio
async def test_attach_token_reuses_same_session(pty_keepalive_harness):
    """Two connects with the same ?attach= token hit one spawned bridge."""
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK1") as ws1:
        ws1.send_bytes(b"hi")
    with client.websocket_connect("/api/pty?attach=TOK1") as ws2:
        ws2.send_bytes(b"again")
    assert len(pty_keepalive_harness) == 1                # reattached, did not respawn


@pytest.mark.asyncio
async def test_attach_token_reuses_same_resume(pty_keepalive_harness):
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK1&resume=same") as ws1:
        ws1.send_bytes(b"hi")
    with client.websocket_connect("/api/pty?attach=TOK1&resume=same") as ws2:
        ws2.send_bytes(b"again")
    assert pty_keepalive_harness == [["x", "same"]]


@pytest.mark.asyncio
async def test_attach_token_does_not_reuse_different_resume(pty_keepalive_harness):
    """A keep-alive token must not pin /chat to an old selected session."""
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK1&resume=old") as ws1:
        ws1.send_bytes(b"hi")
    with client.websocket_connect("/api/pty?attach=TOK1&resume=new") as ws2:
        ws2.send_bytes(b"again")
    assert pty_keepalive_harness == [["x", "old"], ["x", "new"]]


@pytest.mark.asyncio
async def test_attach_token_does_not_reuse_different_profile(pty_keepalive_harness):
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK1&profile=alpha") as ws1:
        ws1.send_bytes(b"hi")
    with client.websocket_connect("/api/pty?attach=TOK1&profile=beta") as ws2:
        ws2.send_bytes(b"again")
    assert len(pty_keepalive_harness) == 2


@pytest.mark.asyncio
async def test_attach_token_reuses_default_chat_after_active_session_fallback(
    pty_keepalive_harness, tmp_path, monkeypatch
):
    from starlette.testclient import TestClient

    active_session_file = tmp_path / "active-session.json"
    monkeypatch.setattr(
        web_server,
        "_active_session_file_for_channel",
        lambda app, channel: active_session_file,
    )

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK1&channel=CHAT") as ws1:
        ws1.send_bytes(b"hi")

    active_session_file.write_text(json.dumps({"session_id": "existing"}))

    with client.websocket_connect("/api/pty?attach=TOK1&channel=CHAT") as ws2:
        ws2.send_bytes(b"again")

    assert pty_keepalive_harness == [["x", "fresh"]]

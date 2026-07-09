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
async def test_same_token_different_resume_spawns_distinct_session(monkeypatch):
    """Switching sessions (same tab token, new ?resume=) must spawn a fresh PTY.

    Regression for #61284: the registry was keyed only on the per-tab attach
    token, so selecting a different session reattached to the previous
    session's live PTY and the newly-requested ``resume`` was silently
    dropped — the terminal kept showing the old session.
    """
    spawned = []          # (argv-resume marker) per spawn

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

    resumes = []

    def fake_spawn(argv, cwd=None, env=None):
        b = FakeBridge()
        spawned.append(b)
        return b

    monkeypatch.setattr(web_server.PtyBridge, "spawn", staticmethod(fake_spawn))
    monkeypatch.setattr(web_server, "_ws_auth_reason", lambda ws: (None, "test"))
    monkeypatch.setattr(web_server, "_ws_host_origin_reason", lambda ws: None)
    monkeypatch.setattr(web_server, "_ws_client_reason", lambda ws: None)

    async def fake_argv(**kw):
        resumes.append(kw.get("resume"))
        return (["x"], "/tmp", {})

    monkeypatch.setattr(web_server, "_resolve_chat_argv_async", fake_argv)

    from starlette.testclient import TestClient

    try:
        client = TestClient(web_server.app)
        with client.websocket_connect("/api/pty?attach=TOK1&resume=SESS_A") as ws1:
            ws1.send_bytes(b"hi")
        with client.websocket_connect("/api/pty?attach=TOK1&resume=SESS_B") as ws2:
            ws2.send_bytes(b"again")
        assert len(spawned) == 2                 # distinct session => distinct PTY
        assert resumes == ["SESS_A", "SESS_B"]   # second PTY honours the new resume
    finally:
        web_server.PTY_REGISTRY._sessions.clear()


@pytest.mark.asyncio
async def test_same_token_same_resume_reattaches(monkeypatch):
    """Refreshing the SAME session (same token + same resume) reattaches."""
    spawned = []

    class FakeBridge:
        def __init__(self):
            self.alive = True

        def read(self, timeout):
            return b""

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
    monkeypatch.setattr(web_server, "_ws_auth_reason", lambda ws: (None, "test"))
    monkeypatch.setattr(web_server, "_ws_host_origin_reason", lambda ws: None)
    monkeypatch.setattr(web_server, "_ws_client_reason", lambda ws: None)

    async def fake_argv(**kw):
        return (["x"], "/tmp", {})

    monkeypatch.setattr(web_server, "_resolve_chat_argv_async", fake_argv)

    from starlette.testclient import TestClient

    try:
        client = TestClient(web_server.app)
        with client.websocket_connect("/api/pty?attach=TOK1&resume=SESS_A") as ws1:
            ws1.send_bytes(b"hi")
        with client.websocket_connect("/api/pty?attach=TOK1&resume=SESS_A") as ws2:
            ws2.send_bytes(b"again")
        assert len(spawned) == 1                 # same session => reattach
    finally:
        web_server.PTY_REGISTRY._sessions.clear()

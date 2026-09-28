import json

import pytest

from hermes_cli import web_server
import hermes_cli.web_server_chat as _web_server_chat


class FakeBridge:
    def __init__(self):
        self.alive = True
        self.accept_input = True
        self.written = bytearray()
        self.resized = None

    def read(self, timeout):
        return b""        # idle forever

    async def write(self, data):
        if not self.accept_input:
            return False
        self.written.extend(data)
        return True

    def resize(self, cols, rows):
        self.resized = (cols, rows)

    def close(self):
        self.alive = False


@pytest.fixture
def pty_keepalive_harness(monkeypatch):
    class Spawned(list):
        pass

    spawned = Spawned()
    spawned.bridges = []

    def fake_spawn(argv, cwd=None, env=None):
        b = FakeBridge()
        spawned.append(argv)
        spawned.bridges.append(b)
        return b

    monkeypatch.setattr(_web_server_chat.PtyBridge, "spawn", staticmethod(fake_spawn))
    monkeypatch.setattr(_web_server_chat, "_ws_auth_reason", lambda ws: (None, "test"))
    monkeypatch.setattr(_web_server_chat, "_ws_host_origin_reason", lambda ws: None)
    monkeypatch.setattr(_web_server_chat, "_ws_client_reason", lambda ws: None)

    async def fake_argv(**kw):
        resume = "child" if kw.get("resume") == "parent" else kw.get("resume")
        env = {"HERMES_TUI_RESUME": resume} if resume else {}
        return (["x", resume or "fresh"], "/tmp", env)

    monkeypatch.setattr(_web_server_chat, "_resolve_chat_argv_async", fake_argv)

    try:
        yield spawned
    finally:
        _web_server_chat.PTY_REGISTRY._sessions.clear()


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
    assert bytes(pty_keepalive_harness.bridges[0].written) == b"hi\x0cagain"


@pytest.mark.asyncio
async def test_stalled_input_closes_only_the_keepalive_socket(
    pty_keepalive_harness,
):
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK1") as ws:
        bridge = pty_keepalive_harness.bridges[0]
        bridge.accept_input = False
        ws.send_bytes(b"input")
        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_bytes()

    assert exc_info.value.code == 1013
    assert web_server.PTY_REGISTRY._sessions["TOK1"].alive is False


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
async def test_attach_token_reuses_canonical_resume(pty_keepalive_harness):
    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK1&resume=parent") as ws1:
        ws1.send_bytes(b"hi")
    with client.websocket_connect("/api/pty?attach=TOK1&resume=child") as ws2:
        ws2.send_bytes(b"again")
    assert pty_keepalive_harness == [["x", "child"]]




@pytest.mark.asyncio
async def test_attach_token_reuses_default_chat_after_active_session_fallback(
    pty_keepalive_harness, tmp_path, monkeypatch
):
    from starlette.testclient import TestClient

    active_session_file = tmp_path / "active-session.json"
    monkeypatch.setattr(
        _web_server_chat,
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


@pytest.mark.asyncio
async def test_inbound_frame_stamps_attached_session_activity(pty_keepalive_harness):
    """Every inbound frame is liveness proof — the resize keepalive included (#110849)."""
    import time as _time

    from starlette.testclient import TestClient

    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?attach=TOK1") as ws:
        deadline = _time.monotonic() + 2.0
        while "TOK1" not in _web_server_chat.PTY_REGISTRY._sessions and _time.monotonic() < deadline:
            _time.sleep(0.01)                     # registration happens in the handler task
        session = _web_server_chat.PTY_REGISTRY._sessions["TOK1"]
        before = session.last_activity

        ws.send_bytes(b"\x1b[RESIZE:80;24]")       # the resize keepalive: consumed, not written
        deadline = _time.monotonic() + 2.0
        while session.last_activity == before and _time.monotonic() < deadline:
            _time.sleep(0.01)                     # server-side frame handling is async

        assert session.last_activity > before
        assert session.bridge.resized == (80, 24)
        assert bytes(session.bridge.written) == b""


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, 600.0),        # key absent -> documented default
        (2, 120.0),
        ("5", 300.0),         # config.yaml values may arrive as strings
        (0, 0.0),             # explicit opt-out
        (-3, 0.0),
        ("nope", 600.0),      # garbage must not disable reaping or raise
    ],
)
def test_pty_attached_idle_ttl_reads_dashboard_config(monkeypatch, raw, expected):
    dashboard = {} if raw is None else {"pty_attached_idle_minutes": raw}
    monkeypatch.setattr(_web_server_chat, "load_config", lambda: {"dashboard": dashboard})
    assert _web_server_chat._pty_attached_idle_ttl() == expected


def test_pty_attached_idle_ttl_tolerates_missing_or_malformed_block(monkeypatch):
    for cfg in ({}, {"dashboard": "x"}, {"dashboard": None}):
        monkeypatch.setattr(_web_server_chat, "load_config", lambda: cfg)
        assert _web_server_chat._pty_attached_idle_ttl() == 600.0

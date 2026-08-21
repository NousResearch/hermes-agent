"""Focused tests for dashboard PTY reconnect breadcrumbs."""

import json
import time
from urllib.parse import urlencode

import pytest


class _OneFrameBridge:
    def __init__(self):
        self._sent = False
        self.closed = False

    @classmethod
    def spawn(cls, *args, **kwargs):
        return cls()

    def read(self, timeout):
        if not self._sent:
            self._sent = True
            return b"ready"
        return None

    def resize(self, *, cols, rows):
        pass

    def write(self, raw):
        pass

    def close(self):
        self.closed = True


@pytest.fixture
def pty_client(monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient

    import hermes_cli.web_server as ws
    from hermes_cli.pty_session import PtySessionRegistry

    monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    monkeypatch.setattr(ws, "_PTY_BRIDGE_AVAILABLE", True)
    monkeypatch.setattr(ws, "PtyBridge", _OneFrameBridge)
    monkeypatch.setattr(
        ws,
        "PTY_REGISTRY",
        PtySessionRegistry(
            ttl=1800,
            max_sessions=16,
            buffer_cap=1024,
            read_timeout=0.01,
        ),
    )
    ws.app.state.pty_active_session_files = {}

    client = TestClient(ws.app)
    return ws, client, ws._SESSION_TOKEN


def _url(token: str, **params: str) -> str:
    return f"/api/pty?{urlencode({'token': token, **params})}"






def test_fresh_param_ignores_channel_active_session_file(pty_client, monkeypatch):
    """Explicit fresh starts must not resurrect the prior channel session."""
    ws, client, token = pty_client
    channel = "fresh-chan"
    active_file = ws._active_session_file_for_channel(ws.app, channel)
    active_file.write_text(json.dumps({"session_id": "sess-old"}), encoding="utf-8")
    captured = {}

    def fake_resolve(resume=None, sidecar_url=None, profile=None, active_session_file=None):
        captured["active_session_file"] = active_session_file
        captured["resume"] = resume
        return (["fake-hermes-tui"], None, None)

    monkeypatch.setattr(ws, "_resolve_chat_argv", fake_resolve)

    with client.websocket_connect(_url(token, channel=channel, fresh="1")) as conn:
        assert conn.receive_bytes() == b"ready"

    assert captured["resume"] is None
    assert captured["active_session_file"] == str(active_file)
    assert not active_file.exists()


def test_child_eof_closes_socket_and_bridge(pty_client, monkeypatch):
    """Child EOF must close the WS server-side and reap the PTY.

    Regression for the FD leak (#54028): the reader task hits EOF when the
    PTY child exits, but if the browser's socket is half-open (no FIN), the
    writer loop's ``ws.receive()`` would block forever and the PTY fds would
    never be closed. The reader now closes the WebSocket on EOF so the
    handler's ``finally`` runs ``bridge.close()``.
    """
    ws, client, token = pty_client
    bridges = []

    class _RecordingBridge(_OneFrameBridge):
        @classmethod
        def spawn(cls, *args, **kwargs):
            b = cls()
            bridges.append(b)
            return b

    monkeypatch.setattr(ws.PtyBridge, "spawn", _RecordingBridge.spawn)
    monkeypatch.setattr(
        ws, "_resolve_chat_argv", lambda **kw: (["fake-hermes-tui"], None, None)
    )

    # The client never sends a disconnect of its own — it only reads the one
    # frame then the server side must tear everything down on child EOF.
    with client.websocket_connect(_url(token, channel="eof-chan")) as conn:
        assert conn.receive_bytes() == b"ready"
        # Server closes the socket after the child EOFs; receiving again
        # surfaces the close rather than hanging.
        with pytest.raises(Exception):
            conn.receive_bytes()

    assert len(bridges) == 1
    # bridge.close() runs in the handler's `finally` via asyncio.to_thread,
    # which can lag the client-side context exit by a tick or two. Poll briefly
    # instead of asserting immediately so the teardown isn't a race.
    deadline = time.monotonic() + 5.0
    while not bridges[0].closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert bridges[0].closed is True


def test_replay_cursor_is_strictly_validated_before_attach(pty_client, monkeypatch):
    from starlette.websockets import WebSocketDisconnect

    ws, client, token = pty_client
    monkeypatch.setattr(
        ws, "_resolve_chat_argv", lambda **kw: (["fake-hermes-tui"], None, None)
    )

    url = _url(
        token,
        channel="bad-cursor",
        attach="bad-cursor",
        epoch="0" * 32,
        offset="+1",
    )
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(url):
            pass
    assert exc_info.value.code == 4400


def test_websocket_reconnect_passes_byte_cursor_to_session(pty_client, monkeypatch):
    """The real /api/pty route resumes raw bytes through its query cursor."""
    ws, client, token = pty_client
    bridges = []

    class _RetainedBridge:
        def __init__(self):
            self._first = True
            self.closed = False

        @classmethod
        def spawn(cls, *args, **kwargs):
            bridge = cls()
            bridges.append(bridge)
            return bridge

        def read(self, timeout):
            if self._first:
                self._first = False
                return b"\xc3"
            return b""

        def resize(self, *, cols, rows):
            pass

        def write(self, raw):
            pass

        def close(self):
            self.closed = True

    monkeypatch.setattr(ws.PtyBridge, "spawn", _RetainedBridge.spawn)
    monkeypatch.setattr(
        ws, "_resolve_chat_argv", lambda **kw: (["fake-hermes-tui"], None, None)
    )

    base = {"channel": "cursor-chan", "attach": "cursor-session"}
    with client.websocket_connect(_url(token, **base)) as conn:
        first = conn.receive_json()
        assert first["reset"] is True
        assert conn.receive_bytes() == b"\xc3"

    # TestClient tears down the per-WebSocket event loop between contexts, so
    # inject detached output at the authoritative retained-buffer boundary.
    # The second connection still exercises the real route's query parsing and
    # attach() cursor propagation end to end.
    session = ws.PTY_REGISTRY._sessions["cursor-session"]
    session.buffer.append(b"\xa9\xff")

    with client.websocket_connect(
        _url(
            token,
            **base,
            epoch=first["epoch"],
            offset="1",
        )
    ) as conn:
        resumed = conn.receive_json()
        assert resumed["reset"] is False
        assert resumed["reason"] == "resume"
        assert resumed["start_offset"] == 1
        assert conn.receive_bytes() == b"\xa9\xff"

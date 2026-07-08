"""Focused tests for dashboard PTY reconnect breadcrumbs."""

import json
import sys
from pathlib import Path
from urllib.parse import urlencode

import pytest


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="PTY bridge is POSIX-only"
)


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

    prev_auth_required = getattr(ws.app.state, "auth_required", None)
    prev_bound_host = getattr(ws.app.state, "bound_host", None)
    prev_bound_port = getattr(ws.app.state, "bound_port", None)

    monkeypatch.setattr(ws, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    monkeypatch.setattr(ws.PtyBridge, "spawn", _OneFrameBridge.spawn)
    ws.app.state.auth_required = False
    # Starlette's TestClient uses Host: testserver for WebSocket handshakes.
    # Pin the bound host to that synthetic test host so these focused PTY tests
    # are isolated from dashboard-auth tests that mutate the shared app state.
    ws.app.state.bound_host = "testserver"
    ws.app.state.bound_port = 80
    ws.app.state.event_channels = {}
    ws.app.state.event_buffers = {}
    ws.app.state.event_publishers = {}
    ws.app.state.pty_active_session_files = {}

    client = TestClient(ws.app, base_url="http://127.0.0.1:9119")
    try:
        yield ws, client, ws._SESSION_TOKEN
    finally:
        ws.app.state.auth_required = prev_auth_required
        ws.app.state.bound_host = prev_bound_host
        ws.app.state.bound_port = prev_bound_port


def _url(token: str, **params: str) -> str:
    return f"/api/pty?{urlencode({'token': token, **params})}"


def test_resolve_chat_argv_sets_active_session_file_env(monkeypatch):
    """Dashboard chat gives the TUI a breadcrumb file for reconnect resume."""
    import hermes_cli.main as main_mod
    import hermes_cli.web_server as ws

    monkeypatch.setattr(
        main_mod,
        "_make_tui_argv",
        lambda project_root, tui_dev=False: (["node", "dist/entry.js"], "/tmp/ui-tui"),
    )

    _argv, _cwd, env = ws._resolve_chat_argv(
        active_session_file="/tmp/hermes-active-session.json"
    )

    assert env["HERMES_TUI_ACTIVE_SESSION_FILE"] == "/tmp/hermes-active-session.json"


def test_channel_reconnect_resumes_active_session_file(pty_client, monkeypatch):
    """A new /api/pty socket on the same channel resumes the last TUI sid."""
    ws, client, token = pty_client
    captured = []

    def fake_resolve(resume=None, sidecar_url=None, profile=None, active_session_file=None):
        captured.append(
            {
                "active_session_file": active_session_file,
                "resume": resume,
                "sidecar_url": sidecar_url,
            }
        )
        if active_session_file and not resume:
            Path(active_session_file).write_text(
                json.dumps({"session_id": "sess-live"}),
                encoding="utf-8",
            )
        return (["fake-hermes-tui"], None, None)

    monkeypatch.setattr(ws, "_resolve_chat_argv", fake_resolve)

    with client.websocket_connect(_url(token, channel="reconnect-chan")) as conn:
        assert conn.receive_bytes() == b"ready"

    with client.websocket_connect(_url(token, channel="reconnect-chan")) as conn:
        assert conn.receive_bytes() == b"ready"

    assert captured[0]["resume"] is None
    assert captured[0]["active_session_file"]
    assert captured[1]["resume"] == "sess-live"
    assert captured[1]["active_session_file"] == captured[0]["active_session_file"]


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


def test_event_subscriber_replays_frames_published_during_refresh_gap(pty_client):
    """A refreshed sidebar catches up with PTY events emitted while detached.

    The PTY byte stream already replays via PtySession's ring buffer. The
    structured sidecar (/api/pub -> /api/events) needs its own short replay
    buffer, otherwise a hard browser refresh can reattach the terminal but lose
    the live tool/progress/thinking events that drive the sidebar animation.
    """
    _ws, client, token = pty_client
    channel = "event-replay-chan"
    frame_1 = '{"jsonrpc":"2.0","method":"event","params":{"type":"tool.start"}}'
    frame_2 = '{"jsonrpc":"2.0","method":"event","params":{"type":"tool.complete"}}'

    with client.websocket_connect(f"/api/pub?token={token}&channel={channel}") as pub:
        # Simulate the refresh gap: the PTY-side publisher remains connected and
        # emits while the browser-side /api/events subscriber is absent.
        pub.send_text(frame_1)

        with client.websocket_connect(f"/api/events?token={token}&channel={channel}") as events:
            assert events.receive_text() == frame_1

            pub.send_text(frame_2)
            assert events.receive_text() == frame_2


def test_event_replay_skips_oversized_frames(pty_client, monkeypatch):
    """Refresh replay is bounded by bytes as well as frame count."""
    ws, client, token = pty_client
    monkeypatch.setattr(ws, "_EVENT_REPLAY_MAX_PAYLOAD_BYTES", 32)
    channel = "event-replay-size-chan"
    oversized = "x" * 128
    small = "ok"

    with client.websocket_connect(f"/api/pub?token={token}&channel={channel}") as pub:
        pub.send_text(oversized)
        pub.send_text(small)

        with client.websocket_connect(f"/api/events?token={token}&channel={channel}") as events:
            assert events.receive_text() == small


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
    import time

    deadline = time.monotonic() + 5.0
    while not bridges[0].closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert bridges[0].closed is True

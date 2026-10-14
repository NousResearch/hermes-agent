import asyncio
import json

import pytest

from hermes_cli import web_server
import hermes_cli.web_server_chat as _web_server_chat


class FakeBridge:
    def __init__(self):
        self.alive = True
        self.accept_input = True
        self.written = bytearray()

    def read(self, timeout):
        return b""        # idle forever

    async def write(self, data):
        if not self.accept_input:
            return False
        self.written.extend(data)
        return True

    def resize(self, cols, rows):
        pass

    def is_alive(self):
        return self.alive

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


# --- #63553: pty_active_session_files must not grow without bound -----------

def _pty_marker_dict():
    from hermes_cli.web_server import _get_pty_active_session_files

    return _get_pty_active_session_files(web_server.app)


@pytest.mark.asyncio
async def test_channel_marker_leaves_no_dict_entry_after_pty_lifecycle(pty_keepalive_harness):
    """The channel→marker mapping is gone once the keep-alive PTY is closed.

    Regression for #63553: `_active_session_file_for_channel` inserted a
    channel→tempfile entry that nothing ever popped, so every dashboard
    reconnect leaked one dict entry plus a forgotten tempfile+inode.
    """
    from starlette.testclient import TestClient

    markers = _pty_marker_dict()
    markers.clear()
    client = TestClient(web_server.app)

    with client.websocket_connect("/api/pty?attach=TOK63553&channel=CHAN63553") as ws:
        ws.send_bytes(b"hi")
    assert "CHAN63553" in markers  # marker registered while the PTY is live

    # End the session: closing the registry (same path as TTL reap or child
    # EOF) must drop both the tempfile and the dict entry.
    marker_path = markers["CHAN63553"]
    await web_server.PTY_REGISTRY.close_all()
    assert marker_path.exists() is False
    assert "CHAN63553" not in markers


@pytest.mark.asyncio
async def test_channel_marker_reconnect_reuses_one_entry(pty_keepalive_harness):
    """Repeated same-channel reconnects keep one mapping, not one per hop."""
    from starlette.testclient import TestClient

    markers = _pty_marker_dict()
    markers.clear()
    client = TestClient(web_server.app)
    attach = "TOK63553R"
    try:
        for _ in range(3):
            with client.websocket_connect(f"/api/pty?attach={attach}&channel=CHAN63553R") as ws:
                ws.send_bytes(b"hi")
        assert list(markers) == ["CHAN63553R"]
    finally:
        await web_server.PTY_REGISTRY.close_all()
    assert markers == {}


@pytest.mark.asyncio
async def test_legacy_channel_connect_pops_marker_on_disconnect(pty_keepalive_harness):
    """A 1:1 (no attach token) PTY dies with its socket; its marker must too."""
    from starlette.testclient import TestClient

    markers = _pty_marker_dict()
    markers.clear()
    client = TestClient(web_server.app)

    with client.websocket_connect("/api/pty?channel=LEGACY63553") as ws:
        ws.send_bytes(b"hi")
        assert "LEGACY63553" in markers

    # The handler pops the marker after _legacy_pump returns, which can lag the
    # client-side context exit by a tick — poll instead of asserting immediately.
    import time

    deadline = time.monotonic() + 5.0
    while "LEGACY63553" in markers and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert "LEGACY63553" not in markers


@pytest.mark.asyncio
async def test_legacy_marker_dropped_when_cancel_lands_mid_close(pty_keepalive_harness, monkeypatch):
    """The disconnect closes the bridge in a worker thread; cancelling the handler while that close is still
    running must not skip the marker discard."""
    import threading
    import time
    from starlette.testclient import TestClient

    close_started = threading.Event()

    def slow_close(self):
        close_started.set()
        time.sleep(0.5)  # still running when the client context exits and cancels the handler
        self.alive = False

    monkeypatch.setattr(FakeBridge, "close", slow_close)
    markers = _pty_marker_dict()
    markers.clear()
    client = TestClient(web_server.app)
    with client.websocket_connect("/api/pty?channel=LEGACYCANCEL") as ws:
        ws.send_bytes(b"hi")
        assert "LEGACYCANCEL" in markers
        ws.close()
        assert close_started.wait(5)  # teardown is inside bridge.close ...
    # ... and leaving the client context cancelled the handler there.
    deadline = time.monotonic() + 5.0
    while "LEGACYCANCEL" in markers and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert "LEGACYCANCEL" not in markers


@pytest.mark.asyncio
async def test_fresh_start_pops_stale_marker_and_reregisters_child_path(
    pty_keepalive_harness, tmp_path, monkeypatch
):
    """`?fresh=1` drops the stale breadcrumb AND keeps the new child's marker.

    Covers the resume regression flagged in #63563 review: popping the stale
    mapping must not leave the fresh child's path unmapped, or the next
    same-channel reconnect allocates a new path and loses the breadcrumb.
    """
    from starlette.testclient import TestClient

    markers = _pty_marker_dict()
    markers.clear()
    client = TestClient(web_server.app)

    with client.websocket_connect("/api/pty?attach=TOKFRESH&channel=CHANFRESH") as ws:
        ws.send_bytes(b"hi")
    stale_path = markers["CHANFRESH"]
    stale_path.write_text(json.dumps({"session_id": "stale"}))

    # Fresh start on the same channel: new PTY, new breadcrumb.
    with client.websocket_connect("/api/pty?attach=TOKFRESH2&channel=CHANFRESH&fresh=1") as ws:
        ws.send_bytes(b"hi")

    assert stale_path.exists() is False            # stale file removed
    assert markers["CHANFRESH"] is stale_path      # same mapping kept for the fresh child
    # The kept mapping is what lets the next same-channel reconnect resume the
    # fresh child instead of allocating a new unmapped path (#63563 review).
    with client.websocket_connect("/api/pty?attach=TOKFRESH2&channel=CHANFRESH") as ws:
        ws.send_bytes(b"hi")
    assert markers["CHANFRESH"] is stale_path
    await web_server.PTY_REGISTRY.close_all()
    assert markers == {}

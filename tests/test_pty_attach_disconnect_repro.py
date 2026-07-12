"""Regression: a browser closing while a keep-alive PTY replays its ring
buffer must not crash the ASGI handler.

Root cause (pre-fix): PtySession.attach() calls ``await ws.send_bytes(snap)``
to replay buffered output on (re)attach, with NO exception guard — unlike the
two sibling send sites (PtySession._drain and web_server._legacy_pump) which
both wrap send_bytes in try/except. When the client socket is already gone,
starlette raises WebSocketDisconnect, which propagated uncaught out of
web_server.pty_ws and up the uvicorn/starlette stack, spewing a traceback and
leaving the operator's terminal needing Ctrl-C.

Two layers are asserted here:
  1. attach() surfaces the disconnect (documents current behavior / the site).
  2. pty_ws swallows it and detaches cleanly (the fix contract).
"""
import asyncio

import pytest

from hermes_cli.pty_session import PtySession
from starlette.websockets import WebSocketDisconnect


class _FakeBridge:
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


class _DeadSocket:
    """Stub whose send_bytes fails exactly like a closed peer."""

    async def send_bytes(self, data):
        raise WebSocketDisconnect(code=1006)

    async def close(self, code=1000):
        pass


class _CountingSocket:
    """Records replayed bytes; used to prove the happy path still replays."""

    def __init__(self):
        self.sent = []

    async def send_bytes(self, data):
        self.sent.append(bytes(data))

    async def close(self, code=1000):
        pass


@pytest.mark.asyncio
async def test_attach_replays_buffer_on_live_socket():
    """Sanity: with buffered output and a live socket, attach() replays it.

    This is the precondition that makes the bug reachable — attach() only
    calls send_bytes when the ring buffer is non-empty.
    """
    session = PtySession("K", _FakeBridge(), buffer_cap=4096, read_timeout=0.05)
    session.buffer.append(b"buffered agent output\r\n")
    ws = _CountingSocket()

    await session.attach(ws)

    assert ws.sent == [b"buffered agent output\r\n"]
    assert session.attached is True


@pytest.mark.asyncio
async def test_attach_raises_on_dead_socket_during_replay():
    """attach() re-raises when the ring-buffer replay hits a dead socket, but
    first rolls the session back to a detached, reapable state — a dead socket
    must never be left looking live (that would block the reaper).
    """
    session = PtySession("K", _FakeBridge(), buffer_cap=4096, read_timeout=0.05)
    session.buffer.append(b"buffered agent output\r\n")
    ws = _DeadSocket()

    with pytest.raises(WebSocketDisconnect):
        await session.attach(ws)

    # Rolled back: not attached, no live socket, timestamped for the reaper.
    assert session.attached is False
    assert session._ws is None
    assert session.last_detached_at is not None
    # Buffer preserved so a genuine reattach still replays history.
    assert session.buffer.snapshot() == b"buffered agent output\r\n"


@pytest.mark.asyncio
async def test_pty_ws_attach_site_swallows_disconnect(monkeypatch):
    """The fix contract: web_server.pty_ws must NOT propagate a
    WebSocketDisconnect raised during session.attach(). It detaches cleanly.

    We exercise the real pty_ws handler up to the attach call by stubbing the
    auth/allow gates and the argv resolver, and forcing the spawned session's
    attach() to raise (dead-socket replay). A green result requires the
    try/except around ``await session.attach(ws)`` in pty_ws.
    """
    from hermes_cli import web_server as ws_mod

    # --- open the gates -------------------------------------------------
    monkeypatch.setattr(ws_mod, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    monkeypatch.setattr(ws_mod, "_PTY_BRIDGE_AVAILABLE", True)
    monkeypatch.setattr(ws_mod, "_ws_auth_reason", lambda ws: (None, "test"))
    monkeypatch.setattr(ws_mod, "_ws_host_origin_reason", lambda ws: None)
    monkeypatch.setattr(ws_mod, "_ws_client_reason", lambda ws: None)
    monkeypatch.setattr(ws_mod, "_channel_or_close_code", lambda ws: None)

    async def fake_argv(**kw):
        return (["/bin/true"], None, {})

    monkeypatch.setattr(ws_mod, "_resolve_chat_argv_async", fake_argv)

    # A session whose attach() blows up like a dead-socket replay.
    class _ExplodingSession:
        def __init__(self):
            self.bridge = _FakeBridge()
            self.detached_with = None

        async def attach(self, ws):
            raise WebSocketDisconnect(code=1006)

    exploding = _ExplodingSession()

    async def fake_attach_or_spawn(token, *, spawn):
        return exploding, True

    detach_calls = []
    monkeypatch.setattr(
        ws_mod.PTY_REGISTRY, "attach_or_spawn", fake_attach_or_spawn
    )
    monkeypatch.setattr(
        ws_mod.PTY_REGISTRY,
        "detach",
        lambda token, ws: detach_calls.append((token, ws)),
    )

    # Minimal WebSocket double covering everything pty_ws touches pre/at attach.
    class _WS:
        def __init__(self):
            self.client = type("C", (), {"host": "127.0.0.1"})()
            self.query_params = {"attach": "TOK"}
            self.app = ws_mod.app
            self.accepted = False
            self.closed_with = None

        async def accept(self):
            self.accepted = True

        async def send_text(self, text):
            pass

        async def close(self, code=1000, reason=None):
            self.closed_with = code

        async def receive(self):
            return {"type": "websocket.disconnect"}

    ws = _WS()

    # The whole point: this call must return without raising.
    await ws_mod.pty_ws(ws)

    assert ws.accepted is True
    assert detach_calls == [("TOK", ws)], (
        "pty_ws must detach the token after a failed attach so the reaper can "
        "reclaim the PTY; instead the disconnect escaped or detach was skipped."
    )

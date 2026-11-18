"""An idle HTTP connection to the dashboard or a listener survives a 5-10 s poll gap.

A reverse proxy or tunnel in front of the server reuses idle upstream connections. uvicorn's 5 s
default closed them between polls, so every poll opened a new connection. Real uvicorn servers on
loopback; one raw socket sends a request, idles past 5 s, and must be answered again on the same
connection.
"""

import socket
import threading
import time

import pytest

import hermes_cli.web_server as ws_mod
from hermes_cli.web_server_lifecycle import build_listener_server

_POLL_GAP_S = 6.5


async def _ok_app(scope, receive, send):
    if scope["type"] != "http":
        return
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]})
    await send({"type": "http.response.body", "body": b"ok"})


def _gateway_listener(_monkeypatch, port):
    return build_listener_server(_ok_app, "127.0.0.1", port, auth_required=False, lifespan="off")


def _dashboard_listener(monkeypatch, port):
    monkeypatch.setattr(ws_mod.app.state, "auth_required", False, raising=False)
    return ws_mod._build_uvicorn_server("127.0.0.1", port)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _get(conn: socket.socket, path: str) -> bytes:
    """One request on *conn*; the status line, or b"" when the server had closed the connection."""
    conn.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode())
    reply = b""
    while b"\r\n\r\n" not in reply:
        chunk = conn.recv(65536)
        if not chunk:
            return b""
        reply += chunk
    head, body = reply.split(b"\r\n\r\n", 1)
    length = next(int(line.split(b":", 1)[1]) for line in head.split(b"\r\n")
                  if line.lower().startswith(b"content-length:"))
    while len(body) < length:
        body += conn.recv(65536)
    return head.split(b"\r\n", 1)[0]


@pytest.mark.parametrize("build", [_gateway_listener, _dashboard_listener], ids=["gateway", "dashboard"])
def test_idle_connection_outlives_a_poll_gap(monkeypatch, build):
    port = _free_port()
    _config, server = build(monkeypatch, port)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started:
            assert time.monotonic() < deadline, "server did not start"
            time.sleep(0.05)
        with socket.create_connection(("127.0.0.1", port), timeout=10) as conn:
            assert _get(conn, "/api/health") == b"HTTP/1.1 200 OK"
            time.sleep(_POLL_GAP_S)  # health: allow HX013 -- the idle gap past uvicorn's 5 s default IS the behaviour under test
            again = _get(conn, "/api/health")
        assert again == b"HTTP/1.1 200 OK", "the server closed an idle connection before the next poll"
    finally:
        server.should_exit = True
        thread.join(timeout=10)

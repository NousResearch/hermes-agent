"""Tests for the /_cred/ management API on the proxy server.

These are integration tests that spin up a real proxy on a Unix socket in a
background thread, send management requests, and verify credentials land in
the daemon's store and are used for substitution.
"""

import asyncio
import http.client
import json
import socket
import threading

import pytest

from cred_proxy.server import run_proxy
from cred_proxy.store import CredStore


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection that connects via a Unix domain socket."""

    def __init__(self, socket_path: str) -> None:
        super().__init__("localhost")
        self._socket_path = socket_path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(5)
        self.sock.connect(self._socket_path)


@pytest.fixture
def proxy_socket(tmp_path):
    """Start a proxy server on a Unix socket in a background thread."""
    store = CredStore()
    sock_path = str(tmp_path / "test-proxy.sock")
    ready = threading.Event()

    def on_started(path: str) -> None:
        ready.set()

    def _run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            run_proxy(unix_socket=sock_path, on_started=on_started, store=store)
        )

    t = threading.Thread(target=_run_loop, daemon=True)
    t.start()

    assert ready.wait(timeout=5), "Proxy did not start in time"
    yield sock_path


def _req(sock_path: str, method: str, path: str, body: dict | None = None) -> dict:
    conn = _UnixHTTPConnection(sock_path)
    headers = {}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(data))
    conn.request(method, path, body=data, headers=headers)
    resp = conn.getresponse()
    result = json.loads(resp.read())
    conn.close()
    if resp.status >= 400:
        raise _HTTPError(resp.status, result)
    return result


class _HTTPError(Exception):
    def __init__(self, code: int, body: dict):
        self.code = code
        self.body = body


def test_list_empty_initially(proxy_socket):
    result = _req(proxy_socket, "GET", "/_cred/list")
    assert result == {"names": []}


def test_add_and_list(proxy_socket):
    result = _req(proxy_socket, "POST", "/_cred/add", {"name": "tok", "value": "secret"})
    assert result["stored"] is True
    assert result["name"] == "tok"

    result = _req(proxy_socket, "GET", "/_cred/list")
    assert "tok" in result["names"]


def test_add_multiple_and_list_sorted(proxy_socket):
    _req(proxy_socket, "POST", "/_cred/add", {"name": "zebra", "value": "z"})
    _req(proxy_socket, "POST", "/_cred/add", {"name": "alpha", "value": "a"})

    result = _req(proxy_socket, "GET", "/_cred/list")
    assert result["names"] == ["alpha", "zebra"]


def test_delete(proxy_socket):
    _req(proxy_socket, "POST", "/_cred/add", {"name": "temp", "value": "val"})
    result = _req(proxy_socket, "POST", "/_cred/delete", {"name": "temp"})
    assert result["deleted"] is True

    result = _req(proxy_socket, "GET", "/_cred/list")
    assert "temp" not in result["names"]


def test_delete_nonexistent_returns_404(proxy_socket):
    with pytest.raises(_HTTPError) as exc_info:
        _req(proxy_socket, "POST", "/_cred/delete", {"name": "ghost"})
    assert exc_info.value.code == 404


def test_add_then_proxy_substitutes(proxy_socket):
    """End-to-end: credential added via management API is used for substitution.

    We add a credential, then send a proxied HTTP request through the Unix
    socket with a placeholder in a header.  A tiny TCP listener verifies the
    proxy forwarded the request with the real secret substituted.
    """
    # Add credential via management API
    _req(proxy_socket, "POST", "/_cred/add", {"name": "e2e_tok", "value": "real-secret"})

    # Start a tiny TCP server to receive the proxied request
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.settimeout(5)
    listener.listen(1)
    target_port = listener.getsockname()[1]

    # Connect to proxy via Unix socket and send a proxied HTTP request
    proxy_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    proxy_sock.settimeout(5)
    proxy_sock.connect(proxy_socket)
    proxy_sock.sendall(
        f"GET http://127.0.0.1:{target_port}/test HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{target_port}\r\n"
        f"Authorization: Bearer hermes-proxy://e2e_tok\r\n"
        f"\r\n".encode()
    )

    # Accept and read what the proxy forwarded to the target
    conn, _ = listener.accept()
    conn.settimeout(5)
    received = conn.recv(4096).decode("latin-1")

    # Send a minimal response so the proxy completes cleanly
    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
    conn.close()
    proxy_sock.close()
    listener.close()

    # Verify the proxy substituted the placeholder with the real value
    assert "real-secret" in received
    assert "hermes-proxy://" not in received

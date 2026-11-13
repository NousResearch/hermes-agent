"""monitor_url sources must not reach private/internal services (#81406).

A real HTTP server on loopback plays both the "private service" and, via a fake public host whose
TCP connects are routed to it, the "public site". Only connects to the public name are rerouted;
SSRF policy still sees the public address for that name and the real loopback address for 127.0.0.1.
"""

from __future__ import annotations

import gzip
import socket
import threading
import tracemalloc
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

PUBLIC_HOST = "monitor-public.example"
PUBLIC_IP = "93.184.215.14"


def _gzip_zeros(size: int) -> bytes:
    co = zlib.compressobj(9, zlib.DEFLATED, 31)
    return b"".join(co.compress(bytes(1 << 20)) for _ in range(size >> 20)) + co.flush()


GZIP_BOMB = _gzip_zeros(32 << 20)  # ~32 KB on the wire, 32 MiB once decoded


@pytest.fixture
def local_server():
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{port}/secret")
                self.end_headers()
                return
            body, encoding = b"x" * 400_000 if self.path == "/big" else b"internal-secret", None
            if self.path == "/gzip-bomb":  # sent whatever the client asked for
                body, encoding = GZIP_BOMB, "gzip"
            elif self.path == "/negotiated" and "gzip" in (self.headers.get("Accept-Encoding") or ""):
                body, encoding = gzip.compress(b"internal-secret"), "gzip"
            self.send_response(200)
            if encoding:
                self.send_header("Content-Encoding", encoding)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield port, hits
    server.shutdown()
    server.server_close()


@pytest.fixture
def public_host(monkeypatch, local_server):
    """Make PUBLIC_HOST a public-address host whose connections land on the local server."""
    from tools import url_safety

    monkeypatch.delenv("HERMES_ALLOW_PRIVATE_URLS", raising=False)
    url_safety._reset_allow_private_cache()
    real_getaddrinfo = url_safety._getaddrinfo
    real_create_connection = socket.create_connection

    def fake_getaddrinfo(hostname, port=None):
        if hostname == PUBLIC_HOST:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IP, port or 0))]
        return real_getaddrinfo(hostname, port)

    def fake_create_connection(address, *args, **kwargs):
        host, port = address[0], address[1]
        if host in (PUBLIC_HOST, PUBLIC_IP):
            host = "127.0.0.1"
        return real_create_connection((host, port), *args, **kwargs)

    monkeypatch.setattr(url_safety, "_getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(socket, "create_connection", fake_create_connection)
    yield local_server
    url_safety._reset_allow_private_cache()


def test_monitor_url_rejects_direct_loopback(public_host):
    from hermes_agent.cron.monitor import _fetch_monitor_url

    port, hits = public_host
    ok, out = _fetch_monitor_url(f"http://127.0.0.1:{port}/secret")
    assert ok is False
    assert "monitor_url blocked:" in out
    assert "internal-secret" not in out
    assert hits == []


def test_monitor_url_rejects_redirect_to_loopback(public_host):
    from hermes_agent.cron.monitor import _fetch_monitor_url

    port, hits = public_host
    ok, out = _fetch_monitor_url(f"http://{PUBLIC_HOST}:{port}/redirect")
    assert ok is False
    assert "monitor_url blocked:" in out
    assert "internal-secret" not in out
    assert hits == ["/redirect"]


def test_monitor_url_public_fetch_still_works_and_is_capped(public_host):
    from hermes_agent.cron.monitor import MAX_URL_BYTES, _fetch_monitor_url

    port, _hits = public_host
    assert _fetch_monitor_url(f"http://{PUBLIC_HOST}:{port}/secret") == (True, "internal-secret")
    ok, out = _fetch_monitor_url(f"http://{PUBLIC_HOST}:{port}/big")
    assert ok is True
    assert len(out) == MAX_URL_BYTES


def test_monitor_url_does_not_inflate_a_compressed_body(public_host):
    from hermes_agent.cron.monitor import MAX_URL_BYTES, _fetch_monitor_url

    port, _hits = public_host
    tracemalloc.start()
    try:
        ok, out = _fetch_monitor_url(f"http://{PUBLIC_HOST}:{port}/gzip-bomb")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert ok is True
    assert len(out) <= MAX_URL_BYTES
    assert peak < 8 << 20


def test_monitor_url_asks_for_an_uncompressed_body(public_host):
    from hermes_agent.cron.monitor import _fetch_monitor_url

    port, _hits = public_host
    assert _fetch_monitor_url(f"http://{PUBLIC_HOST}:{port}/negotiated") == (True, "internal-secret")

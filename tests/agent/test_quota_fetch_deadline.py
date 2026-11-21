"""Dispatch 168 C2 — real aggregate deadline for the host-side quota fetch.

urllib's socket timeout bounds inactivity, not total elapsed time: bytes that
arrive incrementally can keep a request alive far past the timeout. These
tests use a localhost trickle server to prove the TOTAL budget is enforced.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import agent.quota_fetch as qf

SCOPE = "c" * 32
TOKEN = "synthetic-deadline-token-not-real"


def _payload():
    return {"data": {"free_model_daily_requests": {"limit": 10, "remaining": 4, "used": 6}}}


class _Trickle(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.started.set()
        raw = json.dumps(_payload()).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        if getattr(self.server, "fast", False):
            self.wfile.write(raw)
            self.wfile.flush()
            self.server.done.set()
            return
        # Each gap is far below the 0.5s socket timeout, but the sum exceeds
        # the total budget. Inactivity alone must NOT keep this request alive.
        try:
            for offset in range(0, len(raw), 5):
                self.wfile.write(raw[offset:offset + 5])
                self.wfile.flush()
                time.sleep(0.12)
        finally:
            self.server.done.set()


@pytest.fixture
def trickle(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Trickle)
    server.started = threading.Event()
    server.done = threading.Event()
    server.fast = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(qf, "OPENROUTER_KEY_URL", f"http://127.0.0.1:{server.server_port}/v1/key")
    monkeypatch.setattr(qf, "_OPENROUTER_HOSTS", {"127.0.0.1", "localhost"})
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_slow_trickle_respects_total_deadline(trickle):
    started = time.monotonic()
    result = qf.fetch_openrouter_key_limits(TOKEN, credential_scope_id=SCOPE, timeout=0.5)
    elapsed = time.monotonic() - started
    assert result["status"] == "unavailable"
    assert result["reason"] == "fetch_timeout"
    assert elapsed <= 1.5, f"total budget exceeded: {elapsed:.2f}s"
    assert trickle.started.is_set()
    assert TOKEN not in json.dumps(result)


def test_fast_response_still_succeeds(trickle):
    trickle.fast = True
    result = qf.fetch_openrouter_key_limits(TOKEN, credential_scope_id=SCOPE, timeout=2.0)
    assert result["status"] == "ok"
    assert (result["limit"], result["remaining"], result["used"]) == (10, 4, 6)
    assert trickle.done.wait(2)


def test_deadline_refusal_does_not_leave_client_worker(monkeypatch, trickle):
    before = set(t.ident for t in threading.enumerate())
    result = qf.fetch_openrouter_key_limits(TOKEN, credential_scope_id=SCOPE, timeout=0.5)
    after = set(t.ident for t in threading.enumerate())
    assert result["reason"] == "fetch_timeout"
    # The only extra thread is the mock server's per-request handler; wait for
    # it to finish, then confirm the client left no worker of its own behind.
    assert trickle.done.wait(3)
    for _ in range(30):
        if len(threading.enumerate()) <= len(before):
            break
        time.sleep(0.1)
    assert set(t.ident for t in threading.enumerate()) - before == set()


class _OversizeBody(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.requests.append(self.path)
        raw = b"x" * (qf.MAX_BODY_BYTES + 10)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class _Redirect(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.requests.append(self.path)
        self.send_response(302)
        self.send_header("Location", "http://127.0.0.1:9/steal")
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.mark.parametrize("handler", [_OversizeBody, _Redirect])
def test_body_cap_and_no_redirect_never_fabricate_allowance(monkeypatch, handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.requests = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(qf, "OPENROUTER_KEY_URL", f"http://127.0.0.1:{server.server_port}/v1/key")
    monkeypatch.setattr(qf, "_OPENROUTER_HOSTS", {"127.0.0.1", "localhost"})
    try:
        result = qf.fetch_openrouter_key_limits(TOKEN, credential_scope_id=SCOPE, timeout=2.0)
        assert result["status"] == "unavailable"
        assert "limit" not in result and "remaining" not in result
        assert len(server.requests) == 1, "no retries and no redirect follow"
        assert TOKEN not in json.dumps(result)
    finally:
        server.shutdown()
        server.server_close()


class _SlowHeaders(BaseHTTPRequestHandler):
    """Status line + headers in 25 fragments, 95 ms apart (below inactivity timeout)."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.requests.append(self.path)
        raw = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
               b"Content-Length: 2\r\nConnection: close\r\n\r\n{}")
        fragment = max(1, len(raw) // 25)
        try:
            for offset in range(0, len(raw), fragment):
                self.wfile.write(raw[offset:offset + fragment])
                self.wfile.flush()
                time.sleep(0.095)
        finally:
            self.server.done.set()


@pytest.fixture
def slow_headers(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowHeaders)
    server.requests = []
    server.done = threading.Event()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(qf, "OPENROUTER_KEY_URL", f"http://127.0.0.1:{server.server_port}/v1/key")
    monkeypatch.setattr(qf, "_OPENROUTER_HOSTS", {"127.0.0.1", "localhost"})
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_slow_header_fragments_respect_total_deadline(slow_headers):
    started = time.monotonic()
    result = qf.fetch_openrouter_key_limits(TOKEN, credential_scope_id=SCOPE, timeout=0.5)
    elapsed = time.monotonic() - started
    assert result["status"] == "unavailable"
    assert result["reason"] == "fetch_timeout"
    assert elapsed <= 1.2, f"aggregate header budget exceeded: {elapsed:.2f}s"
    assert len(slow_headers.requests) == 1, "no retry after a deadline refusal"
    assert "limit" not in result and "remaining" not in result, "no fabricated allowance"
    assert TOKEN not in json.dumps(result)


def test_dns_delay_is_bounded_before_any_connection(monkeypatch):
    # The resolver helper joins its daemon worker with the remaining budget;
    # even a 3 s resolver must not delay the refusal.
    monkeypatch.setattr(qf.socket, "getaddrinfo", lambda *a, **kw: time.sleep(3))
    started = time.monotonic()
    result = qf.fetch_openrouter_key_limits(TOKEN, credential_scope_id=SCOPE, timeout=0.5)
    elapsed = time.monotonic() - started
    assert result["status"] == "unavailable"
    assert result["reason"] == "fetch_timeout"
    assert elapsed <= 1.2, f"dns budget exceeded: {elapsed:.2f}s"
    assert TOKEN not in json.dumps(result)


def test_connect_delay_is_bounded(monkeypatch):
    # Simulate a connect syscall that blocks until the socket's (refreshed)
    # timeout expires instead of returning promptly.
    def slow_connect(self, address):
        end = time.monotonic() + (self.gettimeout() or 0)
        while time.monotonic() < end:
            time.sleep(0.02)
        raise TimeoutError("synthetic connect timeout")

    monkeypatch.setattr(socket.socket, "connect", slow_connect)
    monkeypatch.setattr(qf, "OPENROUTER_KEY_URL", "https://127.0.0.1:1/v1/key")
    started = time.monotonic()
    result = qf.fetch_openrouter_key_limits(TOKEN, credential_scope_id=SCOPE, timeout=0.5)
    elapsed = time.monotonic() - started
    assert result["status"] == "unavailable"
    assert result["reason"] == "fetch_timeout"
    assert elapsed <= 1.2, f"connect budget exceeded: {elapsed:.2f}s"


def test_tls_handshake_delay_is_bounded(monkeypatch):
    # A handshake that consumes the socket's remaining timeout must be cut off
    # by the same aggregate budget, not allowed to run for multiple seconds.
    class _FakeContext:
        def wrap_socket(self, sock, server_hostname=None, **kwargs):
            end = time.monotonic() + (sock.gettimeout() or 0)
            while time.monotonic() < end:
                time.sleep(0.02)
            raise TimeoutError("synthetic TLS timeout")

    monkeypatch.setattr(qf.ssl, "create_default_context", lambda: _FakeContext())
    monkeypatch.setattr(qf._DeadlineSocket, "connect", lambda self, address: None)
    monkeypatch.setattr(qf, "OPENROUTER_KEY_URL", "https://127.0.0.1:1/v1/key")
    started = time.monotonic()
    result = qf.fetch_openrouter_key_limits(TOKEN, credential_scope_id=SCOPE, timeout=0.5)
    elapsed = time.monotonic() - started
    assert result["status"] == "unavailable"
    assert result["reason"] == "fetch_timeout"
    assert elapsed <= 1.2, f"tls budget exceeded: {elapsed:.2f}s"

"""Dispatch 168 C2 — real aggregate deadline for the host-side quota fetch.

urllib's socket timeout bounds inactivity, not total elapsed time: bytes that
arrive incrementally can keep a request alive far past the timeout. These
tests use a localhost trickle server to prove the TOTAL budget is enforced.
"""
from __future__ import annotations

import json
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

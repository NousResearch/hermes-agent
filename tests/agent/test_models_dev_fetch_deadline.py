"""The models.dev download has a total wall-clock bound, not just a per-read timeout.

A registry that trickles bytes slower than the per-read timeout never trips it, and the cold-cache
fetch runs inside a caller's request (a fresh home's first ``session.create``), so the bound must
cover the whole exchange and fall back to the cache path the client already has.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import agent.models_dev as md

class _Trickle(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.gets += 1
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "1000000")
            self.end_headers()
            for _ in range(80):  # ~8 s of one byte per 0.1 s: far under any per-read timeout
                self.wfile.write(b" ")
                self.wfile.flush()
                if self.server.stop.wait(0.1):
                    return
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture
def trickling_registry(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Trickle)
    server.gets, server.stop = 0, threading.Event()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(md, "_get_models_dev_url", lambda: f"http://127.0.0.1:{server.server_port}/api.json")
    monkeypatch.setattr(md, "_MODELS_DEV_FETCH_DEADLINE", 0.5, raising=False)
    for name, value in (("_models_dev_cache", {}), ("_models_dev_cache_time", 0), ("_models_dev_retry_after", 0)):
        monkeypatch.setattr(md, name, value)
    yield server
    server.stop.set()
    # The abandoned download ends once the body stops; it must hand its slot back.
    assert md._models_dev_download_slot.acquire(timeout=10)
    md._models_dev_download_slot.release()
    server.shutdown()
    server.server_close()


def test_cold_fetch_from_a_trickling_registry_returns_within_the_deadline(trickling_registry):
    assert not md._get_cache_path().exists()
    started = time.monotonic()
    assert md.fetch_models_dev() == {}
    assert time.monotonic() - started < 2.0
    # The failure arms the backoff, and a forced refresh (which skips the backoff) does not stack a
    # second transfer while the abandoned one is still trickling.
    assert md.fetch_models_dev() == {}
    assert md.fetch_models_dev(force_refresh=True) == {}
    assert trickling_registry.gets == 1


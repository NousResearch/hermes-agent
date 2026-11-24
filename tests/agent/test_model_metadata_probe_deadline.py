"""Model-metadata probes have a total wall-clock bound, not just per-op httpx timeouts.

Probes run inside requests (a new session resolves its context window through them), so a
``/models`` body that trickles in under the read timeout must not hold the caller.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import agent.model_metadata as mm
import agent.model_metadata_http as http


class _Trickle(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
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
def trickling_endpoint(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Trickle)
    server.stop = threading.Event()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(http, "TOTAL_DEADLINE_S", 0.5, raising=False)
    yield f"http://127.0.0.1:{server.server_port}"
    server.stop.set()
    server.shutdown()
    server.server_close()


def test_endpoint_models_probe_on_a_trickling_server_returns_within_the_deadline(trickling_endpoint, monkeypatch):
    monkeypatch.setattr(mm, "is_local_endpoint", lambda _url: False)  # remote shape: /models probes only
    started = time.monotonic()
    assert mm.fetch_endpoint_model_metadata(trickling_endpoint + "/v1", force_refresh=True) == {}
    # Two candidates (/v1 and bare), each capped: well under the ~8 s the body takes to arrive.
    assert time.monotonic() - started < 3.0

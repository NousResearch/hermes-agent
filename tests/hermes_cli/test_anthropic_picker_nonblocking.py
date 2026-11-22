"""Cold Anthropic discovery leaves the shared picker response responsive."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hermes_cli import models as M


def test_cold_picker_returns_before_body_and_background_refresh_warms_cache(monkeypatch):
    headers_sent = threading.Event()
    release_body = threading.Event()
    stored = threading.Event()
    returned = threading.Event()
    cache = {}
    live = ["claude-quasar-6-1", "claude-opus-test"]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_GET(self):
            body = json.dumps({"data": [{"id": mid} for mid in live], "has_more": False}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            headers_sent.set()
            release_body.wait(10)
            self.wfile.write(body)

    def store(provider, row):
        cache[provider] = row
        stored.set()

    def read():
        M.cached_provider_model_ids("anthropic", non_blocking=True)
        returned.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    monkeypatch.setattr(M, "_load_provider_models_cache", lambda: dict(cache))
    monkeypatch.setattr(M, "_credential_fingerprint", lambda *_: "account")
    monkeypatch.setattr(M, "_store_cache_entry", store)
    monkeypatch.setattr(M, "_anthropic_models_url", lambda *_a, **_k: f"http://127.0.0.1:{server.server_port}/v1/models")
    monkeypatch.setattr("hermes_cli.runtime_provider.anthropic_discovery_token", lambda *_: "fake-console-key")
    monkeypatch.setattr(M, "_get_model_config_dict", dict)
    caller = threading.Thread(target=read, daemon=True)
    try:
        caller.start()
        assert headers_sent.wait(5), "discovery did not start"
        returned_before_body = returned.wait(2)
    finally:
        release_body.set()
        caller.join(5)
        assert stored.wait(5), "background discovery did not populate the cache"
        server.shutdown()
        server.server_close()
        serving.join(5)
    assert returned_before_body, "cache-only picker waited for the HTTP body"
    assert set(live).issubset(M.cached_provider_model_ids("anthropic", non_blocking=True))

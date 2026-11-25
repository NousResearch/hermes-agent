"""Regression for #159799: ``/model`` re-resolves the window of the route it switches to.

A local server restarted with a bigger window under the same model id kept being judged by the
memo of its old ``/models`` answer, so the next request failed the 64K minimum with the old
window. A switch now drops what is cached for that route.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import agent.model_metadata as mm
from agent.model_metadata_memo_bounds import forget_route_context_length
from hermes_cli.model_switch import switch_model

_VALID = {"accepted": True, "persist": True, "recognized": True, "message": None}


@pytest.fixture
def local_server():
    window = {"n": 32768}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.rstrip("/") not in ("/v1/models", "/models"):
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps({"object": "list", "data": [{"id": "stub-model", "max_model_len": window["n"]}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.send_response(404)
            self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()
    mm._LOCAL_CTX_PROBE_CACHE.clear()
    yield f"http://127.0.0.1:{server.server_port}/v1", window
    server.shutdown()


def _switch(base_url):
    return switch_model(
        raw_input="stub-model", current_provider="custom", current_model="stub-model",
        current_base_url=base_url, current_api_key="", explicit_provider="custom",
        user_providers={}, custom_providers=[])


def test_switching_to_a_restarted_server_reads_its_new_window(local_server, monkeypatch):
    base_url, window = local_server
    monkeypatch.setattr("hermes_cli.models_validate.validate_requested_model", lambda *a, **k: _VALID)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_info", lambda *a, **k: None)
    monkeypatch.setattr("hermes_cli.model_switch.get_model_capabilities", lambda *a, **k: None)
    assert mm.get_model_context_length("stub-model", base_url, provider="custom") == 32768

    window["n"] = 65536  # the server is restarted with a bigger window
    assert mm.get_model_context_length("stub-model", base_url, provider="custom") == 32768  # memo, no switch yet

    result = _switch(base_url)

    assert result.success is True and result.base_url == base_url
    assert mm.get_model_context_length("stub-model", base_url, provider="custom") == 65536


def test_forgetting_a_route_leaves_other_endpoints_memoized(local_server):
    base_url, _ = local_server
    other = ("http://10.0.0.9:8000/v1", "")
    mm._endpoint_model_metadata_cache[other] = {"m": {"context_length": 1}}
    mm._endpoint_model_metadata_cache_time[other] = 1e18
    mm.get_model_context_length("stub-model", base_url, provider="custom")

    forget_route_context_length("stub-model", base_url)

    assert all(key[0] != mm._normalize_base_url(base_url) for key in mm._endpoint_model_metadata_cache)
    assert other in mm._endpoint_model_metadata_cache

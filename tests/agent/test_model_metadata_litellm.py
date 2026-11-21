"""LiteLLM proxy ``/model/info`` vision probe against a real local HTTP server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent import model_metadata_litellm
from agent.image_routing import _probe_litellm

# Shape captured from a real `litellm --config` proxy (1.106.4): operator aliases, capabilities
# from LiteLLM's model map, ``supports_vision: null`` when the map does not say.
_MODEL_INFO = {"data": [
    {"model_name": "team-vision", "litellm_params": {"model": "openrouter/google/gemini-2.5-flash"},
     "model_info": {"key": "openrouter/google/gemini-2.5-flash", "supports_vision": True}},
    {"model_name": "team-text", "litellm_params": {"model": "openrouter/deepseek/deepseek-chat"},
     "model_info": {"key": "openrouter/deepseek/deepseek-chat", "supports_vision": None}},
    {"model_name": "team-blind", "model_info": {"supports_vision": False}},
    {"model_name": "team-mixed", "model_info": {"supports_vision": True}},
    {"model_name": "team-mixed", "model_info": {"supports_vision": False}},
]}


@pytest.fixture
def server():
    hits, state = [], {"litellm": True}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append((self.path, self.headers.get("Authorization")))
            if state["litellm"] and self.path == "/model/info":
                body = json.dumps(_MODEL_INFO).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
            else:
                body = b'{"detail":"Not Found"}'
                self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    model_metadata_litellm._MODEL_INFO_CACHE.clear()
    yield httpd, hits, state
    httpd.shutdown()
    model_metadata_litellm._MODEL_INFO_CACHE.clear()


def _cfg(httpd):
    return {"model": {"provider": "custom", "base_url": f"http://127.0.0.1:{httpd.server_port}/v1",
                      "api_key": "sk-test-not-real"}}


def test_custom_litellm_alias_resolves_vision_from_model_info(server):
    httpd, hits, _ = server
    cfg = _cfg(httpd)
    assert _probe_litellm("custom", "team-vision", cfg) is True
    assert _probe_litellm("custom", "team-blind", cfg) is False
    # Unreported, disagreeing, or unknown aliases stay unknown so the next source decides.
    assert _probe_litellm("custom", "team-text", cfg) is None
    assert _probe_litellm("custom", "team-mixed", cfg) is None
    assert _probe_litellm("custom", "nope", cfg) is None
    # One authenticated request at the server root serves every lookup in the TTL.
    assert hits == [("/model/info", "Bearer sk-test-not-real")]


def test_non_litellm_and_builtin_providers_are_not_misread(server):
    httpd, hits, state = server
    state["litellm"] = False
    cfg = _cfg(httpd)
    assert _probe_litellm("custom", "team-vision", cfg) is None
    assert _probe_litellm("custom", "team-vision", cfg) is None
    assert len(hits) == 1  # the 404 is memoized, not re-probed per image turn
    hits.clear()
    assert _probe_litellm("openrouter", "team-vision", cfg) is None
    assert hits == []

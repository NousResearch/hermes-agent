"""Tests for hermes_cli.azure_detect — transport & model auto-detection."""

from __future__ import annotations

import http.server
import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import azure_detect


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

class _FakeHTTPResponse:
    """Minimal stand-in for urllib.request.urlopen's context manager."""

    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._body


def _openai_models_body(*ids: str) -> bytes:
    return json.dumps({
        "object": "list",
        "data": [{"id": i, "object": "model"} for i in ids],
    }).encode()


def _anthropic_error_body(msg: str = "model not found") -> bytes:
    return json.dumps({
        "type": "error",
        "error": {"type": "invalid_request_error", "message": msg},
    }).encode()


# ----------------------------------------------------------------------
# _looks_like_anthropic_path
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
# _extract_model_ids
# ----------------------------------------------------------------------





# ----------------------------------------------------------------------
# detect() integration
# ----------------------------------------------------------------------



def test_detect_openai_models_probe_success():
    """/models probe returning a model list → chat_completions."""
    def _fake_get(url, api_key, timeout=6.0, **kwargs):
        assert "key-abc" == api_key
        return 200, json.loads(_openai_models_body("gpt-5.4", "claude-opus-4-6"))

    with patch.object(azure_detect, "_http_get_json", side_effect=_fake_get):
        result = azure_detect.detect(
            "https://my.openai.azure.com/openai/v1", "key-abc",
        )
    assert result.api_mode == "chat_completions"
    assert result.models_probe_ok is True
    assert result.models == ["gpt-5.4", "claude-opus-4-6"]
    assert "/models" in result.reason


# ----------------------------------------------------------------------
# _probe_openai_models URL list (Azure vs v1 api-version)
# ----------------------------------------------------------------------

def test_probe_openai_models_tries_multiple_api_versions():
    """First call (no api-version) fails, api-version fallback succeeds."""
    calls = []

    def _fake_get(url, api_key, timeout=6.0, **kwargs):
        calls.append(url)
        if "api-version" not in url:
            return 404, None
        return 200, json.loads(_openai_models_body("gpt-4.1"))

    with patch.object(azure_detect, "_http_get_json", side_effect=_fake_get):
        ok, models = azure_detect._probe_openai_models(
            "https://my.openai.azure.com/openai/v1", "k",
        )
    assert ok is True
    assert models == ["gpt-4.1"]
    # Should have tried without api-version first, then with at least one
    assert any("api-version" not in u for u in calls)
    assert any("api-version" in u for u in calls)


# ----------------------------------------------------------------------
# _http_get_json error handling
# ----------------------------------------------------------------------



@pytest.fixture
def probe_server():
    bodies = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self.respond(200, bodies["models"])

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.respond(400, bodies["error"])

        def respond(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The bounded reader may close before the oversized body is sent.

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", bodies
        finally:
            server.shutdown()
            worker.join(timeout=5)


def test_http_get_json_bounds_success_body(probe_server):
    """The real credential-safe HTTP path parses small models, not oversized JSON."""
    url, bodies = probe_server
    bodies["models"] = _openai_models_body("test-deployment")
    assert azure_detect._http_get_json(url + "/models", "synthetic-key") == (
        200, json.loads(bodies["models"]),
    )
    bodies["models"] += b" " * azure_detect._AZURE_DETECT_JSON_BODY_MAX_BYTES
    assert azure_detect._http_get_json(url + "/models", "synthetic-key") == (200, None)


def test_probe_anthropic_messages_bounds_error_body(probe_server):
    """Oversized real HTTPError bodies cannot supply the detection signal."""
    url, bodies = probe_server
    bodies["error"] = _anthropic_error_body()
    assert azure_detect._probe_anthropic_messages(url, "synthetic-key") is True
    bodies["error"] += b" " * azure_detect._AZURE_DETECT_ERROR_BODY_MAX_BYTES
    assert azure_detect._probe_anthropic_messages(url, "synthetic-key") is False


# ----------------------------------------------------------------------
# lookup_context_length
# ----------------------------------------------------------------------


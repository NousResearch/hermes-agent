"""E2E: ``/api/show`` probes must send the modern ``model`` request key.

Ollama's ``/api/show`` request body moved from ``{"name": ...}`` to
``{"model": ...}``. Ollama itself still accepts the legacy key, so the bug
is invisible against real Ollama — but OpenAI-compatible servers that
implement the *current* Ollama-compat schema (SGLang's ``--enable-metrics``
Ollama shim, and vLLM's, both of which answer ``/api/tags`` and are
therefore detected as ``server_type == "ollama"``) validate strictly and
reject a ``name``-only body with HTTP 400:

    {"name": "m"}  -> 400  1 validation error: ('body','model') Field required
    {"model": "m"} -> 200

Every capability the ``/api/show`` probes exist to discover (context
length, vision, thinking) was therefore silently lost against those
servers, and the failure surfaced only as recurring 400s in the log.

These tests run against a real ``http.server`` that enforces the modern
schema, so they exercise the true httpx request-body path rather than
asserting on a mock's call args.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading

import pytest

from agent import model_metadata
from agent.model_metadata import (
    query_ollama_num_ctx,
    query_ollama_supports_vision,
)
from hermes_cli.models import ollama_model_supports_thinking


MODEL = "qwen3-27b"

_SHOW_BODY = {
    "parameters": "num_ctx 262144",
    "model_info": {f"{MODEL}.context_length": 262144},
    "capabilities": ["completion", "vision", "thinking"],
}


class _StrictOllamaHandler(http.server.BaseHTTPRequestHandler):
    """Ollama-compatible server that requires the modern ``model`` key."""

    # Set by the fixture so a test can inspect what was actually sent.
    received: list = []

    def log_message(self, *args):  # noqa: D102 - silence test output
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        # /api/tags makes detect_local_server_type() classify us as "ollama",
        # which is precisely how a strict non-Ollama server ends up on the
        # /api/show code path in the first place.
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": MODEL, "model": MODEL}]})
            return
        self._send(404, {"detail": "Not Found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/api/show":
            self._send(404, {"detail": "Not Found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            payload = {}
        type(self).received.append(payload)
        if not payload.get("model"):
            self._send(
                400,
                {
                    "object": "error",
                    "message": (
                        "1 validation error:\n  {'type': 'missing', "
                        "'loc': ('body', 'model'), 'msg': 'Field required'}"
                    ),
                },
            )
            return
        self._send(200, _SHOW_BODY)


@pytest.fixture
def strict_ollama(tmp_path, monkeypatch):
    """A live strict-schema server, with all probe caches isolated."""
    # Probe results are memoized in-process AND on disk under HERMES_HOME;
    # point both somewhere disposable so a previous run can't mask the fix.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(model_metadata, "_endpoint_probe_path_cache", {}, raising=False)
    monkeypatch.setattr(model_metadata, "_endpoint_blackhole_cache", {}, raising=False)
    for name in (
        "_query_ollama_api_show",
        "_query_local_context_length",
        "query_ollama_num_ctx",
        "query_ollama_supports_vision",
    ):
        fn = getattr(model_metadata, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()

    _StrictOllamaHandler.received = []
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("127.0.0.1", 0), _StrictOllamaHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_num_ctx_probe_survives_strict_schema(strict_ollama):
    """Context length is recovered, not lost to a 400."""
    assert query_ollama_num_ctx(MODEL, strict_ollama) == 262144


def test_vision_probe_survives_strict_schema(strict_ollama):
    """Vision capability is recovered, not lost to a 400."""
    assert query_ollama_supports_vision(MODEL, strict_ollama) is True


def test_thinking_probe_survives_strict_schema(strict_ollama):
    """Thinking capability is recovered, not lost to a 400."""
    assert ollama_model_supports_thinking(MODEL, strict_ollama) is True


def test_show_requests_carry_the_model_key(strict_ollama):
    """Every /api/show body sends ``model``; ``name`` stays for old servers."""
    query_ollama_num_ctx(MODEL, strict_ollama)
    assert _StrictOllamaHandler.received, "no /api/show request was made"
    for payload in _StrictOllamaHandler.received:
        assert payload.get("model") == MODEL
        # Ollama releases predating the rename only understand ``name``;
        # sending both keeps those working.
        assert payload.get("name") == MODEL

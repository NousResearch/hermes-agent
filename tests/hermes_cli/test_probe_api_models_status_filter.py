"""probe_api_models() must drop catalog entries the endpoint itself flags as
unavailable (status: Shutdown/Retiring/etc.), while keeping standard entries
that carry no status field. Regression test for the /model picker flooding with
dead models on OpenAI-compatible catalogs that populate ``status``.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from hermes_cli.models import probe_api_models


class _MixedStatusModelsHandler(BaseHTTPRequestHandler):
    # Mixed catalog: a few servable, several explicitly unavailable.
    CATALOG: list[dict[str, Any]] = [
        {"id": "glm-5-2-active"},  # no status field -> keep
        {"id": "ark-code-latest", "status": "Running"},  # non-standard active -> keep
        {"id": "dead-shutdown", "status": "Shutdown"},  # drop
        {"id": "soon-retiring", "status": "Retiring"},  # drop
        {"id": "old-deprecated", "status": "deprecated"},  # drop (case-insensitive)
        {"id": "off-disabled", "status": "Disabled"},  # drop
        {"id": "vision-only", "status": "ACTIVE"},  # active variant -> keep
    ]

    def do_GET(self):
        body = json.dumps({"data": type(self).CATALOG}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


def _serve(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_probe_api_models_drops_unavailable_status_entries():
    server = _serve(_MixedStatusModelsHandler)
    try:
        result = probe_api_models(
            api_key="provider-key",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            timeout=3,
        )
    finally:
        server.shutdown()

    models = result["models"]
    # Servable entries survive regardless of whether status is absent, an active
    # variant, or a non-standard active value.
    assert "glm-5-2-active" in models
    assert "ark-code-latest" in models
    assert "vision-only" in models
    # Every entry the endpoint flagged unavailable is filtered out.
    assert "dead-shutdown" not in models
    assert "soon-retiring" not in models
    assert "old-deprecated" not in models
    assert "off-disabled" not in models
    # Ordering of the kept entries is preserved.
    assert models == ["glm-5-2-active", "ark-code-latest", "vision-only"]


def test_probe_api_models_keeps_all_entries_without_status():
    """A standard OpenAI-style response (no status field anywhere) is unchanged."""

    class _NoStatusHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(
                {"data": [{"id": "gpt-x"}, {"id": "gpt-y"}, {"id": "gpt-z"}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            pass

    server = _serve(_NoStatusHandler)
    try:
        result = probe_api_models(
            api_key="provider-key",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            timeout=3,
        )
    finally:
        server.shutdown()

    assert result["models"] == ["gpt-x", "gpt-y", "gpt-z"]

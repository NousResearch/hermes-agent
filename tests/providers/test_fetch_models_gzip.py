"""Regression: fetch_models must decode a gzip-encoded catalog response.

A provider whose ``default_headers`` request ``Accept-Encoding: gzip`` (Kimi /
Moonshot does, to avoid brotli on SSE) gets a gzip body back. The catalog
reader used to hand those raw bytes straight to ``json.loads``, raise
``UnicodeDecodeError`` and swallow the exception in ``logger.debug`` — so the
live catalog silently degraded to the static curated list and the model picker
never showed the provider's current models.
"""

import gzip
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from providers.base import ProviderProfile


class _GzipModelHandler(BaseHTTPRequestHandler):
    """Serves /models as gzip-compressed JSON."""

    models = [{"id": "gzip-model-1"}, {"id": "gzip-model-2"}]

    def do_GET(self):
        if self.path.rstrip("/") == "/models":
            body = gzip.compress(json.dumps({"data": self.models}).encode())
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Encoding", "gzip")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress noise


def _start_server():
    """Start a local HTTP server returning a gzip-compressed model list."""
    server = HTTPServer(("127.0.0.1", 0), _GzipModelHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


class TestFetchModelsGzipResponse:
    """A gzip-encoded /models body must be decoded before JSON parsing."""

    def test_gzip_response_is_decoded(self):
        server, port = _start_server()
        try:
            profile = ProviderProfile(
                name="test",
                base_url=f"http://127.0.0.1:{port}",
                default_headers={"Accept-Encoding": "gzip"},
            )
            result = profile.fetch_models(
                api_key="test-key",
                base_url=f"http://127.0.0.1:{port}",
            )
            assert result == ["gzip-model-1", "gzip-model-2"]
        finally:
            server.shutdown()

    def test_gzip_response_without_explicit_accept_encoding_still_decoded(self):
        """The server may compress regardless of what the client asked for."""
        server, port = _start_server()
        try:
            profile = ProviderProfile(
                name="test",
                base_url=f"http://127.0.0.1:{port}",
            )
            result = profile.fetch_models(
                api_key="test-key",
                base_url=f"http://127.0.0.1:{port}",
            )
            assert result == ["gzip-model-1", "gzip-model-2"]
        finally:
            server.shutdown()

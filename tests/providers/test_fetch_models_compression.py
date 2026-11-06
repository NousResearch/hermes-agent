"""Model discovery must decode gzip negotiated by provider default headers."""

import gzip
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from providers.base import ProviderProfile


@pytest.fixture
def model_endpoint():
    class Handler(BaseHTTPRequestHandler):
        body = b""
        encoding = None
        received_encoding = None

        def do_GET(self):
            type(self).received_encoding = self.headers.get("Accept-Encoding")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            if self.encoding:
                self.send_header("Content-Encoding", self.encoding)
            self.send_header("Content-Length", str(len(self.body)))
            self.end_headers()
            self.wfile.write(self.body)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", Handler
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("encoding", ["identity", "gzip", "gzip-without-header"])
def test_fetch_models_decodes_response_before_filtering(model_endpoint, encoding):
    url, handler = model_endpoint
    payload = json.dumps({"data": [
        {"id": "chat-model", "modalities": {"output": ["text"]}},
        {"id": "image-model", "modalities": {"output": ["image"]}},
    ]}).encode()
    handler.body = gzip.compress(payload) if encoding.startswith("gzip") else payload
    handler.encoding = "gzip" if encoding == "gzip" else None
    profile = ProviderProfile(
        name="compressed-catalog", base_url=url,
        default_headers={"Accept-Encoding": "gzip"},
    )

    assert profile.fetch_models(api_key="test-key") == ["chat-model"]
    assert handler.received_encoding == "gzip"


def test_fetch_models_corrupt_gzip_keeps_fallback_contract(model_endpoint):
    url, handler = model_endpoint
    handler.body = b"\x1f\x8btruncated"
    handler.encoding = "gzip"
    profile = ProviderProfile(name="broken-catalog", base_url=url)

    assert profile.fetch_models(api_key="test-key") is None

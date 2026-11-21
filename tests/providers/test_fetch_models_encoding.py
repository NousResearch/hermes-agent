"""fetch_models() must decode compressed /models responses (issue #156423).

The Kimi Coding profile ships ``Accept-Encoding: gzip`` in ``default_headers``,
so ``api.kimi.com/coding/v1/models`` answers with a gzipped body. Reading it
straight into ``json.loads(...decode())`` raised on the gzip magic bytes; the
exception is swallowed by design (the caller falls back to the curated
catalog), so the failure was invisible and newly released models simply never
showed up in the picker.
"""

import gzip
import json
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from providers.base import ProviderProfile


class _EncodedModelHandler(BaseHTTPRequestHandler):
    """Serves /models with a body compressed using the given encoding."""

    encoding = "gzip"
    omit_encoding = False
    models = [{"id": "k3.1-pro"}, {"id": "k3-256k"}]

    def do_GET(self):
        if self.path.rstrip("/") != "/models":
            self.send_response(404)
            self.end_headers()
            return
        raw = json.dumps({"data": self.models}).encode()
        if self.encoding in ("gzip", "x-gzip"):
            body = gzip.compress(raw)
        elif self.encoding == "deflate":
            body = zlib.compress(raw)
        else:
            body = raw
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if not self.omit_encoding:
            self.send_header("Content-Encoding", self.encoding)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress noise


def _start_server(encoding, models=None, *, omit_encoding=False):
    handler = type(
        "_Handler",
        (_EncodedModelHandler,),
        {
            "encoding": encoding,
            "models": models or _EncodedModelHandler.models,
            "omit_encoding": omit_encoding,
        },
    )
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def _fetch(encoding, models=None, *, omit_encoding=False):
    server, port = _start_server(encoding, models, omit_encoding=omit_encoding)
    try:
        profile = ProviderProfile(name="test", base_url=f"http://127.0.0.1:{port}")
        return profile.fetch_models(base_url=f"http://127.0.0.1:{port}")
    finally:
        server.shutdown()
        server.server_close()


class TestFetchModelsCompressedResponse:
    @pytest.mark.parametrize("encoding", ["gzip", "x-gzip"])
    def test_gzip_body_is_decompressed(self, encoding):
        """A gzipped catalog must be parsed, not silently dropped to None."""
        assert _fetch(encoding) == ["k3.1-pro", "k3-256k"]

    def test_deflate_body_is_decompressed(self):
        assert _fetch("deflate") == ["k3.1-pro", "k3-256k"]

    def test_identity_body_still_parses(self):
        """Uncompressed responses keep working (no Content-Encoding header path)."""
        server, port = _start_server("identity")
        try:
            profile = ProviderProfile(name="test", base_url=f"http://127.0.0.1:{port}")
            # No Content-Encoding at all: the handler still sets "identity", which must be a no-op.
            assert profile.fetch_models(base_url=f"http://127.0.0.1:{port}") == ["k3.1-pro", "k3-256k"]
        finally:
            server.shutdown()
            server.server_close()

    def test_gzip_body_without_header_is_detected(self):
        """An intermediary may forward gzip bytes with no Content-Encoding.

        The gzip magic number is the only signal left, so decoding must fall back to it
        instead of handing the compressed bytes to UTF-8.
        """
        assert _fetch("gzip", omit_encoding=True) == ["k3.1-pro", "k3-256k"]

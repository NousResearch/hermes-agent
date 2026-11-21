"""Regression: catalog JSON readers must decode a compressed response body.

``urllib`` never transparently decompresses a response, so a provider whose
profile asks for gzip (Kimi/Moonshot does, to sidestep httpx's brotli SSE
decode bug) hands back compressed bytes. Both catalog readers used to hand
those raw bytes straight to ``json.loads``:

* ``providers.base.ProviderProfile.fetch_models`` — swallowed the
  ``UnicodeDecodeError`` in ``logger.debug`` and returned ``None``, so the
  model picker silently fell back to the provider's stale static list.
* ``hermes_cli.models._get_json`` — only inflated a body when the *request*
  asked for gzip AND the *response* advertised it, so a server that
  compressed without advertising the encoding still failed to parse.

Both now go through ``hermes_cli.urllib_security.decode_response_body``.
"""

from __future__ import annotations

import gzip
import json
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from hermes_cli.models import _get_json
from hermes_cli.urllib_security import decode_response_body


class _FakeResponse:
    """Minimal stand-in for the response object the decoder inspects."""

    def __init__(self, body: bytes, content_encoding: str = "") -> None:
        self._body = body
        self.headers = {"Content-Encoding": content_encoding} if content_encoding else {}

    def read(self) -> bytes:
        return self._body


class _CompressedHandler(BaseHTTPRequestHandler):
    """Serves /models as a compressed JSON body."""

    models = [{"id": "decoded-model-1"}, {"id": "decoded-model-2"}]
    encoding = "gzip"
    advertise_encoding = True

    def do_GET(self):
        if self.path.rstrip("/") != "/models":
            self.send_response(404)
            self.end_headers()
            return
        payload = json.dumps({"data": self.models}).encode()
        if self.encoding == "gzip":
            body = gzip.compress(payload)
        elif self.encoding == "deflate":
            body = zlib.compress(payload)
        else:
            body = payload
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        if self.advertise_encoding and self.encoding:
            self.send_header("Content-Encoding", self.encoding)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress noise


def _start_server(handler_class=_CompressedHandler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


class TestDecodeResponseBody:
    """The shared decoder handles the encodings a catalog endpoint may use."""

    def test_plain_body_is_returned_unchanged(self):
        body = b'{"data": []}'
        assert decode_response_body(_FakeResponse(body), body) == body

    def test_gzip_body_with_content_encoding_is_decoded(self):
        body = gzip.compress(b'{"data": []}')
        assert decode_response_body(_FakeResponse(body, "gzip"), body) == b'{"data": []}'

    def test_gzip_body_without_content_encoding_is_decoded_by_magic_bytes(self):
        """A server may compress the body without advertising the encoding."""
        body = gzip.compress(b'{"data": []}')
        assert decode_response_body(_FakeResponse(body), body) == b'{"data": []}'

    def test_deflate_body_is_decoded(self):
        body = zlib.compress(b'{"data": []}')
        assert decode_response_body(_FakeResponse(body, "deflate"), body) == b'{"data": []}'

    def test_brotli_is_left_alone(self):
        """brotli stays undecoded: httpx's brotlicffi backend has a streaming
        decode bug on Moonshot's brotli SSE responses (#28043, #48428, #59556)."""
        body = b"\x0b\x02\x80raw-brotli-ish"
        assert decode_response_body(_FakeResponse(body, "br"), body) == body


class TestGetJsonDecodesCompressedBody:
    """``hermes_cli.models._get_json`` must parse a compressed catalog body."""

    def test_gzip_response_is_parsed(self):
        server, port = _start_server()
        try:
            payload = _get_json(f"http://127.0.0.1:{port}/models", timeout=10)
        finally:
            server.shutdown()
        assert [row["id"] for row in payload["data"]] == ["decoded-model-1", "decoded-model-2"]

    def test_gzip_response_without_advertised_encoding_is_parsed(self):
        """The request asked for plain JSON; the server compressed anyway."""
        server, port = _start_server()
        try:
            _CompressedHandler.advertise_encoding = False
            payload = _get_json(f"http://127.0.0.1:{port}/models", timeout=10)
        finally:
            server.shutdown()
            _CompressedHandler.advertise_encoding = True
        assert [row["id"] for row in payload["data"]] == ["decoded-model-1", "decoded-model-2"]


@pytest.mark.parametrize(
    "encoding",
    ["gzip", "deflate"],
)
def test_get_json_handles_supported_encodings(encoding):
    server, port = _start_server()
    try:
        _CompressedHandler.encoding = encoding
        payload = _get_json(f"http://127.0.0.1:{port}/models", timeout=10)
    finally:
        server.shutdown()
        _CompressedHandler.encoding = "gzip"
    assert [row["id"] for row in payload["data"]] == ["decoded-model-1", "decoded-model-2"]

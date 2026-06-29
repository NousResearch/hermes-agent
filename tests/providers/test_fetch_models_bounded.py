"""Tests for bounded model-catalog reads in fetch_models (issue #54735).

A malicious / compromised / misconfigured model endpoint must not be able to
exhaust memory during model discovery. ``ProviderProfile.fetch_models`` should
read at most a bounded body and fail closed (return ``None``, falling back to
the static model list) on oversize responses.

These assert the behavior contract (oversize -> None, normal -> list), not a
snapshot of any particular model catalog.
"""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from providers.base import ProviderProfile, _read_json_capped


class _OversizedContentLengthHandler(BaseHTTPRequestHandler):
    """Declares a huge Content-Length without streaming the bytes."""

    cap = 0

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        # Lie about the size: claim far more than the cap allows.
        self.send_header("Content-Length", str(self.cap * 4))
        self.end_headers()
        # Body is irrelevant — the Content-Length guard should reject first.
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


class _OversizedStreamedBodyHandler(BaseHTTPRequestHandler):
    """Sends a body larger than the cap without an honest Content-Length."""

    cap = 0

    def do_GET(self):
        # A valid-looking JSON object padded past the cap with whitespace
        # inside the string value, so the body genuinely exceeds the limit.
        filler = b" " * (self.cap + 1024)
        body = b'{"data": [{"id": "m1"}], "_pad": "' + filler + b'"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        # Deliberately omit Content-Length to force the streamed-read guard.
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _SmallBodyHandler(BaseHTTPRequestHandler):
    """A normal, well-behaved /models response."""

    def do_GET(self):
        body = json.dumps({"data": [{"id": "ok-model-1"}, {"id": "ok-model-2"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _serve(handler_cls):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    Thread(target=server.serve_forever, daemon=True).start()
    return server, port


class TestFetchModelsBounded:
    def test_oversized_content_length_returns_none(self):
        """A response declaring Content-Length > cap fails closed."""
        _OversizedContentLengthHandler.cap = 4096
        server, port = _serve(_OversizedContentLengthHandler)
        try:
            profile = ProviderProfile(name="t", base_url=f"http://127.0.0.1:{port}")
            # Shrink the cap via the helper default by monkeypatching the module
            # constant is unnecessary — the handler scales to its own `cap`.
            import providers.base as pb
            orig = pb._MAX_MODELS_RESPONSE_BYTES
            pb._MAX_MODELS_RESPONSE_BYTES = 4096
            try:
                assert profile.fetch_models(api_key="k") is None
            finally:
                pb._MAX_MODELS_RESPONSE_BYTES = orig
        finally:
            server.shutdown()

    def test_oversized_streamed_body_returns_none(self):
        """A body that runs past the cap with no honest Content-Length fails closed."""
        _OversizedStreamedBodyHandler.cap = 4096
        server, port = _serve(_OversizedStreamedBodyHandler)
        try:
            profile = ProviderProfile(name="t", base_url=f"http://127.0.0.1:{port}")
            import providers.base as pb
            orig = pb._MAX_MODELS_RESPONSE_BYTES
            pb._MAX_MODELS_RESPONSE_BYTES = 4096
            try:
                assert profile.fetch_models(api_key="k") is None
            finally:
                pb._MAX_MODELS_RESPONSE_BYTES = orig
        finally:
            server.shutdown()

    def test_normal_small_body_still_works(self):
        """A normal small catalog response is parsed correctly (no false reject)."""
        server, port = _serve(_SmallBodyHandler)
        try:
            profile = ProviderProfile(name="t", base_url=f"http://127.0.0.1:{port}")
            assert profile.fetch_models(api_key="k") == ["ok-model-1", "ok-model-2"]
        finally:
            server.shutdown()


class _FakeResp:
    """Minimal response stand-in to assert the read is bounded, not unbounded."""

    def __init__(self, body: bytes, declared_len=None):
        self._body = body
        self.read_arg = None

        class _H:
            def __init__(self, v):
                self._v = v

            def get(self, _key):
                return self._v

        self.headers = _H(str(declared_len) if declared_len is not None else None)

    def read(self, n=None):
        self.read_arg = n
        if n is None:
            return self._body
        return self._body[:n]


class TestReadJsonCappedHelper:
    def test_passes_bounded_size_to_read(self):
        """_read_json_capped must call read(cap + 1), never an unbounded read()."""
        body = json.dumps({"data": [{"id": "x"}]}).encode()
        resp = _FakeResp(body)
        _read_json_capped(resp, max_bytes=1024)
        assert resp.read_arg == 1025  # cap + 1, i.e. bounded

    def test_rejects_declared_oversize(self):
        """A Content-Length over the cap is rejected before reading the body."""
        resp = _FakeResp(b"{}", declared_len=10_000)
        try:
            _read_json_capped(resp, max_bytes=1024)
            assert False, "expected ValueError for oversize Content-Length"
        except ValueError:
            pass

    def test_rejects_streamed_oversize(self):
        """A body exceeding the cap is rejected even without Content-Length."""
        resp = _FakeResp(b"x" * 5000)
        try:
            _read_json_capped(resp, max_bytes=1024)
            assert False, "expected ValueError for oversize streamed body"
        except ValueError:
            pass

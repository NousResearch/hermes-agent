"""Auth-boundary shard for the Anthropic adapter.

Credential-isolation regressions (api-key construction vs the ambient
ANTHROPIC_AUTH_TOKEN bearer, fail-closed fallback when the SDK Omit
sentinel is unavailable) live here rather than in the main adapter
suite to respect the 2,000-line per-file ceiling. Both suites run
together in CI; coverage is unchanged.
"""

import pytest


def _final_request_options(anthropic_sdk):
    from anthropic._models import FinalRequestOptions

    return FinalRequestOptions(method="post", url="/v1/messages", json_data={})


def _start_header_capturing_server():
    """Local HTTP server that records the headers of each POST and returns a minimal Messages
    response. Returns ``(server, captured)``; the caller derives the base_url from
    ``server.server_port`` and must call ``server.shutdown()``. Used by the fail-closed fallback
    tests, where the strip happens at the httpx transport layer (not in ``_build_headers``) and so
    can only be observed on a real request."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    captured = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured["headers"] = {k.lower(): v for k, v in self.headers.items()}
            self.rfile.read(int(self.headers.get("content-length", 0)))
            body = _json.dumps({
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "model": "test",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, captured


class TestApiKeyConstructionClearsEnvBearerToken:
    """Api-key-style clients must not inherit ANTHROPIC_AUTH_TOKEN from the environment.

    The Anthropic SDK fills an unset ``auth_token`` from ANTHROPIC_AUTH_TOKEN and then sends
    ``Authorization: Bearer ***`` alongside ``x-api-key`` on every request, shipping a
    foreign shell credential to third-party Anthropic-compatible endpoints.
    """

    def test_api_key_style_client_drops_env_derived_auth_token(self, monkeypatch):
        anthropic_sdk = pytest.importorskip("anthropic")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sentinel-env-token-DO-NOT-SEND")
        from agent.anthropic_adapter import _new_sdk_client

        client = _new_sdk_client(
            anthropic_sdk,
            {"api_key": "provider-key", "base_url": "http://127.0.0.1:1"},
            {},
        )
        assert client.api_key == "provider-key"
        # The guard is a copy-safe Omit() default header: the SDK keeps the env-derived
        # auth_token attribute, but the request headers (what reaches the wire) never
        # contain Authorization — on the original client and on any with_options() copy.
        for wire_client in (client, client.with_options(timeout=30)):
            headers = dict(
                wire_client._build_headers(_final_request_options(anthropic_sdk))
            )
            assert headers.get("x-api-key") == "provider-key"
            assert "authorization" not in headers

    def test_bearer_style_client_keeps_its_auth_token(self, monkeypatch):
        anthropic_sdk = pytest.importorskip("anthropic")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sentinel-env-key-DO-NOT-SEND")
        from agent.anthropic_adapter import _new_sdk_client

        client = _new_sdk_client(
            anthropic_sdk,
            {"auth_token": "bearer-secret", "base_url": "http://127.0.0.1:1"},
            {},
        )
        assert client.auth_token == "bearer-secret"
        assert client.api_key is None
        assert client.auth_headers == {"Authorization": "Bearer bearer-secret"}

    def test_third_party_request_carries_no_foreign_bearer(self, monkeypatch):
        """End-to-end over a local header-capturing server: x-api-key intact, sentinel absent."""
        anthropic_sdk = pytest.importorskip("anthropic")
        http_server = pytest.importorskip("http.server")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sentinel-env-token-DO-NOT-SEND")

        import json as _json
        import threading

        captured = {}

        class _Handler(http_server.BaseHTTPRequestHandler):
            def do_POST(self):
                captured["headers"] = {k.lower(): v for k, v in self.headers.items()}
                self.rfile.read(int(self.headers.get("content-length", 0)))
                body = _json.dumps({
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "model": "test",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http_server.HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = build_anthropic_client(
                "third-party-provider-key",
                base_url=f"http://127.0.0.1:{server.server_port}",
            )
            client.messages.create(
                model="test-model",
                max_tokens=8,
                messages=[{"role": "user", "content": "hi"}],
            )
        finally:
            server.shutdown()

        headers = captured["headers"]
        assert headers.get("x-api-key") == "third-party-provider-key"
        assert "sentinel-env-token-DO-NOT-SEND" not in headers.get("authorization", "")

    def test_with_options_copy_carries_no_foreign_bearer(self, monkeypatch):
        """``with_options()`` re-runs the constructor with ``auth_token=None``, so an attribute
        clear would re-leak ANTHROPIC_AUTH_TOKEN on the copy; the Omit() default header
        propagates through copies and keeps the sentinel off the wire."""
        anthropic_sdk = pytest.importorskip("anthropic")
        http_server = pytest.importorskip("http.server")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sentinel-env-token-DO-NOT-SEND")

        import json as _json
        import threading

        captured = {}

        class _Handler(http_server.BaseHTTPRequestHandler):
            def do_POST(self):
                captured["headers"] = {k.lower(): v for k, v in self.headers.items()}
                self.rfile.read(int(self.headers.get("content-length", 0)))
                body = _json.dumps({
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "model": "test",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http_server.HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = build_anthropic_client(
                "third-party-provider-key",
                base_url=f"http://127.0.0.1:{server.server_port}",
            )
            client.with_options(timeout=30).messages.create(
                model="test-model",
                max_tokens=8,
                messages=[{"role": "user", "content": "hi"}],
            )
        finally:
            server.shutdown()

        headers = captured["headers"]
        assert headers.get("x-api-key") == "third-party-provider-key"
        assert "sentinel-env-token-DO-NOT-SEND" not in headers.get("authorization", "")

    def test_api_key_style_fails_closed_when_omit_unavailable(self, monkeypatch):
        """When ``anthropic._types.Omit`` cannot be imported, the copy-safe Omit default header
        is unavailable — but the api-key path must still not leak ANTHROPIC_AUTH_TOKEN. It falls
        back to a request hook that strips Authorization on the wire (the strip is at the httpx
        transport layer, hence the real request), on the original client AND on a with_options()
        copy, while x-api-key and request success are preserved."""
        anthropic_sdk = pytest.importorskip("anthropic")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sentinel-env-token-DO-NOT-SEND")
        monkeypatch.setattr("agent.anthropic_adapter._sdk_omit_sentinel", lambda sdk: None)

        server, captured = _start_header_capturing_server()
        try:
            client = build_anthropic_client(
                "third-party-provider-key",
                base_url=f"http://127.0.0.1:{server.server_port}",
            )
            for wire_client in (client, client.with_options(timeout=30)):
                captured.clear()
                wire_client.messages.create(
                    model="test-model",
                    max_tokens=8,
                    messages=[{"role": "user", "content": "hi"}],
                )
                headers = captured["headers"]
                assert headers.get("x-api-key") == "third-party-provider-key"
                assert "sentinel-env-token-DO-NOT-SEND" not in headers.get("authorization", "")
        finally:
            server.shutdown()

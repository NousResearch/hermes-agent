"""Auth-boundary shard for the Anthropic adapter.

An api-key-style client must not inherit ANTHROPIC_AUTH_TOKEN from the environment: the SDK
fills an unset ``auth_token`` from that variable and then sends ``Authorization: Bearer ***``
alongside ``x-api-key`` on every request, shipping a foreign shell credential to third-party
Anthropic-compatible endpoints (#105774). Kept out of ``test_anthropic_adapter.py`` for the
per-file line ceiling.
"""

import pytest

from agent.anthropic_adapter import build_anthropic_client

SENTINEL = "sentinel-env-token-DO-NOT-SEND"


def test_api_key_client_and_its_copies_never_carry_the_env_bearer(monkeypatch):
    """The guard is a copy-safe Omit() default header: ``with_options()`` re-runs the constructor
    and re-reads ANTHROPIC_AUTH_TOKEN, so an attribute clear alone would re-leak on the copy."""
    anthropic_sdk = pytest.importorskip("anthropic")
    from anthropic._models import FinalRequestOptions

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", SENTINEL)
    from agent.anthropic_adapter import _new_sdk_client

    client = _new_sdk_client(anthropic_sdk, {"api_key": "provider-key", "base_url": "http://127.0.0.1:1"}, {})
    for wire_client in (client, client.with_options(timeout=30)):
        headers = dict(wire_client._build_headers(
            FinalRequestOptions(method="post", url="/v1/messages", json_data={})))
        assert headers.get("x-api-key") == "provider-key"
        assert "authorization" not in headers

    # Mirror case: bearer-style construction must not ship an env ANTHROPIC_API_KEY, on copies either.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sentinel-env-key-DO-NOT-SEND")
    bearer = _new_sdk_client(anthropic_sdk, {"auth_token": "bearer-secret", "base_url": "http://127.0.0.1:1"}, {})
    for wire_client in (bearer, bearer.with_options(timeout=30)):
        headers = dict(wire_client._build_headers(
            FinalRequestOptions(method="post", url="/v1/messages", json_data={})))
        assert headers.get("authorization") == "Bearer bearer-secret"
        assert "x-api-key" not in headers


def test_third_party_request_on_the_wire_carries_no_foreign_bearer(monkeypatch):
    """End-to-end through ``build_anthropic_client`` against a local header-capturing server."""
    pytest.importorskip("anthropic")
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", SENTINEL)
    captured = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            captured["headers"] = {k.lower(): v for k, v in self.headers.items()}
            self.rfile.read(int(self.headers.get("content-length", 0)))
            body = json.dumps({
                "id": "msg_test", "type": "message", "role": "assistant",
                "content": [{"type": "text", "text": "ok"}], "model": "test",
                "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1},
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
    try:
        client = build_anthropic_client("third-party-provider-key", base_url=f"http://127.0.0.1:{server.server_port}")
        client.with_options(timeout=30).messages.create(
            model="test-model", max_tokens=8, messages=[{"role": "user", "content": "hi"}])
    finally:
        server.shutdown()

    assert captured["headers"].get("x-api-key") == "third-party-provider-key"
    assert "authorization" not in captured["headers"]

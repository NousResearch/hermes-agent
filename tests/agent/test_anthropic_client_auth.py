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


def wire_headers(client) -> dict:
    """Headers the SDK would put on a /v1/messages POST (the Omit() default is resolved here)."""
    from anthropic._models import FinalRequestOptions

    return dict(client._build_headers(FinalRequestOptions(method="post", url="/v1/messages", json_data={})))



    # The bearer mirror (no env x-api-key beside a portal JWT) is owned by
    # tests/agent/test_nous_portal_anthropic_wire.py::TestClientShape.


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


@pytest.mark.parametrize(("family", "oauth"), [("api03", False), ("usr01", False), ("svc01", False), ("admin01", False), ("oat01", True)])
def test_console_key_families_carry_x_api_key_without_claude_code_identity(monkeypatch, family, oauth):
    """#133856: Console key families must not ride Bearer + the Claude Code beta/user-agent (that request
    shape bills the subscription bucket: 400 "credit balance too low"); OAuth setup tokens still do."""
    pytest.importorskip("anthropic")
    monkeypatch.setattr("agent.anthropic_adapter._claude_code_version_cache", "0.0.0")  # skip the `claude --version` probe
    key = "sk-" + "ant-" + family + "-" + "X" * 24
    headers = {k.lower(): v for k, v in wire_headers(build_anthropic_client(key, "https://api.anthropic.com")).items()}
    assert (headers.get("authorization") == "Bearer " + key) is oauth
    assert (headers.get("x-api-key") == key) is (not oauth)
    assert ("claude-code-20250219" in headers.get("anthropic-beta", "")) is oauth
    assert ("claude-code/" in headers.get("user-agent", "")) is oauth


@pytest.mark.parametrize(("base_url", "family", "expect"), [
    ("https://api.anthropic.com", "usr01", "x-api-key"), ("https://api.anthropic.com", "api03", "x-api-key"),
    ("https://api.anthropic.com", "oat01", "bearer"), ("https://gateway.invalid/anthropic", "usr01", "bearer"),
    (None, "svc01", "x-api-key"),  # an empty base_url is the native default
])
def test_key_cmd_token_lands_in_the_header_its_family_needs(monkeypatch, base_url, family, expect):
    """A key_cmd (callable) credential takes the per-request hook path: on the native host a Console key it
    prints must go out as x-api-key like the static path; OAuth tokens and non-native gateways keep Bearer."""
    pytest.importorskip("anthropic")
    import httpx

    monkeypatch.setattr("agent.anthropic_adapter._claude_code_version_cache", "0.0.0")
    key = "sk-" + "ant-" + family + "-" + "X" * 24
    client = build_anthropic_client(lambda: key, base_url)
    request = httpx.Request("POST", (base_url or "https://api.anthropic.com") + "/v1/messages", headers={"Authorization": "Bearer entra-id-bearer-via-http-hook"})
    client._client.event_hooks["request"][0](request)
    sent = {"x-api-key": request.headers.get("x-api-key"), "bearer": request.headers.get("authorization")}
    assert sent == ({"x-api-key": key, "bearer": None} if expect == "x-api-key" else {"x-api-key": None, "bearer": "Bearer " + key})

"""Auth-boundary shard for the Anthropic adapter.

An api-key-style client must not inherit ANTHROPIC_AUTH_TOKEN from the environment: the SDK
fills an unset ``auth_token`` from that variable and then sends ``Authorization: Bearer ***``
alongside ``x-api-key`` on every request, shipping a foreign shell credential to third-party
Anthropic-compatible endpoints (#105774). Kept out of ``test_anthropic_adapter.py`` for the
per-file line ceiling.
"""

import pytest

from agent.anthropic_adapter import build_anthropic_client
from agent.anthropic_credentials import _is_oauth_token

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


@pytest.mark.parametrize(("family", "source", "expected"), [
    ("api03", "env", False), ("usr01", "env", False), ("svc01", "env", False), ("admin01", "env", False),
    # A Console family Anthropic ships after _API_KEY_FAMILIES: the declaration decides, not the
    # prefix list, or the next family repeats #137334 (Claude Code identity -> subscription billing
    # -> 400 "credit balance is too low").
    ("team01", "env", False),
    # Same family declared by its own source object — a credential_pool row's auth_type or a static
    # config key passes declared_api_key and must not be re-guessed from shape either.
    ("team01", "param", False), ("team01", "param-oauth", True),
    # Undeclared values keep the historical OAuth default (#137140: OAuth prefixes shipped later
    # must keep working), and positively OAuth shapes stay OAuth wherever they are stored.
    ("team01", "none", True), ("sid01", "none", True), ("oat01", "env", True), ("oat01", "param", True),
])
def test_source_declaration_decides_the_ambiguous_family_band(monkeypatch, family, source, expected):
    """An ``sk-ant-*`` family whose shape is ambiguous is classified by its source's declaration."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    key = "sk-" + "ant-" + family + "-" + "X" * 24
    declared_api_key = None
    if source == "env":
        monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    elif source == "param":
        declared_api_key = True
    elif source == "param-oauth":
        declared_api_key = False
    else:
        monkeypatch.setenv("ANTHROPIC_TOKEN", key)
    assert _is_oauth_token(key, declared_api_key=declared_api_key) is expected


def test_declared_env_api_key_never_rides_the_claude_code_identity(monkeypatch):
    """Wire proof for #137334's class: a declared API key of an unknown family must not go out with the
    Claude Code identity (that request shape is billed against the subscription bucket, not the credit)."""
    pytest.importorskip("anthropic")
    monkeypatch.setattr("agent.anthropic_adapter._claude_code_version_cache", "0.0.0")
    key = "sk-" + "ant-" + "team01" + "-" + "X" * 24
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    headers = {k.lower(): v for k, v in wire_headers(build_anthropic_client(key, "https://api.anthropic.com")).items()}
    assert headers.get("x-api-key") == key
    assert headers.get("authorization") is None
    assert "claude-code-20250219" not in headers.get("anthropic-beta", "")
    assert "claude-code/" not in headers.get("user-agent", "")
    # The same declaration arriving with the credential (pool auth_type / static config key) instead
    # of the env var: identical wire shape.
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    headers = {k.lower(): v for k, v in wire_headers(
        build_anthropic_client(key, "https://api.anthropic.com", declared_api_key=True)).items()}
    assert headers.get("x-api-key") == key and headers.get("authorization") is None
    assert "claude-code-20250219" not in headers.get("anthropic-beta", "")
    # Control: no declaration at all keeps the OAuth default.
    headers = {k.lower(): v for k, v in wire_headers(build_anthropic_client(key, "https://api.anthropic.com")).items()}
    assert headers.get("authorization") == "Bearer " + key
    assert "claude-code-20250219" in headers.get("anthropic-beta", "")

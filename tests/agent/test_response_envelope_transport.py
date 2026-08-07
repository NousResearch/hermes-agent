"""Unit tests for OpenAI-compat response-envelope unwrap transport.

Some gateways wrap non-streaming Chat Completions in
``{"data": {...}, "success": true}``. The transport unwraps that on JSON
responses so the OpenAI SDK sees top-level ``choices``, while leaving SSE
streams and non-enveloped bodies untouched.
"""

from __future__ import annotations

import asyncio

import httpx

from agent.process_bootstrap import build_keepalive_http_client
from agent.response_envelope_transport import (
    _unwrap_envelope_bytes,
    build_envelope_unwrap_transport,
    is_envelope_unwrap_host,
)

ENVELOPED = (
    b'{"data":{"id":"gen_1","object":"chat.completion","model":"deepseek/x",'
    b'"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},'
    b'"finish_reason":"stop"}],"usage":{"total_tokens":3}},"success":true}'
)
STANDARD = (
    b'{"id":"gen_1","object":"chat.completion",'
    b'"choices":[{"index":0,"message":{"role":"assistant","content":"hi"}}]}'
)
ERROR_ENVELOPE = b'{"error":"Not Found","success":false}'


class TestHostDetection:
    def test_matches_known_host(self):
        assert is_envelope_unwrap_host("https://api.cline.bot/api/v1") is True
        assert is_envelope_unwrap_host("https://API.CLINE.BOT/api/v1") is True
        assert is_envelope_unwrap_host("https://api.cline.bot:443/api/v1") is True
        assert is_envelope_unwrap_host("http://api.cline.bot/v1") is True
        # Bare host / host+path without scheme still match on hostname.
        assert is_envelope_unwrap_host("api.cline.bot/api/v1") is True

    def test_rejects_others(self):
        assert is_envelope_unwrap_host("https://openrouter.ai/api/v1") is False
        assert is_envelope_unwrap_host("https://api.mistral.ai/v1") is False
        assert is_envelope_unwrap_host(None) is False
        assert is_envelope_unwrap_host("") is False

    def test_rejects_substring_false_positives(self):
        # Old check was `host in str(base_url)`, which opted these in.
        assert is_envelope_unwrap_host("https://api.cline.bot.example/v1") is False
        assert is_envelope_unwrap_host("https://evil.example/api.cline.bot/v1") is False
        assert is_envelope_unwrap_host("https://evil.example/?next=api.cline.bot") is False
        assert is_envelope_unwrap_host("https://user:api.cline.bot@evil.example/v1") is False
        assert is_envelope_unwrap_host("https://not-api.cline.bot/v1") is False


class TestUnwrapBytes:
    def test_unwraps_envelope(self):
        import json

        out = _unwrap_envelope_bytes(ENVELOPED)
        body = json.loads(out)
        assert "data" not in body
        assert body["choices"][0]["message"]["content"] == "hi"

    def test_leaves_standard_body(self):
        assert _unwrap_envelope_bytes(STANDARD) == STANDARD

    def test_leaves_error_envelope(self):
        # No data.choices: the 404/error body must reach the SDK intact.
        assert _unwrap_envelope_bytes(ERROR_ENVELOPE) == ERROR_ENVELOPE

    def test_leaves_non_json(self):
        assert _unwrap_envelope_bytes(b"not json at all") == b"not json at all"

    def test_leaves_data_without_choices(self):
        raw = b'{"data":{"ok":true},"success":true}'
        assert _unwrap_envelope_bytes(raw) == raw


class TestSyncTransport:
    def _transport_returning(self, monkeypatch, *, content, content_type):
        def fake_handle(self, request):
            return httpx.Response(
                200,
                headers={"content-type": content_type},
                content=content,
                request=request,
            )

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_handle)
        return build_envelope_unwrap_transport(async_mode=False)

    def test_unwraps_json_response(self, monkeypatch):
        t = self._transport_returning(
            monkeypatch, content=ENVELOPED, content_type="application/json"
        )
        resp = t.handle_request(
            httpx.Request(
                "POST", "https://api.cline.bot/api/v1/chat/completions"
            )
        )
        body = resp.json()
        assert "data" not in body
        assert body["choices"][0]["message"]["content"] == "hi"

    def test_passes_through_sse(self, monkeypatch):
        sse = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        t = self._transport_returning(
            monkeypatch, content=sse, content_type="text/event-stream"
        )
        resp = t.handle_request(
            httpx.Request(
                "POST", "https://api.cline.bot/api/v1/chat/completions"
            )
        )
        assert resp.headers["content-type"] == "text/event-stream"
        assert resp.read() == sse

    def test_error_envelope_survives(self, monkeypatch):
        t = self._transport_returning(
            monkeypatch, content=ERROR_ENVELOPE, content_type="application/json"
        )
        resp = t.handle_request(
            httpx.Request("GET", "https://api.cline.bot/api/v1/models")
        )
        assert resp.json() == {"error": "Not Found", "success": False}


class TestAsyncTransport:
    def test_unwraps_json_response(self, monkeypatch):
        async def fake_handle(self, request):
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=ENVELOPED,
                request=request,
            )

        monkeypatch.setattr(
            httpx.AsyncHTTPTransport, "handle_async_request", fake_handle
        )
        t = build_envelope_unwrap_transport(async_mode=True)

        async def go():
            resp = await t.handle_async_request(
                httpx.Request(
                    "POST", "https://api.cline.bot/api/v1/chat/completions"
                )
            )
            return resp.json()

        body = asyncio.run(go())
        assert "data" not in body
        assert body["choices"][0]["message"]["content"] == "hi"


class TestClientBuilderWiring:
    def test_known_host_uses_unwrap_transport(self):
        client = build_keepalive_http_client(
            "https://api.cline.bot/api/v1", verify=True
        )
        assert isinstance(client, httpx.Client)
        # Mounts carry the unwrap transport class for both schemes.
        mounts = client._mounts
        assert mounts
        names = {type(transport).__name__ for transport in mounts.values()}
        assert names == {"_EnvelopeUnwrapTransport"}
        client.close()

    def test_other_host_keeps_default_transport(self):
        client = build_keepalive_http_client(
            "https://openrouter.ai/api/v1", verify=True
        )
        assert isinstance(client, httpx.Client)
        mounts = client._mounts
        assert mounts
        names = {type(transport).__name__ for transport in mounts.values()}
        assert names == {"HTTPTransport"}
        client.close()

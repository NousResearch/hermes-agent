"""Unit tests for the ClinePass response-envelope unwrap transport.

ClinePass wraps non-streaming responses in ``{"data": {...}, "success": true}``.
The transport unwraps that on JSON responses so the OpenAI SDK sees top-level
``choices``, while leaving SSE streams and non-enveloped bodies untouched.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from agent.clinepass_transport import (
    build_clinepass_transport,
    is_clinepass_base_url,
    _unwrap_envelope_bytes,
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
    def test_matches_clinepass(self):
        assert is_clinepass_base_url("https://api.cline.bot/api/v1") is True
        assert is_clinepass_base_url("https://API.CLINE.BOT/api/v1") is True

    def test_rejects_others(self):
        assert is_clinepass_base_url("https://openrouter.ai/api/v1") is False
        assert is_clinepass_base_url("https://api.mistral.ai/v1") is False
        assert is_clinepass_base_url(None) is False


class TestUnwrapBytes:
    def test_unwraps_envelope(self):
        out = _unwrap_envelope_bytes(ENVELOPED)
        import json

        body = json.loads(out)
        assert "data" not in body
        assert body["choices"][0]["message"]["content"] == "hi"

    def test_leaves_standard_body(self):
        assert _unwrap_envelope_bytes(STANDARD) == STANDARD

    def test_leaves_error_envelope(self):
        # No data.choices -> the 404/error body must reach the SDK intact.
        assert _unwrap_envelope_bytes(ERROR_ENVELOPE) == ERROR_ENVELOPE

    def test_leaves_non_json(self):
        assert _unwrap_envelope_bytes(b"not json at all") == b"not json at all"


class TestSyncTransport:
    def _transport_returning(self, monkeypatch, *, content, content_type):
        def fake_handle(self, request):
            return httpx.Response(
                200, headers={"content-type": content_type}, content=content, request=request
            )

        monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fake_handle)
        return build_clinepass_transport(async_mode=False, socket_options=[])

    def test_unwraps_json_response(self, monkeypatch):
        t = self._transport_returning(
            monkeypatch, content=ENVELOPED, content_type="application/json"
        )
        resp = t.handle_request(
            httpx.Request("POST", "https://api.cline.bot/api/v1/chat/completions")
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
            httpx.Request("POST", "https://api.cline.bot/api/v1/chat/completions")
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

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fake_handle)
        t = build_clinepass_transport(async_mode=True, socket_options=[])

        async def go():
            resp = await t.handle_async_request(
                httpx.Request("POST", "https://api.cline.bot/api/v1/chat/completions")
            )
            return resp.json()

        body = asyncio.run(go())
        assert "data" not in body
        assert body["choices"][0]["message"]["content"] == "hi"

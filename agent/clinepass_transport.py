"""Response-envelope normalization for ClinePass.

ClinePass serves an OpenAI-compatible Chat Completions API, but its gateway
wraps NON-streaming responses in an envelope::

    {"data": {<openai chat completion>}, "success": true}

rather than returning the completion at the top level. The OpenAI SDK parses
that shape with ``choices == None``, which breaks every non-streaming consumer
(auxiliary titles, summaries, compaction, memory) that reads
``response.choices[0].message``. Streaming responses are already standard SSE.

This module provides an httpx transport that unwraps the envelope on JSON
responses so the SDK sees the completion where it expects it. SSE streams and
non-JSON / non-enveloped bodies pass through untouched. It is applied only for
the ClinePass host so no other provider is affected.

The same idea is already used elsewhere in the codebase for non-OpenAI
responses (Bedrock Converse, Anthropic messages, Codex Responses) via the
transport ``normalize_response`` seam; doing it at the wire level keeps it in
one place so every client construction path is covered regardless of how the
caller reads the response.
"""

from __future__ import annotations

import json
from typing import Any

CLINEPASS_HOST = "api.cline.bot"


def is_clinepass_base_url(base_url: Any) -> bool:
    """True if ``base_url`` points at the ClinePass gateway."""
    return CLINEPASS_HOST in str(base_url or "").lower()


def _unwrap_envelope_bytes(raw: bytes) -> bytes:
    """Return the inner completion bytes if ``raw`` is a ClinePass envelope.

    Only unwraps ``{"data": {... "choices": ...}, ...}`` shapes where the top
    level carries no ``choices`` of its own. Error envelopes (``{"error": ...,
    "success": false}``) and already-standard bodies pass through unchanged.
    """
    try:
        body = json.loads(raw)
    except Exception:
        return raw
    if (
        isinstance(body, dict)
        and "choices" not in body
        and isinstance(body.get("data"), dict)
        and "choices" in body["data"]
    ):
        try:
            return json.dumps(body["data"]).encode("utf-8")
        except Exception:
            return raw
    return raw


def _is_json_response(response: Any) -> bool:
    ctype = ""
    try:
        ctype = response.headers.get("content-type", "") or ""
    except Exception:
        return False
    return "application/json" in ctype.lower()


def _rebuild_response(httpx_mod: Any, response: Any, request: Any, new_bytes: bytes) -> Any:
    """Build a fresh httpx.Response carrying the (decoded, unwrapped) body.

    Strips length/encoding headers because ``new_bytes`` is already-decoded
    identity content; httpx recomputes Content-Length from the body.
    """
    headers = httpx_mod.Headers(response.headers)
    for stale in ("content-length", "content-encoding", "transfer-encoding"):
        if stale in headers:
            del headers[stale]
    return httpx_mod.Response(
        status_code=response.status_code,
        headers=headers,
        content=new_bytes,
        request=request,
        extensions=getattr(response, "extensions", None) or {},
    )


def build_clinepass_transport(*, async_mode: bool, socket_options: Any) -> Any:
    """Return an httpx transport that unwraps the ClinePass response envelope.

    ``socket_options`` mirror the keepalive options used by the normal
    keepalive client so behaviour is identical apart from the unwrap.
    """
    import httpx

    if async_mode:

        class _ClinePassUnwrapAsyncTransport(httpx.AsyncHTTPTransport):
            async def handle_async_request(self, request: Any) -> Any:
                response = await super().handle_async_request(request)
                if not _is_json_response(response):
                    return response
                await response.aread()
                return _rebuild_response(
                    httpx, response, request, _unwrap_envelope_bytes(response.content)
                )

        return _ClinePassUnwrapAsyncTransport(socket_options=socket_options)

    class _ClinePassUnwrapTransport(httpx.HTTPTransport):
        def handle_request(self, request: Any) -> Any:
            response = super().handle_request(request)
            if not _is_json_response(response):
                return response
            response.read()
            return _rebuild_response(
                httpx, response, request, _unwrap_envelope_bytes(response.content)
            )

    return _ClinePassUnwrapTransport(socket_options=socket_options)

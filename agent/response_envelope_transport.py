"""Wire-level unwrap for OpenAI-compat gateways that nest completions.

Some OpenAI-compatible gateways wrap non-streaming Chat Completions in::

    {"data": {<openai chat completion>}, "success": true}

instead of returning the completion at the top level. The OpenAI SDK then
parses ``choices is None``, which breaks every non-streaming consumer
(auxiliary titles, summaries, compaction, memory). Streaming SSE is already
standard and must pass through unchanged.

This module provides sync and async httpx transports that unwrap that
envelope on JSON responses when the client targets a known host. Other hosts
keep the default transport. The transports honor the same ``verify`` / proxy
kwargs the keepalive builders already pass for TLS and proxy policy.

This is HTTP response normalization in core client construction. It is not a
provider plugin, catalog entry, or product onboarding path. A standalone
model-provider plugin cannot inject an httpx transport into the keepalive
builders, so the envelope cannot be fixed out-of-tree.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

# Hosts known to wrap non-streaming Chat Completions in {data, success}.
# Compared as exact hostnames only (not substring of the full URL) so
# lookalikes like api.cline.bot.example never opt in.
ENVELOPE_UNWRAP_HOSTS = frozenset({"api.cline.bot"})


def is_envelope_unwrap_host(base_url: Any) -> bool:
    """True if ``base_url``'s hostname is on the envelope-unwrap allowlist.

    Parses the URL and compares the normalized hostname. Substring matches on
    the raw string (userinfo, path, sibling domains) do not count.
    """
    raw = str(base_url or "").strip()
    if not raw:
        return False
    # Bare host or host/path without a scheme still has to resolve.
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.hostname or "").lower()
    return host in ENVELOPE_UNWRAP_HOSTS


def _unwrap_envelope_bytes(raw: bytes) -> bytes:
    """Return the inner completion bytes if ``raw`` is a nested envelope.

    Only unwraps ``{"data": {... "choices": ...}, ...}`` shapes where the top
    level carries no ``choices`` of its own. Error envelopes
    (``{"error": ..., "success": false}``) and already-standard bodies pass
    through unchanged.
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


def _rebuild_response(
    httpx_mod: Any, response: Any, request: Any, new_bytes: bytes
) -> Any:
    """Build a fresh httpx.Response carrying the (decoded, unwrapped) body.

    Strips length/encoding headers because ``new_bytes`` is already-decoded
    identity content. httpx recomputes Content-Length from the body.
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


def build_envelope_unwrap_transport(
    *,
    async_mode: bool,
    verify: Any = True,
    proxy: Any = None,
) -> Any:
    """Return an httpx transport that unwraps nested completion envelopes.

    ``verify`` and ``proxy`` mirror the kwargs used by the keepalive client
    builders so TLS and proxy behavior stay identical apart from the unwrap.
    """
    import httpx

    transport_kwargs: dict[str, Any] = {"verify": verify}
    if proxy is not None:
        transport_kwargs["proxy"] = proxy

    if async_mode:

        class _EnvelopeUnwrapAsyncTransport(httpx.AsyncHTTPTransport):
            async def handle_async_request(self, request: Any) -> Any:
                response = await super().handle_async_request(request)
                if not _is_json_response(response):
                    return response
                await response.aread()
                return _rebuild_response(
                    httpx,
                    response,
                    request,
                    _unwrap_envelope_bytes(response.content),
                )

        return _EnvelopeUnwrapAsyncTransport(**transport_kwargs)

    class _EnvelopeUnwrapTransport(httpx.HTTPTransport):
        def handle_request(self, request: Any) -> Any:
            response = super().handle_request(request)
            if not _is_json_response(response):
                return response
            response.read()
            return _rebuild_response(
                httpx,
                response,
                request,
                _unwrap_envelope_bytes(response.content),
            )

    return _EnvelopeUnwrapTransport(**transport_kwargs)

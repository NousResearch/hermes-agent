"""Describe a non-2xx answer on the gateway -> connector HTTP routes. EXPERIMENTAL.

An egress proxy between the gateway and the connector answers with its OWN
status codes. A proxy's 403 page looks exactly like a connector refusal unless
the log says who sent it. The connector always answers a refusal with a JSON
``{"error": "<reason>"}`` body. A proxy answers with an HTML page and identifies
itself in ``Server`` / ``Via`` / ``Proxy-Status`` (RFC 9209) / ``X-Squid-Error``.

Only RESPONSE data is described: request headers (the bearer) are never read
here, and the body excerpt is short and redacted.
"""

from __future__ import annotations

import http.client
import json
import urllib.error

from agent.redact import redact_sensitive_text

# Response headers that name the hop that answered.
_ORIGIN_HEADERS = ("Server", "Via", "Proxy-Status", "X-Squid-Error")
BODY_EXCERPT_CHARS = 200
_BODY_READ_LIMIT = 4096


def read_error_body(exc: urllib.error.HTTPError) -> bytes:
    """The (bounded) body of an HTTP error response; ``b""`` when unreadable."""
    try:
        return exc.read(_BODY_READ_LIMIT) or b""
    except (OSError, ValueError, http.client.HTTPException):
        return b""


def connector_reason(body: bytes) -> str:
    """The connector's own refusal reason (JSON ``{"error": ...}``), else ``""``."""
    try:
        parsed = json.loads(body.decode("utf-8"))
    except ValueError:  # not UTF-8 / not JSON: a proxy page, not the connector
        return ""
    reason = parsed.get("error") if isinstance(parsed, dict) else None
    return reason.strip() if isinstance(reason, str) else ""


def describe_http_error(exc: urllib.error.HTTPError, body: bytes) -> str:
    """``HTTP 403 (server=squid/6.10 x-squid-error=ERR_DNS_FAIL 0 body='<html>…')``.

    The body is reduced to one line and its first ``BODY_EXCERPT_CHARS`` characters,
    then redacted, so a misbehaving hop that echoes a credential cannot get it logged.
    """
    parts = []
    headers = getattr(exc, "headers", None)
    for name in _ORIGIN_HEADERS:
        value = headers.get(name) if headers is not None else None
        if value:
            parts.append(f"{name.lower()}={' '.join(str(value).split())[:120]}")
    text = " ".join(body.decode("utf-8", "replace").split())
    if text:
        excerpt = text[:BODY_EXCERPT_CHARS] + ("…" if len(text) > BODY_EXCERPT_CHARS else "")
        parts.append(f"body={redact_sensitive_text(excerpt, force=True)!r}")
    return f"HTTP {exc.code}" + (f" ({' '.join(parts)})" if parts else "")


__all__ = ["BODY_EXCERPT_CHARS", "connector_reason", "describe_http_error", "read_error_body"]

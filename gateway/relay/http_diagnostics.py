"""Describe a non-2xx answer on the gateway -> connector HTTP routes. EXPERIMENTAL.

An egress proxy between the gateway and the connector answers with its OWN
status codes. A proxy's 403 page looks exactly like a connector refusal unless
the log says who sent it. The connector always answers a refusal with a JSON
``{"error": "<reason>"}`` body and sets none of the headers below. A proxy
answers with an HTML page (some, Cloudflare/Envoy style, with JSON) and identifies
itself in ``Server`` / ``Via`` / ``Proxy-Status`` (RFC 9209) / ``X-Squid-Error``;
``answered_by_proxy`` reads those so a proxy's JSON is not taken for a refusal.

Only RESPONSE data is described: request headers (the bearer) are never read
here, and the body excerpt is short and redacted. A hop can still echo the
request's bearer into its body bare (``token <x> rejected``), where no generic
pattern recognises it, so callers pass the exact credentials they sent as
``secrets`` and those are replaced literally first.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
from typing import Iterable

from agent.redact import redact_sensitive_text

# Response headers that name the hop that answered.
_ORIGIN_HEADERS = ("Server", "Via", "Proxy-Status", "X-Squid-Error")
# Headers only an intermediary sets (the connector sets none of them, not even ``Server``).
_PROXY_ONLY_HEADERS = ("Via", "Proxy-Status", "X-Squid-Error")
# ``Server`` values naming an egress/CDN/mesh proxy that can answer on its own with a
# JSON ``{"error"}`` body. A hosting edge's own name (e.g. ``railway-edge``) is not here.
_PROXY_SERVER_NAMES = ("squid", "envoy", "cloudflare", "zscaler", "bluecoat", "proxysg", "netskope", "forcepoint")
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


def answered_by_proxy(exc: urllib.error.HTTPError) -> bool:
    """A response header names a hop other than the connector, so a JSON ``{"error"}``
    body is that hop's (a Cloudflare/Envoy-style 403), not a connector refusal."""
    headers = getattr(exc, "headers", None)
    if headers is None:
        return False
    if any(headers.get(name) for name in _PROXY_ONLY_HEADERS):
        return True
    server = str(headers.get("Server") or "").lower()
    return any(name in server for name in _PROXY_SERVER_NAMES)


def redact_excerpt(text: str, secrets: Iterable[str] = ()) -> str:
    """``text`` with every exact credential in ``secrets`` replaced, then generically redacted."""
    return redact_sensitive_text(_scrub(text, secrets), force=True)


def _scrub(text: str, secrets: Iterable[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


def describe_http_error(exc: urllib.error.HTTPError, body: bytes, *, secrets: Iterable[str] = ()) -> str:
    """``HTTP 403 (server=squid/6.10 x-squid-error=ERR_DNS_FAIL 0 body='<html>…')``.

    The body is reduced to one line, stripped of the exact credentials the caller sent
    (``secrets``) before it is cut, so a token straddling the cut cannot leave a prefix,
    then its first ``BODY_EXCERPT_CHARS`` characters are redacted.
    """
    parts = []
    headers = getattr(exc, "headers", None)
    for name in _ORIGIN_HEADERS:
        value = headers.get(name) if headers is not None else None
        if value:
            parts.append(f"{name.lower()}={' '.join(str(value).split())[:120]}")
    secrets = tuple(secrets)
    text = _scrub(" ".join(body.decode("utf-8", "replace").split()), secrets)
    if text:
        excerpt = text[:BODY_EXCERPT_CHARS] + ("…" if len(text) > BODY_EXCERPT_CHARS else "")
        parts.append(f"body={redact_excerpt(excerpt, secrets)!r}")
    return f"HTTP {exc.code}" + (f" ({' '.join(parts)})" if parts else "")


__all__ = [
    "BODY_EXCERPT_CHARS", "answered_by_proxy", "connector_reason", "describe_http_error", "read_error_body",
    "redact_excerpt",
]

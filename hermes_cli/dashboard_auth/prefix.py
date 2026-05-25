"""Helpers for X-Forwarded-Prefix support.

Mission-control style deploys reverse-proxy the dashboard at a path
prefix (e.g. ``mission-control.tilos.com/hermes/*`` -> dashboard on
:9119). The proxy injects ``X-Forwarded-Prefix: /hermes`` so the
backend can reconstruct prefixed URLs (Location: headers, OAuth
redirect_uri, cookie Path attributes, SPA asset URLs).

The single source of truth for the parsed prefix lives here so the
gate middleware, the OAuth routes, the cookie helpers, and the SPA
mount all agree on validation rules.
"""
from __future__ import annotations

from typing import Optional


def normalise_prefix(raw: Optional[str]) -> str:
    """Normalise an X-Forwarded-Prefix header value.

    Returns a string like ``"/hermes"`` (no trailing slash) or ``""``
    when no prefix is set / the header is malformed. We deliberately
    reject anything containing ``..`` or non-printable bytes so a
    hostile proxy can't inject HTML or path-traversal sequences via the
    prefix.
    """
    if not raw:
        return ""
    p = raw.strip()
    if not p:
        return ""
    if not p.startswith("/"):
        p = "/" + p
    p = p.rstrip("/")
    if (
        "//" in p
        or ".." in p
        or any(c in p for c in ('"', "'", "<", ">", " ", "\n", "\r", "\t"))
    ):
        return ""
    if len(p) > 64:
        return ""
    return p


def prefix_from_request(request) -> str:
    """Convenience wrapper that reads the header off a Starlette/FastAPI
    Request and normalises it. Returns ``""`` when no prefix.
    """
    return normalise_prefix(request.headers.get("x-forwarded-prefix"))

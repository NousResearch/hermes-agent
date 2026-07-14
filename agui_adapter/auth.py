"""Authorization for the AG-UI adapter's HTTP surface (SECURITY.md §2.6).

Loopback bind → OS-level access control is the authorization; a session token
is optional defense-in-depth. Non-loopback bind → fail closed: refuse to start
without a usable HERMES_AGUI_SESSION_TOKEN. Mirrors the API server
(gateway/platforms/api_server.py) and dashboard (hermes_cli/web_server.py).
"""
from __future__ import annotations

import hmac
import logging
from typing import Optional

from gateway.platforms.base import is_network_accessible

logger = logging.getLogger(__name__)

_MIN_TOKEN_LEN = 16
_SESSION_HEADER_NAME = "X-Hermes-Session-Token"


def _usable(token: Optional[str]) -> bool:
    try:
        from hermes_cli.auth import has_usable_secret
        return bool(token) and has_usable_secret(token, min_length=_MIN_TOKEN_LEN)
    except ImportError:
        logger.warning(
            "hermes_cli.auth.has_usable_secret unavailable; using a reduced "
            "inline length + placeholder-denylist check for the AG-UI bind (the "
            "full hermes_cli placeholder set was not applied).", )
        t = (token or "").strip()
        _FALLBACK_PLACEHOLDERS = {"changeme", "placeholder", "your_api_key_here",
                                  "changeme00000000", "placeholder-value", "secret",
                                  "token", "your_token_here"}
        return len(t) >= _MIN_TOKEN_LEN and t.lower() not in _FALLBACK_PLACEHOLDERS


def require_token_or_refuse(host: str, token: Optional[str]) -> None:
    """Exit the process if *host* is network-accessible without a usable token."""
    if not is_network_accessible(host):
        return  # loopback: OS boundary is the authorization
    if not _usable(token):
        logger.error(
            "Refusing to start: HERMES_AGUI_HOST=%s is network-accessible, so "
            "HERMES_AGUI_SESSION_TOKEN is required and must be >=%d chars. This "
            "endpoint can dispatch terminal-capable agent work; an open or "
            "guessable bind is remote code execution. Generate a strong secret "
            "(e.g. `openssl rand -hex 32`).",
            host, _MIN_TOKEN_LEN,
        )
        raise SystemExit(1)


def token_valid(request, token: str) -> bool:
    """True if the request carries the configured session token (header or ?token=)."""
    header = request.headers.get(_SESSION_HEADER_NAME, "")
    if header and hmac.compare_digest(header.encode(), token.encode()):
        return True
    q = request.query_params.get("token", "")
    return bool(q) and hmac.compare_digest(q.encode(), token.encode())


def host_accepted(host_header: str, bound_host: str) -> bool:
    """DNS-rebinding guard: True if the Host header targets the bound interface.

    Loopback bind accepts loopback names; an explicit 0.0.0.0/:: bind accepts
    anything (operator opted into all interfaces); otherwise require an exact
    host match. Based on hermes_cli/web_server.py::_is_accepted_host, with an
    added branch for bare (unbracketed) IPv6 literals like ``::1`` that the
    sibling mangles.
    """
    loopback = {"localhost", "127.0.0.1", "::1"}
    if not host_header:
        return False
    h = host_header.strip()
    if h.startswith("["):
        close = h.find("]")
        host_only = h[1:close] if close != -1 else h.strip("[]")
    elif h.count(":") > 1:
        host_only = h            # bare IPv6 literal, no port
    else:
        host_only = h.rsplit(":", 1)[0] if ":" in h else h
    host_only = host_only.lower()
    if bound_host in {"0.0.0.0", "::"}:
        return True
    b = bound_host.lower()
    if b in loopback:
        return host_only in loopback
    return host_only == b

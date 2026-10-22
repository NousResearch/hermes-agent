"""Authorization for the AG-UI adapter's HTTP surface (SECURITY.md §2.6).

Loopback bind → OS-level access control is the authorization; a session token
is optional defense-in-depth. Non-loopback bind → fail closed: refuse to start
without a usable HERMES_AGUI_SESSION_TOKEN. Mirrors the API server
(gateway/platforms/api_server.py) and dashboard (hermes_cli/web_server.py).
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import re
from typing import Optional

from gateway.platforms.base import is_network_accessible

logger = logging.getLogger(__name__)

_MIN_TOKEN_LEN = 16
_SESSION_HEADER_NAME = "X-Hermes-Session-Token"


def _usable(token: Optional[str]) -> bool:
    if (token or "").strip().lower() in {"changeme00000000", "placeholder-value"}:
        return False
    try:
        from hermes_cli.auth import has_usable_secret

        return bool(token) and has_usable_secret(token, min_length=_MIN_TOKEN_LEN)
    except ImportError:
        logger.warning(
            "hermes_cli.auth.has_usable_secret unavailable; using a reduced "
            "inline length + placeholder-denylist check for the AG-UI bind (the "
            "full hermes_cli placeholder set was not applied).",
        )
        t = (token or "").strip()
        _FALLBACK_PLACEHOLDERS = {
            "changeme",
            "placeholder",
            "your_api_key_here",
            "changeme00000000",
            "placeholder-value",
            "secret",
            "token",
            "your_token_here",
        }
        return len(t) >= _MIN_TOKEN_LEN and t.lower() not in _FALLBACK_PLACEHOLDERS


def require_token_or_refuse(host: str, token: Optional[str]) -> None:
    """Exit the process if *host* is network-accessible without a usable token."""
    if not is_network_accessible(host):
        return  # loopback: OS boundary is the authorization
    if not _usable(token):
        logger.error(
            "Refusing to start: bind host %s (config.yaml `agui.host` or --host) "
            "is network-accessible, so a session token is required and must be "
            ">=%d chars. Set the HERMES_AGUI_SESSION_TOKEN secret (or --token). "
            "This endpoint can dispatch terminal-capable agent work; an open or "
            "guessable bind is remote code execution. Generate a strong secret "
            "(e.g. `openssl rand -hex 32`).",
            host,
            _MIN_TOKEN_LEN,
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

    Accept complete hostname/IP authorities, including supported bare IPv6.
    Loopback aliases share the startup guard's effective loopback policy;
    wildcard binds still require valid authority syntax before accepting it.
    """
    match = re.fullmatch(
        r"(?:\[([0-9a-fA-F:.]+)\]|([a-zA-Z0-9_.-]+))(?::([0-9]+))?"
        r"|([0-9a-fA-F:.]*:[0-9a-fA-F:.]+)",
        host_header,
    )
    if not match:
        return False
    host_only = match[1] or match[2] or match[4]
    if match[3] is not None:
        port = match[3].lstrip("0") or "0"
        if len(port) > 5 or int(port) > 65535:
            return False
    try:
        address = ipaddress.ip_address(host_only)
    except ValueError:
        if match[1] or match[4]:  # Bracketed and bare IPv6 must be IP literals.
            return False
        address = None
    if match[1] and address.version != 6:
        return False
    try:
        bound = ipaddress.ip_address(bound_host)
    except ValueError:
        bound = None
    # IPv4-mapped loopback has the same OS boundary as its IPv4 address.
    effective_address = getattr(address, "ipv4_mapped", None) or address
    effective_bound = getattr(bound, "ipv4_mapped", None) or bound
    if bound and bound.is_unspecified:
        return True
    if bound_host.lower() == "localhost" or (
        effective_bound and effective_bound.is_loopback
    ):
        return host_only.lower() == "localhost" or bool(
            effective_address and effective_address.is_loopback
        )
    return (
        address == bound
        if address and bound
        else host_only.lower() == bound_host.lower()
    )

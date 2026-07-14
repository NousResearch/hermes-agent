"""Shared helpers for the two Compresr plugins (``context_engine/compresr`` and
``tool_output_compresr``).

Kept in one module so the security-sensitive base-URL / host validation and the
secret sanitization can't drift between the plugins. stdlib-only.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from typing import Any

logger = logging.getLogger("plugins.compresr")

# Cap on any single API response we read into memory.
MAX_RESPONSE_BYTES = 32 * 1024 * 1024

_SECRET_CTL_RE = re.compile(r"[\r\n\x00]")
# Numeric-shorthand IPv4 (decimal or hex, e.g. 2852039166 / 0xa9fea9fe) that
# getaddrinfo would resolve to a real address — refused outright.
_NUMERIC_ONLY_HOST_RE = re.compile(r"\A(?:0x[0-9a-fA-F]+|[0-9]+)\Z")

_BLOCKED_METADATA_HOST_NAMES = frozenset({
    "metadata.google.internal", "metadata.goog", "metadata",
})
# Link-local (incl. cloud metadata), private, CGNAT, and IPv6 local ranges.
_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(n) for n in (
        "169.254.0.0/16", "fd00::/8", "fe80::/10",
        "100.100.100.200/32", "168.63.129.16/32",
        "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10",
    )
)
_LOCALHOST_NAMES = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def read_with_cap(resp: Any, cap: int) -> bytes:
    """Bounded read tolerant of mocks whose ``read()`` takes no size arg."""
    try:
        raw = resp.read(cap + 1)
    except TypeError:
        raw = resp.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if len(raw) > cap:
        raise RuntimeError(f"response exceeded {cap} bytes")
    return raw


def sanitize_secret(raw: str, label: str) -> str:
    """Strip whitespace and reject CR/LF/NUL (urllib would raise with the raw key
    in the message). Returns ``""`` on rejection, never the value."""
    if not raw:
        return ""
    stripped = raw.strip()
    if _SECRET_CTL_RE.search(stripped):
        logger.error("compresr: %s contained CR/LF/NUL and was rejected", label)
        return ""
    return stripped


def _resolve_host_ips(host: str) -> tuple:
    if not host:
        return ()
    ips: list = []
    try:
        ip = ipaddress.ip_address(host)
        mapped = getattr(ip, "ipv4_mapped", None)
        ips.append(mapped if mapped is not None else ip)
        return tuple(ips)
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, socket.herror, UnicodeError, OSError):
        return ()
    for _, _, _, _, sockaddr in infos:
        try:
            ip = ipaddress.ip_address(sockaddr[0].split("%", 1)[0])
        except ValueError:
            continue
        mapped = getattr(ip, "ipv4_mapped", None)
        ips.append(mapped if mapped is not None else ip)
    return tuple(ips)


def _is_blocked_host(host: str) -> bool:
    h = (host or "").rstrip(".").lower()
    if h in _BLOCKED_METADATA_HOST_NAMES:
        return True
    return any(ip in net for ip in _resolve_host_ips(h) for net in _BLOCKED_NETWORKS)


def _safe_url_repr(parsed) -> str:
    if parsed is None:
        return "?"
    try:
        return f"{parsed.scheme or '?'}://{parsed.hostname or '?'}"
    except Exception:
        return "?"


def secure_base_url(url: str, default: str) -> str:
    """Reject non-HTTPS (except localhost), numeric-shorthand IPv4 literals,
    cloud-metadata hosts, and hosts resolving to a private/link-local net.
    Logs only scheme://host — the raw URL may carry userinfo credentials.
    """
    from urllib.parse import urlparse

    try:
        parsed = urlparse(url)
    except Exception:
        parsed = None
    host = (parsed.hostname or "").lower() if parsed else ""
    safe = _safe_url_repr(parsed)

    if parsed and host and _NUMERIC_ONLY_HOST_RE.match(host):
        logger.warning("compresr: refusing numeric-shorthand IPv4 host %s; using %s", safe, default)
        return default
    if parsed and parsed.scheme in ("http", "https") and host in _LOCALHOST_NAMES:
        return url
    if parsed and _is_blocked_host(host):
        logger.warning("compresr: refusing base_url %s (metadata/private host); using %s", safe, default)
        return default
    if parsed and parsed.scheme == "https":
        return url
    logger.warning("compresr: ignoring insecure base_url %s (must be https); using %s", safe, default)
    return default

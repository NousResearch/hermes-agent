"""Fail-closed URL identity normalization for model/provider routes."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit


def normalize_route_base_url(base_url: Any) -> str:
    """Canonicalize only proven-equivalent endpoint URL components."""
    raw = str(base_url or "")
    if not raw:
        return ""
    if any(ord(char) <= 0x20 for char in raw):
        return raw
    had_query_delimiter = "?" in raw.split("#", 1)[0]
    try:
        parsed = urlsplit(raw)
        hostname = parsed.hostname
        if not parsed.scheme or not hostname:
            return raw
        scheme = parsed.scheme.lower()
        if "%" in hostname:
            address, zone = hostname.split("%", 1)
            host = f"{address.lower()}%{zone}"
        else:
            host = hostname.lower()
        port = parsed.port
    except (TypeError, ValueError):
        return raw

    route_host = parsed.netloc.rsplit("@", 1)[-1]
    if route_host.startswith("[") or ":" in host:
        host = f"[{host}]"
    if port is not None and (scheme, port) not in {("http", 80), ("https", 443)}:
        host = f"{host}:{port}"
    if "@" in parsed.netloc:
        host = f"{parsed.netloc.rsplit('@', 1)[0]}@{host}"

    path = parsed.path
    if path.endswith("/") and not had_query_delimiter:
        path = path[:-1]

    normalized = urlunsplit((scheme, host, path, parsed.query, ""))
    if had_query_delimiter and not parsed.query:
        normalized += "?"
    return normalized


def should_clear_context_pin(
    configured_model: Any,
    active_model: Any,
    configured_base_url: Any,
    active_base_url: Any,
    configured_provider: Any,
    active_provider: Any,
) -> bool:
    """True when a configured ``model.context_length`` pin no longer matches its runtime route.

    Fail-closed: any error during route comparison returns ``True`` (drop the pin)
    so a stale window never silently inflates the compression threshold.
    """
    configured_model = str(configured_model or "").strip()
    if configured_model and configured_model != str(active_model or "").strip():
        return True
    try:
        from agent.agent_init import _context_route_mismatch

        return _context_route_mismatch(
            configured_base_url,
            active_base_url,
            configured_provider,
            active_provider,
        )
    except Exception:
        return True

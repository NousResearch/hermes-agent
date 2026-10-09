"""Canonical provider-route identity helpers."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit, urlunsplit

from providers.identity import normalize_provider


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


def is_foreign_provider_endpoint(provider: str = "", base_url: str = "") -> bool:
    """Whether *base_url* is another registered provider's canonical endpoint."""
    from providers.registry import get_provider_profile, list_providers

    profile = get_provider_profile(str(provider or "").strip().lower())
    route_url = normalize_route_base_url(base_url)
    if profile is None or not route_url:
        return False
    own_url = normalize_route_base_url(profile.base_url)
    if route_url == own_url:
        return False
    return any(
        route_url == normalize_route_base_url(other.base_url)
        for other in list_providers()
        if other.base_url
    )


def is_actual_route(provider: str = "", base_url: str = "") -> bool:
    """Return whether provider identity or endpoint selects the Actual runtime."""
    if normalize_provider(provider or "") == "actual":
        return True
    try:
        hostname = (urlsplit(str(base_url or "")).hostname or "").lower().rstrip(".")
    except (TypeError, ValueError):
        return False
    return hostname == "api.actual.inc"


def is_actual_local_base_url(base_url: str) -> bool:
    """Actual's local no-auth transport is restricted to loopback hosts."""
    try:
        hostname = (urlsplit(str(base_url or "")).hostname or "").lower().rstrip(".")
    except (TypeError, ValueError):
        return False
    return hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def normalize_actual_base_url(base_url: str) -> str:
    """Actual's OpenAI-compatible hosted and loopback endpoints require /v1."""
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        from providers.registry import get_provider_profile

        profile = get_provider_profile("actual")
        return str(getattr(profile, "base_url", "") or "https://api.actual.inc/v1")
    try:
        parts = urlsplit(url)
        hostname = (parts.hostname or "").lower().rstrip(".")
        path = parts.path.rstrip("/")
    except (TypeError, ValueError):
        return url
    if path in {"", "/"} and (hostname == "api.actual.inc" or is_actual_local_base_url(url)):
        return url + "/v1"
    return url


__all__ = [
    "is_actual_local_base_url",
    "is_actual_route",
    "is_foreign_provider_endpoint",
    "normalize_actual_base_url",
    "normalize_route_base_url",
]

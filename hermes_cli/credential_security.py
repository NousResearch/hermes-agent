"""
Credential security helpers — shared across web_server, CLI, and resolver.

These validators prevent SSRF, path traversal, and injection attacks on
credential pool operations. They are intentionally side-effect free and
raise ValueError (not HTTPException) so they can be used from any layer.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

# ── Blocked hosts for SSRF prevention ─────────────────────────────────────

_BLOCKED_HOSTS = frozenset({
    "169.254.169.254",      # AWS / Azure / GCP metadata (IPv4)
    "metadata.google.internal",  # GCP metadata (DNS)
    "metadata",             # Short form
    "fd00:ec2::254",        # AWS IMDSv6
})

# Link-local prefix (169.254.0.0/16) — includes AWS metadata
_LINK_LOCAL_RE = re.compile(r"^169\.254\.")

# Allowed schemes
_ALLOWED_SCHEMES = frozenset({"http", "https", ""})


def validate_base_url_safe(url: str) -> str:
    """Sanitize a user-supplied base_url to prevent SSRF.

    Blocks:
    - Non-http(s) schemes (file://, gopher://, dict://, etc.)
    - Cloud metadata endpoints (169.254.169.254, metadata.google.internal)
    - Internal link-local addresses (169.254.x.x)
    - Null byte injection

    Returns the URL unchanged if valid.
    Raises ValueError if blocked.
    """
    if not url or not url.strip():
        return url

    url = url.strip()

    # Null byte injection
    if "\x00" in url:
        raise ValueError("Null byte in URL")

    # Parse scheme
    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()

    # Allow bare hostnames (no scheme) — the caller will prepend https://
    if scheme and scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Blocked scheme: {scheme}://")

    # Check host against blocked list
    host = (parsed.hostname or "").lower().rstrip(".")
    if host in _BLOCKED_HOSTS:
        raise ValueError(f"Blocked internal host: {host}")
    if _LINK_LOCAL_RE.match(host):
        raise ValueError(f"Blocked link-local host: {host}")

    return url


def validate_provider_name(provider: str) -> str:
    """Validate provider name to prevent path traversal.

    Only allows lowercase alphanumeric, hyphens, and underscores.
    Max 64 characters.
    """
    if not provider or not provider.strip():
        raise ValueError("Provider name required")

    provider = provider.strip().lower()

    if len(provider) > 64:
        raise ValueError("Provider name too long (max 64)")

    if "/" in provider or "\\" in provider or ".." in provider or "\x00" in provider:
        raise ValueError("Invalid provider name")

    safe_chars = set("abcdefghijklmnopqrstuvwxyz0123456789-_")
    if any(c not in safe_chars for c in provider):
        raise ValueError("Invalid provider name")

    return provider


# ── Supported pool strategies ─────────────────────────────────────────────

SUPPORTED_POOL_STRATEGIES = frozenset({
    "fill_first",
    "round_robin",
    "least_used",
    "random",
    "failover",
})


def validate_pool_strategy(strategy: str) -> str:
    """Validate that a strategy name is in the supported set."""
    if not strategy or not strategy.strip():
        raise ValueError("Strategy required")
    strategy = strategy.strip().lower()
    if strategy not in SUPPORTED_POOL_STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    return strategy

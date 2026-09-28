"""Shared HTTP client factory for long-lived platform adapters.

httpx's default ``keepalive_expiry`` (5s) lets peer-initiated FIN sit in
``CLOSE_WAIT`` behind transparent proxies (macOS + Cloudflare Warp); across 7
adapters plus LLM/MCP clients that walks into the default 256 fd limit.
``platform_httpx_limits()`` returns tighter ``httpx.Limits``: 10 keepalive
connections (platform APIs rarely parallelise beyond this), 2.0s expiry.
Override via ``HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY`` /
``HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE``; a ``HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE``
of ``0`` disables keepalive entirely (httpx keeps no idle connections).
"""

from __future__ import annotations

import os

try:
    import httpx
except ImportError:  # pragma: no cover — optional dep
    httpx = None  # type: ignore[assignment]


_DEFAULT_KEEPALIVE_EXPIRY_S = 2.0
_DEFAULT_MAX_KEEPALIVE = 10


def _parse_env(name: str, default, cast):
    """``cast(env)`` when set and parseable; else *default*."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


def _positive_env(name: str, default, cast):
    """``cast(env)`` when set, parseable and > 0; else *default*."""
    val = _parse_env(name, default, cast)
    return val if val > 0 else default


def _non_negative_env(name: str, default, cast):
    """``cast(env)`` when set, parseable and >= 0; else *default*.

    Zero is a meaningful httpx setting (keep no idle connections), so it is
    preserved here; only negatives fall back to *default*.
    """
    val = _parse_env(name, default, cast)
    return val if val >= 0 else default


def platform_httpx_limits() -> "httpx.Limits | None":
    """``httpx.Limits`` tuned for persistent platform-adapter clients; ``None`` without httpx."""
    if httpx is None:
        return None
    # max_connections stays at the httpx default (100) — plenty of headroom.
    return httpx.Limits(
        max_keepalive_connections=_non_negative_env(
            "HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", _DEFAULT_MAX_KEEPALIVE, int),
        keepalive_expiry=_positive_env(
            "HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", _DEFAULT_KEEPALIVE_EXPIRY_S, float),
    )

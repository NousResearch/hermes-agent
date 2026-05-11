"""Shared HTTP client factory for long-lived platform adapters.

Gateway messaging platforms (QQ Bot, Feishu, WeCom, DingTalk, Signal,
BlueBubbles, WeCom-callback) keep a persistent ``httpx.AsyncClient``
alive for the adapter's lifetime.  That amortises TLS/connection setup
across many API calls, but it also means the process's file-descriptor
pressure is sensitive to how aggressively the pool recycles idle keep-
alive connections.

httpx's default ``keepalive_expiry`` is 5 seconds.  On macOS behind
Cloudflare Warp (and other transparent proxies), peer-initiated FIN can
sit in ``CLOSE_WAIT`` longer than that before the local socket actually
drains — which, multiplied across 7 long-lived adapters plus the LLM
client and MCP clients, walks straight into the default 256 fd limit.
See #18451.

``platform_httpx_limits()`` returns a tighter ``httpx.Limits`` the
adapter factories use instead of the httpx default.  The values chosen:

* ``max_keepalive_connections=10`` — plenty for any single adapter;
  platform APIs rarely parallelise beyond this.
* ``keepalive_expiry=2.0`` — close idle sockets aggressively so a
  proxy's lingering CLOSE_WAIT window can't starve the process.

Override via ``HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY`` /
``HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE`` env vars when tuning under load.
"""

from __future__ import annotations

import os

try:
    import httpx
except ImportError:  # pragma: no cover — optional dep
    httpx = None  # type: ignore[assignment]


_DEFAULT_KEEPALIVE_EXPIRY_S = 2.0
_DEFAULT_MAX_KEEPALIVE = 10


def platform_httpx_limits() -> "httpx.Limits | None":
    """Return ``httpx.Limits`` tuned for persistent platform-adapter clients.

    Returns ``None`` when httpx isn't importable, so callers can fall
    back to httpx's built-in default without a hard dependency on this
    helper being reachable.
    """
    if httpx is None:
        return None

    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return default
        return val if val > 0 else default

    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            val = int(raw)
        except (TypeError, ValueError):
            return default
        return val if val > 0 else default

    keepalive_expiry = _env_float(
        "HERMES_GATEWAY_HTTPX_KEEPALIVE_EXPIRY", _DEFAULT_KEEPALIVE_EXPIRY_S
    )
    max_keepalive = _env_int(
        "HERMES_GATEWAY_HTTPX_MAX_KEEPALIVE", _DEFAULT_MAX_KEEPALIVE
    )

    return httpx.Limits(
        max_keepalive_connections=max_keepalive,
        # Leave max_connections at httpx default (100) — plenty of headroom.
        keepalive_expiry=keepalive_expiry,
    )

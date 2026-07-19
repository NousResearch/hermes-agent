"""Provider-agnostic rate/credit probe for the Arc status bar's CAPACITY line.

The user's Claude Code statusline shows Anthropic subscription 5H/7D windows +
overage credits. Hermes is provider-agnostic, so there is no single "rate limit"
concept. What every provider *does* expose is some notion of remaining balance.
This module returns that as a normalized ``rate_limits`` dict the TUI renders:

    {"source": "openrouter",
     "credits": {"remaining_usd": 12.34, "limit_usd": 20.0,
                 "used_percentage": 38, "label": "CREDIT"}}

Design constraints (mirroring the user's shell script):
  * Never block the render. We serve the cached value and refresh in a daemon
    thread when the cache is stale, so the status feed pays zero network latency.
  * Fail open. Any error → return None (renderer falls back to "NO RATE DATA").
  * Gated. Only probes providers we have a real endpoint for (currently
    OpenRouter's /api/v1/key). Everything else returns None — no fabrication.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from typing import Any, Optional

_CACHE_TTL = 300.0  # 5 minutes, matching the Claude Code statusline cache
_TIMEOUT = 3.0

# keyed by (provider, api_key_tail) → (fetched_at, payload_or_None)
_cache: dict[tuple[str, str], tuple[float, Optional[dict[str, Any]]]] = {}
_inflight: set[tuple[str, str]] = set()
_lock = threading.Lock()


def _openrouter_key_balance(api_key: str) -> Optional[dict[str, Any]]:
    """Fetch OpenRouter key usage/limit via GET /api/v1/key.

    Response shape (relevant fields):
      {"data": {"usage": <float $ spent>, "limit": <float $ cap | null>,
                "limit_remaining": <float $ | null>}}
    A null limit means "no cap" (pay-as-you-go) — we still surface spent as a
    bare balance line rather than a gauge.
    """
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    data = raw.get("data") or {}
    usage = data.get("usage")
    limit = data.get("limit")
    remaining = data.get("limit_remaining")
    if usage is None and remaining is None:
        return None

    credits: dict[str, Any] = {"label": "CREDIT"}
    if remaining is not None:
        credits["remaining_usd"] = float(remaining)
    if limit is not None:
        credits["limit_usd"] = float(limit)
        if float(limit) > 0 and usage is not None:
            credits["used_percentage"] = max(0, min(100, round(float(usage) / float(limit) * 100)))
    return {"source": "openrouter", "credits": credits}


def _fetch(provider: str, api_key: str) -> Optional[dict[str, Any]]:
    try:
        if provider == "openrouter":
            return _openrouter_key_balance(api_key)
    except Exception:
        return None
    return None


def _refresh_async(key: tuple[str, str], provider: str, api_key: str) -> None:
    def _run() -> None:
        payload = _fetch(provider, api_key)
        with _lock:
            _cache[key] = (time.time(), payload)
            _inflight.discard(key)

    threading.Thread(target=_run, daemon=True).start()


def get_rate_limits(provider: str, api_key: str) -> Optional[dict[str, Any]]:
    """Return the cached rate/credit dict, refreshing in the background if stale.

    Non-blocking: on a cache miss the very first call returns None (no data yet)
    and kicks off a background fetch; subsequent calls serve the cached value.
    """
    provider = (provider or "").strip().lower()
    api_key = api_key or ""
    if not provider or not api_key or provider != "openrouter":
        return None

    key = (provider, api_key[-8:])
    now = time.time()
    with _lock:
        entry = _cache.get(key)
        stale = entry is None or (now - entry[0]) >= _CACHE_TTL
        if stale and key not in _inflight:
            _inflight.add(key)
            _refresh_async(key, provider, api_key)
        return entry[1] if entry else None

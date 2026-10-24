"""Prompt-cache retention window: when this session's cached prompt prefix expires.

Explicit-TTL routes (Anthropic-family ``cache_control`` markers) keep the cached prefix for
the marker's TTL from the last request that read or wrote it. Past that point the next
message re-processes the whole prompt at the cache-write rate, which is exactly what a
person returning to a long session wants to know before typing. Routes with automatic,
unpublished retention (OpenAI, DeepSeek, implicit Gemini) report nothing rather than a guess.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from agent.i18n import t

_TTL_SECONDS = {"5m": 300, "1h": 3600}


def stamp_prompt_cache_activity(agent: Any, usage: Any, *, model: str, provider: str) -> None:
    """Record that the provider just read or wrote this session's cached prefix.

    ``model``/``provider`` are the request's real destination (the MoA aggregator slot, not
    the virtual preset), so the TTL is clamped for the route that holds the cache. The TTL
    restarts when the successful attempt is sent (``_api_attempt_started_at``); retries and
    their waits before it never shorten the countdown. The display route is kept too, so a
    ``/model`` switch or fallback (a different cache) never inherits the stamp.
    """
    if int(getattr(usage, "cache_read_tokens", 0) or 0) + int(getattr(usage, "cache_write_tokens", 0) or 0) <= 0:
        return
    from agent.prompt_caching import effective_cache_ttl

    ttl = effective_cache_ttl(getattr(agent, "_cache_ttl", None), model=model or "", provider=provider or "")
    agent._prompt_cache_stamp = {
        "refreshed_at": float(getattr(agent, "_api_attempt_started_at", None) or time.time()),
        "ttl": ttl,
        "route": (getattr(agent, "model", "") or "", getattr(agent, "provider", "") or ""),
    }


def prompt_cache_window(agent: Any) -> Optional[dict[str, Any]]:
    """``{"ttl", "ttl_seconds", "refreshed_at", "expires_at"}`` (epoch seconds) for the live
    route's cached prefix, or None when nothing is cached or the retention is unknown."""
    stamp = getattr(agent, "_prompt_cache_stamp", None)
    if not isinstance(stamp, dict) or not getattr(agent, "_use_prompt_caching", False):
        return None
    if stamp.get("route") != (getattr(agent, "model", "") or "", getattr(agent, "provider", "") or ""):
        return None
    seconds = _TTL_SECONDS.get(stamp.get("ttl"))
    if not seconds:
        return None
    refreshed_at = float(stamp["refreshed_at"])
    return {"ttl": stamp["ttl"], "ttl_seconds": seconds, "refreshed_at": refreshed_at, "expires_at": refreshed_at + seconds}


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def render_prompt_cache_line(window: Optional[dict[str, Any]], *, now: Optional[float] = None) -> Optional[str]:
    """One ``/context`` line: warm with the time left, or expired and how long ago."""
    if not isinstance(window, dict) or not window.get("expires_at"):
        return None
    remaining = float(window["expires_at"]) - (time.time() if now is None else now)
    if remaining > 0:
        return t("gateway.context.cache_warm", remaining=_duration(remaining), ttl=window.get("ttl", ""))
    return t("gateway.context.cache_expired", ago=_duration(-remaining), ttl=window.get("ttl", ""))

"""Compact an idle chat while its prompt cache is still warm, just before the window closes.

A person who steps away for an hour comes back to a cold cache: the next message re-writes the
whole context at the cache-write rate (2x input on Anthropic's 1h tier). Compacting at ~0.9 of the
window instead means that return pays only for the small compacted context. Idea credited to the
community ``idle-compact`` plugin (#151785); this is the core version, run through the normal
compression pipeline and recorded like any other automatic compaction.

Scope (by design):

- Only on a 1-hour (or longer) guaranteed window. A 5-minute window closes during an ordinary
  pause, so compacting at 4.5 minutes would summarise conversations mid-thought. Nous Portal is
  capped at 5m today (its upstream drops 1h entries), so it never fires there; lifting the cap
  (``prompt_caching.route_ttl_ceilings``) turns it on with no code change.
- Only while the window is still guaranteed warm (``proven``/``declared``); unknown or
  estimate-only windows never fire.
- Only for contexts of at least :data:`MIN_CONTEXT_TOKENS`, and above the compressor's own
  post-compaction floor, so the compaction has something to reclaim.
- Only for human-paced sessions: subagents and machine-paced sources (cron, webhooks, one-shots,
  API server, ...) never fire. The owning surface also skips a session whose turn is running.

A committed compaction bumps the session's context generation, which makes the clock's window
``unknown`` (``compacted``), so one window fires at most once.
"""

from __future__ import annotations

import time
from typing import Any, Optional

#: Fire at this fraction of the cache window (54 minutes on a 1-hour tier).
FIRE_FRACTION = 0.9
#: Shortest window worth compacting before: below an hour, ordinary pauses outlast the window.
MIN_WINDOW_SECONDS = 3600
#: Smallest context (real prompt tokens of the last request) worth a compaction.
MIN_CONTEXT_TOKENS = 100_000
#: How often the surface owners look (the fire range is the last 10% of the window: 6 minutes).
CHECK_INTERVAL_SECONDS = 30.0
#: Telemetry trigger name for these compactions.
TRIGGER = "cache_expiry"


def _context_tokens(agent: Any) -> int:
    tokens = getattr(getattr(agent, "context_compressor", None), "last_prompt_tokens", 0)
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else 0


def _human_paced(agent: Any) -> bool:
    from agent.prompt_caching import MACHINE_PACED_SOURCES
    from agent.session_source import session_source_for

    if int(getattr(agent, "_delegate_depth", 0) or 0) > 0 or getattr(agent, "side_agent", False) is True:
        return False
    return session_source_for(getattr(agent, "platform", None)) not in MACHINE_PACED_SOURCES


def _compressor_blocked(agent: Any) -> Optional[str]:
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return "no_compressor"
    if getattr(compressor, "awaiting_real_usage_after_compression", False) is True:
        return "awaiting_usage"
    cooldown = getattr(compressor, "get_active_compression_failure_cooldown", None)
    if callable(cooldown) and cooldown():
        return "cooldown"
    return None


def _floor_tokens(agent: Any) -> int:
    compressor = agent.context_compressor
    try:
        return int(compressor.threshold_tokens * compressor.summary_target_ratio)
    except (AttributeError, TypeError, ValueError):
        return 0


def _window_due(agent: Any, now: float) -> tuple[bool, str]:
    from agent.prompt_cache_clock import DECLARED, PROVEN, cache_certainly_warm, cache_window
    from agent.prompt_caching import TTL_SECONDS

    window = cache_window(agent)
    if window is None:
        return False, "no_cache"
    if window.confidence not in (PROVEN, DECLARED) or window.guaranteed_until is None:
        return False, f"window_{window.confidence}:{window.reason}"
    tier_seconds = TTL_SECONDS.get(window.tier or "", 0)
    if tier_seconds < MIN_WINDOW_SECONDS:
        return False, f"short_window:{window.tier}"
    if not cache_certainly_warm(window, now=now):
        return False, "already_cold"
    fire_at = float(window.guaranteed_until) - (1.0 - FIRE_FRACTION) * tier_seconds
    return (now >= fire_at), ("due" if now >= fire_at else "not_yet")


def cache_expiry_compaction_due(agent: Any, *, now: Optional[float] = None) -> tuple[bool, str]:
    """``(due, reason)``: whether *agent*'s idle session should compact now, before its cache
    window closes. Cheap gates run first, so an active session never reads the clock's store."""
    now = time.time() if now is None else now
    if agent is None or not getattr(agent, "compression_enabled", False):
        return False, "disabled"
    if getattr(agent, "api_mode", None) == "codex_app_server":
        return False, "codex_app_server"
    if now - float(getattr(agent, "_last_activity_ts", now) or now) < FIRE_FRACTION * MIN_WINDOW_SECONDS:
        return False, "recently_active"
    if not _human_paced(agent):
        return False, "machine_paced"
    tokens = _context_tokens(agent)
    if tokens < max(MIN_CONTEXT_TOKENS, _floor_tokens(agent) + 1):
        return False, "small_context"
    if blocked := _compressor_blocked(agent):
        return False, blocked
    return _window_due(agent, now)

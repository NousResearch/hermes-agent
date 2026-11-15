"""Prompt-cache clock: when this session's cached prompt prefix expires, per route.

One record per successful attempt that read or wrote the cache: the tier the request actually
asked for (read from the payload sent, not from config), its send time, and the tier the
response proved (Anthropic ``ephemeral_5m/1h_input_tokens``, OpenAI Responses
``ttl_30m/12h_tokens``). Every consumer (``/context`` today; keep-alive, cold-cache pruning and
idle compaction later) reads :func:`cache_window` instead of re-deriving a TTL.

Refresh rule (Anthropic, OpenAI 5.6+, Kimi and Qwen explicit all document it): a read restarts
the entry's lifetime with the tier it was WRITTEN at, so ``expires = latest touch + tier``. The
clock returns facts plus a confidence; consumers apply their own safety margins.

Confidence: ``proven`` (the response named the tier), ``declared`` (what we sent, on a route
known to honour it), ``assumed`` (a typical figure, no guarantee), ``unknown`` (no published
retention, or the cache moved: route switch, fallback, compaction).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Optional

from agent.prompt_caching import TTL_SECONDS, route_ttl_ceiling
from utils import base_url_host_matches, base_url_hostname

PROVEN, DECLARED, ASSUMED, UNKNOWN = "proven", "declared", "assumed", "unknown"
_OPENAI_HOSTS = ("api.openai.com", "chatgpt.com")
_GPT_VERSION = re.compile(r"gpt-(\d+)(?:\.(\d+))?")


@dataclass(frozen=True)
class CacheEvent:
    """The latest successful attempt that touched the cache (only the latest matters: the tier
    is fixed at write and a read refreshes with that tier)."""

    route: tuple[str, str, str]  # (provider, base_url host, model) the agent displayed when sent
    route_class: str
    requested_ttl: Optional[str]
    sent_at: float
    read_tokens: int
    write_tokens: int
    tier: Optional[str]
    confidence: str
    reason: str
    session_id: str = ""
    compression_count: int = 0


@dataclass(frozen=True)
class CacheWindow:
    expires_at: Optional[float]  # best estimate of expiry; None when unknown
    floor_at: Optional[float]  # guaranteed-warm-until (documented or proven minimum); None if no guarantee
    tier: Optional[str]
    confidence: str
    reason: str
    refreshed_at: Optional[float] = None


@dataclass(frozen=True)
class _Ctx:
    route_class: str
    requested_ttl: Optional[str]
    retention: Optional[str]
    write_tiers: dict[str, int]
    write_tokens: int
    ceiling: Optional[str]


# ── Request side ────────────────────────────────────────────────────────────


def _block_marker_ttls(block: Any, out: set[str]) -> None:
    if not isinstance(block, dict):
        return
    content = block.get("content")
    for part in (block, *(content if isinstance(content, list) else ())):
        marker = part.get("cache_control") if isinstance(part, dict) else None
        if isinstance(marker, dict):
            out.add(marker.get("ttl") or "5m")


def _list_field(api_kwargs: dict, key: str) -> list:
    value = api_kwargs.get(key)
    return value if isinstance(value, list) else []


def _marker_ttl(api_kwargs: dict) -> Optional[str]:
    """Shortest ``cache_control`` TTL on the wire (the planner puts breakpoints on the system
    prompt, the tools tail and the last few messages), or None when the request has no marker."""
    found: set[str] = set()
    messages, tools = _list_field(api_kwargs, "messages"), _list_field(api_kwargs, "tools")
    for block in (*_list_field(api_kwargs, "system"), *messages[:2], *messages[-4:], *tools[-1:]):
        _block_marker_ttls(block, found)
    return min(found, key=lambda ttl: TTL_SECONDS.get(ttl, 0)) if found else None


def _wire_field(api_kwargs: dict, key: str) -> Any:
    extra = api_kwargs.get("extra_body")
    return api_kwargs.get(key) if key in api_kwargs else (extra.get(key) if isinstance(extra, dict) else None)


def record_attempt_sent(agent: Any, api_kwargs: Any) -> None:
    """Stamp THIS attempt's send time and the tier it asks for (retries never shorten the window)."""
    if not isinstance(api_kwargs, dict):
        return
    options = _wire_field(api_kwargs, "prompt_cache_options")
    agent._prompt_cache_attempt = {
        "sent_at": time.time(),
        "marker_ttl": _marker_ttl(api_kwargs),
        "openai_ttl": options.get("ttl") if isinstance(options, dict) else None,
        "retention": _wire_field(api_kwargs, "prompt_cache_retention"),
    }


# ── Route classes and rules ─────────────────────────────────────────────────


def _is_gpt_56_plus(model: str) -> bool:
    match = _GPT_VERSION.search((model or "").lower())
    return bool(match) and (int(match.group(1)), int(match.group(2) or 0)) >= (5, 6)


def classify_route(*, provider: str, base_url: str, api_mode: str, model: str,
                   has_markers: bool, retention: Optional[str]) -> str:
    """Which rule set governs the cache this request reached (ordered: first match wins)."""
    host = base_url_hostname(base_url)
    openai_host = any(base_url_host_matches(base_url, h) for h in _OPENAI_HOSTS)
    checks: tuple[tuple[bool, str], ...] = (
        (has_markers and route_ttl_ceiling(provider, base_url) is not None, "route_ceiling"),
        (has_markers and api_mode == "anthropic_messages" and (provider == "anthropic" or host == "api.anthropic.com"),
         "anthropic_native"),
        (has_markers and base_url_host_matches(base_url, "openrouter.ai") and "claude" in model.lower(), "openrouter_claude"),
        (has_markers, "anthropic_gateway"),
        (openai_host and _is_gpt_56_plus(model), "openai_ttl"),
        (openai_host or retention is not None, "openai_retention"),
    )
    return next((name for matched, name in checks if matched), "implicit")


def _largest_tier(tiers: dict[str, int]) -> Optional[str]:
    written = [tier for tier, tokens in tiers.items() if tokens > 0 and tier in TTL_SECONDS]
    return max(written, key=TTL_SECONDS.__getitem__) if written else None


def _rule_anthropic_native(ctx: _Ctx) -> tuple[Optional[str], str, str]:
    if proven := _largest_tier(ctx.write_tiers):
        return proven, PROVEN, "anthropic_usage_split"
    return ctx.requested_ttl, DECLARED, "anthropic_requested"


def _rule_openrouter_claude(ctx: _Ctx) -> tuple[Optional[str], str, str]:
    # Live probe: OpenRouter's Anthropic/Bedrock/Vertex Claude routes honour ttl 1h (write billed 2x).
    return ctx.requested_ttl, DECLARED, "openrouter_honours_ttl"


def _rule_route_ceiling(ctx: _Ctx) -> tuple[Optional[str], str, str]:
    requested, ceiling = ctx.requested_ttl or "5m", ctx.ceiling or "5m"
    tier = min((requested, ceiling), key=lambda ttl: TTL_SECONDS.get(ttl, 0))
    return tier, DECLARED, f"route_ceiling_{ceiling}"


def _rule_anthropic_gateway(ctx: _Ctx) -> tuple[Optional[str], str, str]:
    if ctx.requested_ttl == "5m":
        return "5m", DECLARED, "gateway_requested_5m"
    return "5m", ASSUMED, "gateway_1h_untrusted"


def _rule_openai_ttl(ctx: _Ctx) -> tuple[Optional[str], str, str]:
    if proven := _largest_tier(ctx.write_tiers):
        return proven, PROVEN, "openai_write_tier"
    return ctx.requested_ttl or "30m", DECLARED, "openai_min_30m"


def _rule_openai_retention(ctx: _Ctx) -> tuple[Optional[str], str, str]:
    if ctx.retention == "in_memory":
        return "5m", ASSUMED, "openai_in_memory"
    return "30m", ASSUMED, "openai_24h_typical_30m"


def _rule_implicit(ctx: _Ctx) -> tuple[Optional[str], str, str]:
    return None, UNKNOWN, "implicit_cache"


_RULES: dict[str, Callable[[_Ctx], tuple[Optional[str], str, str]]] = {
    "route_ceiling": _rule_route_ceiling,
    "anthropic_native": _rule_anthropic_native,
    "openrouter_claude": _rule_openrouter_claude,
    "anthropic_gateway": _rule_anthropic_gateway,
    "openai_ttl": _rule_openai_ttl,
    "openai_retention": _rule_openai_retention,
    "implicit": _rule_implicit,
}


# ── Response side ───────────────────────────────────────────────────────────


def _display_route(agent: Any) -> tuple[str, str, str]:
    return (str(getattr(agent, "provider", "") or ""), base_url_hostname(getattr(agent, "base_url", "") or ""),
            str(getattr(agent, "model", "") or ""))


def _compression_count(agent: Any) -> int:
    return int(getattr(getattr(agent, "context_compressor", None), "compression_count", 0) or 0)


def _requested_ttl(agent: Any, attempt: dict, *, provider: str, base_url: str, model: str) -> Optional[str]:
    if attempt.get("marker_ttl") or attempt.get("openai_ttl"):
        return attempt.get("marker_ttl") or attempt.get("openai_ttl")
    if (getattr(agent, "provider", "") or "") == "moa" and getattr(agent, "_use_prompt_caching", False):
        # MoA decorates each slot inside its client, after this payload was stamped.
        from agent.prompt_caching import effective_cache_ttl
        return effective_cache_ttl(getattr(agent, "_cache_ttl", None), provider=provider, model=model, base_url=base_url)
    return None


def record_cache_response(agent: Any, usage: Any, *, model: str, provider: str, base_url: str,
                          api_mode: str) -> Optional[CacheEvent]:
    """Fold one response into the clock. ``model``/``provider``/``base_url`` are the request's real
    destination (the MoA aggregator slot, not the virtual preset). Returns the new event, or None
    when the response touched no cache (the previous window stands)."""
    read, write = int(getattr(usage, "cache_read_tokens", 0) or 0), int(getattr(usage, "cache_write_tokens", 0) or 0)
    if read + write <= 0:
        return None
    attempt = getattr(agent, "_prompt_cache_attempt", None) or {}
    requested = _requested_ttl(agent, attempt, provider=provider, base_url=base_url, model=model)
    ctx = _Ctx(
        route_class=classify_route(provider=provider, base_url=base_url, api_mode=api_mode or "", model=model or "",
                                   has_markers=requested is not None and not attempt.get("openai_ttl"),
                                   retention=attempt.get("retention")),
        requested_ttl=requested, retention=attempt.get("retention"),
        write_tiers=dict(getattr(usage, "cache_write_by_ttl", ()) or ()), write_tokens=write,
        ceiling=route_ttl_ceiling(provider, base_url),
    )
    tier, confidence, reason = _RULES[ctx.route_class](ctx)
    previous = getattr(agent, "_prompt_cache_clock", None)
    route = _display_route(agent)
    if write == 0 and isinstance(previous, CacheEvent) and previous.route == route and previous.confidence == PROVEN:
        # A pure read proves the entry was alive; its tier is still the one the write proved.
        tier, confidence, reason = previous.tier, PROVEN, previous.reason
    event = CacheEvent(
        route=route, route_class=ctx.route_class, requested_ttl=requested,
        sent_at=float(attempt.get("sent_at") or time.time()), read_tokens=read, write_tokens=write,
        tier=tier, confidence=confidence, reason=reason,
        session_id=str(getattr(agent, "session_id", "") or ""), compression_count=_compression_count(agent),
    )
    agent._prompt_cache_clock = event
    return event


# ── Persistence ─────────────────────────────────────────────────────────────


def event_to_json(event: CacheEvent) -> str:
    return json.dumps(asdict(event), separators=(",", ":"))


def event_from_json(raw: Any) -> Optional[CacheEvent]:
    try:
        data: Any = json.loads(raw) if isinstance(raw, str) else None
        return CacheEvent(**{**data, "route": tuple(data["route"])}) if isinstance(data, dict) else None
    except (TypeError, ValueError, KeyError):  # a column written by a different build: no clock
        return None


def _fallback_event(agent: Any, last_assistant_at: float) -> Optional[CacheEvent]:
    """No stored clock (a session from before this column): anchor on the newest assistant row and
    apply the current route's rule with no response evidence, never better than ``assumed``."""
    from agent.prompt_caching import effective_cache_ttl
    provider, base_url = str(getattr(agent, "provider", "") or ""), str(getattr(agent, "base_url", "") or "")
    model = str(getattr(agent, "model", "") or "")
    caching = bool(getattr(agent, "_use_prompt_caching", False))
    requested = effective_cache_ttl(getattr(agent, "_cache_ttl", None), provider=provider, model=model,
                                    base_url=base_url) if caching else None
    route_class = classify_route(provider=provider, base_url=base_url, api_mode=str(getattr(agent, "api_mode", "") or ""),
                                 model=model, has_markers=caching, retention=None)
    ctx = _Ctx(route_class, requested, None, {}, 0, route_ttl_ceiling(provider, base_url))
    tier, confidence, reason = _RULES[route_class](ctx)
    return CacheEvent(
        route=_display_route(agent), route_class=route_class, requested_ttl=requested, sent_at=last_assistant_at,
        read_tokens=0, write_tokens=0, tier=tier, confidence=UNKNOWN if confidence == UNKNOWN else ASSUMED,
        reason=f"restored_from_transcript:{reason}", session_id=str(getattr(agent, "session_id", "") or ""),
        compression_count=_compression_count(agent),
    )


def restore_cache_clock(agent: Any) -> Optional[CacheEvent]:
    """Load the session's stored clock once per agent (resume, gateway re-creation). The stored
    compression count is re-based to this process's compressor, which starts from zero."""
    agent._prompt_cache_clock_restored = True
    db, session_id = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    reader = getattr(db, "get_prompt_cache_clock", None)
    if not session_id or not callable(reader):
        return None
    stored: Any = reader(session_id) or {}
    event = event_from_json(stored.get("clock"))
    if event is not None:
        event = replace(event, session_id=str(session_id), compression_count=_compression_count(agent))
    elif isinstance(stored.get("last_assistant_at"), (int, float)):
        event = _fallback_event(agent, float(stored["last_assistant_at"]))
    agent._prompt_cache_clock = event
    return event


# ── The answer every consumer reads ─────────────────────────────────────────


def _unknown(reason: str, event: Optional[CacheEvent] = None) -> CacheWindow:
    return CacheWindow(None, None, None, UNKNOWN, reason, event.sent_at if event else None)


def cache_window(agent: Any) -> Optional[CacheWindow]:
    """The live route's cache window, or None when this session never touched a cache."""
    event = getattr(agent, "_prompt_cache_clock", None)
    if event is None and not getattr(agent, "_prompt_cache_clock_restored", False):
        event = restore_cache_clock(agent)
    if not isinstance(event, CacheEvent):
        return None
    if event.session_id and event.session_id != str(getattr(agent, "session_id", "") or ""):
        return _unknown("session_rotated", event)
    if event.compression_count != _compression_count(agent):
        return _unknown("compacted", event)
    if event.route != _display_route(agent):
        return _unknown("route_changed", event)
    seconds = TTL_SECONDS.get(event.tier or "")
    if event.confidence == UNKNOWN or not seconds:
        return _unknown(event.reason, event)
    expires_at = event.sent_at + seconds
    floor_at = expires_at if event.confidence in (PROVEN, DECLARED) else None
    return CacheWindow(expires_at, floor_at, event.tier, event.confidence, event.reason, event.sent_at)


def cache_window_payload(agent: Any) -> Optional[dict[str, Any]]:
    """``cache_window`` as the JSON shape surfaces carry (``PromptCacheWindow`` contract)."""
    window = cache_window(agent)
    return asdict(window) if window is not None else None


# ── /context line (surface absorbed from #142387) ──────────────────────────

_MOVED_REASONS = frozenset({"session_rotated", "compacted", "route_changed"})


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
    """One ``/context`` line: warm until a wall-clock time, expired and how long ago, or unknown."""
    from agent.i18n import t

    if not isinstance(window, dict):
        return None
    expires_at, confidence = window.get("expires_at"), window.get("confidence")
    if not expires_at or confidence == UNKNOWN:
        moved = window.get("reason") in _MOVED_REASONS
        return t("gateway.context.cache_unknown_moved" if moved else "gateway.context.cache_unknown_route")
    label = t(f"gateway.context.cache_confidence.{confidence}")
    remaining = float(expires_at) - (time.time() if now is None else now)
    if remaining > 0:
        until = time.strftime("%H:%M", time.localtime(float(expires_at)))
        return t("gateway.context.cache_warm", until=until, remaining=_duration(remaining),
                 ttl=window.get("tier") or "", confidence=label)
    return t("gateway.context.cache_expired", ago=_duration(-remaining), ttl=window.get("tier") or "", confidence=label)


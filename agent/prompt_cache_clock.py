"""Prompt-cache clock: until when this session's cached prompt prefix stays warm, per route.

One record per successful attempt that read or wrote the cache: the tier the request actually
asked for (read from the payload sent, not from config), its send time, the tier(s) the response
proved (Anthropic ``ephemeral_5m/1h_input_tokens``, OpenAI Responses ``ttl_30m/12h_tokens``), and
the resulting COVERAGE: one deadline per tier the cached prefix holds. Every consumer (``/context``
today; keep-alive, cold-cache pruning and idle compaction later) reads :func:`cache_window` (or
:func:`stored_cache_window` with no live agent) instead of re-deriving a TTL.

Refresh rules, per route class:

- Anthropic family (native, OpenRouter Claude, route ceilings, other Anthropic-wire gateways): the
  tier is fixed at write and a read restarts it, so ``deadline = read + tier``.
- OpenAI GPT-5.6+ (``prompt_cache_options.ttl``): a read refreshes coverage to at least 30 minutes
  after it, ``deadline = max(existing, read + 30m)``. A 12h deadline is fixed at ``write + 12h``: a
  read never restarts it, it only extends the 30-minute floor.
- Everything else has no published guarantee (OpenAI before 5.6 ``prompt_cache_retention``, an
  Anthropic-wire gateway asked for 1h, implicit caches): no coverage, at most an estimated end.

Two ends per window, because a documented minimum is not an expiry:

- ``guaranteed_until``: the EARLIEST coverage deadline, so the whole cached prefix is warm before
  it (a response that wrote two tiers is only as warm as its shorter one). ``None`` = no guarantee.
- ``estimated_until``: after it the entry is certainly gone. Anthropic tiers are exact, so it is the
  latest coverage deadline; OpenAI "may retain entries longer than the requested minimum" and keeps
  cache state at most 24 hours, so it is the last touch + 24h (``in_memory``: up to 1h).

Consumers ask :func:`cache_certainly_warm` (before ``guaranteed_until``) or
:func:`cache_certainly_cold` (past ``estimated_until``; never past only a minimum). Between the two
the cache MAY still be warm. An unknown window is neither.

Storage is advisory here: a failed read (SQLite, a worker's owner RPC) makes the window
``unknown`` (``storage_unavailable``) and never raises into the turn that produced a reply.

Validity: an event is bound to the session id, the full route (provider, normalized base URL with
scheme, host, port and path, model) and the session's durable context generation (how many rows a
compaction or prune archived: it survives restarts and only grows). Any difference makes the window
``unknown``. A resumed agent restores and validates the saved event BEFORE folding its first
response, so a first pure read keeps the saved proof.

Confidence: ``proven`` (the response named the tier), ``declared`` (what we sent, on a route known
to honour it), ``assumed`` (no guarantee), ``unknown`` (no published retention, or the cache moved:
route switch, fallback, compaction, or a runtime whose requests Hermes does not build).
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Optional

from agent.prompt_caching import TTL_SECONDS, route_ttl_ceiling
from utils import base_url_host_matches, base_url_hostname

logger = logging.getLogger(__name__)

PROVEN, DECLARED, ASSUMED, UNKNOWN = "proven", "declared", "assumed", "unknown"
STORAGE_UNAVAILABLE = -1  # a generation read that failed: never equal to a real (>= 0) generation
_CONFIDENCE_RANK = {UNKNOWN: 0, ASSUMED: 1, DECLARED: 2, PROVEN: 3}
_OPENAI_HOSTS = ("api.openai.com", "chatgpt.com")
_GPT_VERSION = re.compile(r"gpt-(\d+)(?:\.(\d+))?")
_OPENAI_READ_FLOOR = "30m"
_OPENAI_MAX_RETENTION_SECONDS = 86_400
_IN_MEMORY_MAX_SECONDS = 3_600


@dataclass(frozen=True)
class CacheEvent:
    """The cache state after the latest successful attempt that touched it."""

    route: tuple[str, str, str]  # (provider, normalized base_url, model) the agent displayed when sent
    destination: tuple[str, str, str]  # where the request went (the MoA aggregator slot for MoA)
    route_class: str
    requested_ttl: Optional[str]
    sent_at: float
    read_tokens: int
    write_tokens: int
    tier: Optional[str]  # tier of the earliest deadline (or the assumed figure when nothing is guaranteed)
    confidence: str
    reason: str
    coverage: dict[str, float] = field(default_factory=dict)  # tier -> deadline, every tier still covered
    guaranteed_until: Optional[float] = None
    estimated_until: Optional[float] = None
    session_id: str = ""
    generation: Optional[int] = None  # SessionDB.prompt_cache_generation when recorded


@dataclass(frozen=True)
class CacheWindow:
    guaranteed_until: Optional[float]  # the whole prefix is warm before this; None = no guarantee
    estimated_until: Optional[float]  # certainly gone after this; None = unknown
    tier: Optional[str]
    confidence: str
    reason: str
    refreshed_at: Optional[float] = None
    coverage: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class _Ctx:
    route_class: str
    requested_ttl: Optional[str]
    retention: Optional[str]
    write_tiers: dict[str, int]
    write_tokens: int
    ceiling: Optional[str]


@dataclass(frozen=True)
class _Verdict:
    tiers: tuple[str, ...]  # tiers this response guarantees for what it wrote (or read, with no prior state)
    confidence: str
    reason: str
    tier: Optional[str] = None  # display tier when ``tiers`` is empty
    estimate_seconds: Optional[int] = None  # longest the entry can live past the touch, beyond its tiers


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


def attempt_facts(api_kwargs: Any) -> Optional[dict[str, Any]]:
    """What one request payload asks of the cache, stamped with its send time."""
    if not isinstance(api_kwargs, dict):
        return None
    options = _wire_field(api_kwargs, "prompt_cache_options")
    return {
        "sent_at": time.time(),
        "marker_ttl": _marker_ttl(api_kwargs),
        "openai_ttl": options.get("ttl") if isinstance(options, dict) else None,
        "retention": _wire_field(api_kwargs, "prompt_cache_retention"),
    }


def record_attempt_sent(agent: Any, api_kwargs: Any) -> None:
    """Stamp THIS attempt's send time and the tier it asks for (retries never shorten the window)."""
    facts = attempt_facts(api_kwargs)
    if facts is not None:
        agent._prompt_cache_attempt = facts


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


def _written_tiers(tiers: dict[str, int]) -> tuple[str, ...]:
    return tuple(sorted((t for t, n in tiers.items() if n > 0 and t in TTL_SECONDS), key=TTL_SECONDS.__getitem__))


def _rule_anthropic_native(ctx: _Ctx) -> _Verdict:
    if proven := _written_tiers(ctx.write_tiers):
        return _Verdict(proven, PROVEN, "anthropic_usage_split")
    return _Verdict((ctx.requested_ttl or "5m",), DECLARED, "anthropic_requested")


def _rule_openrouter_claude(ctx: _Ctx) -> _Verdict:
    # Live probe: OpenRouter's Anthropic/Bedrock/Vertex Claude routes honour ttl 1h (write billed 2x).
    return _Verdict((ctx.requested_ttl or "5m",), DECLARED, "openrouter_honours_ttl")


def _rule_route_ceiling(ctx: _Ctx) -> _Verdict:
    requested, ceiling = ctx.requested_ttl or "5m", ctx.ceiling or "5m"
    tier = min((requested, ceiling), key=lambda ttl: TTL_SECONDS.get(ttl, 0))
    return _Verdict((tier,), DECLARED, f"route_ceiling_{ceiling}")


def _rule_anthropic_gateway(ctx: _Ctx) -> _Verdict:
    if (ctx.requested_ttl or "5m") == "5m":
        return _Verdict(("5m",), DECLARED, "gateway_requested_5m")
    # A gateway that may keep only 5m: nothing is guaranteed; gone after the tier asked for.
    return _Verdict((), ASSUMED, "gateway_1h_untrusted", tier=ctx.requested_ttl,
                    estimate_seconds=TTL_SECONDS.get(ctx.requested_ttl or "", 300))


def _rule_openai_ttl(ctx: _Ctx) -> _Verdict:
    if proven := _written_tiers(ctx.write_tiers):
        return _Verdict(proven, PROVEN, "openai_write_tier", estimate_seconds=_OPENAI_MAX_RETENTION_SECONDS)
    if ctx.write_tokens <= 0:  # a read guarantees at least 30 minutes after it, whatever was asked
        return _Verdict((_OPENAI_READ_FLOOR,), DECLARED, "openai_read_min_30m",
                        estimate_seconds=_OPENAI_MAX_RETENTION_SECONDS)
    return _Verdict((ctx.requested_ttl or _OPENAI_READ_FLOOR,), DECLARED, "openai_min_lifetime",
                    estimate_seconds=_OPENAI_MAX_RETENTION_SECONDS)


def _rule_openai_retention(ctx: _Ctx) -> _Verdict:
    if ctx.retention == "in_memory":
        return _Verdict((), ASSUMED, "openai_in_memory", tier="in_memory", estimate_seconds=_IN_MEMORY_MAX_SECONDS)
    return _Verdict((), ASSUMED, "openai_24h_retention", tier="24h", estimate_seconds=_OPENAI_MAX_RETENTION_SECONDS)


def _rule_implicit(ctx: _Ctx) -> _Verdict:
    return _Verdict((), UNKNOWN, "implicit_cache")


_RULES: dict[str, Callable[[_Ctx], _Verdict]] = {
    "route_ceiling": _rule_route_ceiling,
    "anthropic_native": _rule_anthropic_native,
    "openrouter_claude": _rule_openrouter_claude,
    "anthropic_gateway": _rule_anthropic_gateway,
    "openai_ttl": _rule_openai_ttl,
    "openai_retention": _rule_openai_retention,
    "implicit": _rule_implicit,
}


def _refresh_restart(deadline: float, tier: str, at: float) -> float:
    return at + TTL_SECONDS[tier]


def _refresh_openai(deadline: float, tier: str, at: float) -> float:
    return max(deadline, at + TTL_SECONDS[_OPENAI_READ_FLOOR])


# How a read moves a covered tier's deadline; route classes absent here never hold coverage.
_REFRESH: dict[str, Callable[[float, str, float], float]] = {
    "route_ceiling": _refresh_restart, "anthropic_native": _refresh_restart, "openrouter_claude": _refresh_restart,
    "anthropic_gateway": _refresh_restart, "openai_ttl": _refresh_openai,
}


def _fold_coverage(previous: Optional[CacheEvent], verdict: _Verdict, route_class: str, at: float,
                   read: int, write: int) -> tuple[dict[str, float], bool]:
    """``(coverage after this response, whether any earlier coverage carried over)``. A read refreshes
    every still-live tier by the route's rule; a write (or a read with nothing carried) adds the
    tiers it guarantees. A tier present twice keeps the EARLIER deadline."""
    refresh = _REFRESH.get(route_class)
    if refresh is None:
        return {}, False
    coverage: dict[str, float] = {}
    if read > 0 and previous is not None:
        coverage = {tier: refresh(deadline, tier, at) for tier, deadline in previous.coverage.items()
                    if tier in TTL_SECONDS and deadline > at}
    carried = bool(coverage)
    if write > 0 or not carried:
        for tier in verdict.tiers:
            deadline = at + TTL_SECONDS[tier]
            coverage[tier] = min(coverage.get(tier, deadline), deadline)
    return coverage, carried


def _ends(coverage: dict[str, float], verdict: _Verdict, at: float) -> tuple[Optional[float], Optional[float]]:
    guaranteed = min(coverage.values()) if coverage else None
    latest = max(coverage.values()) if coverage else None
    if verdict.estimate_seconds is None:
        return guaranteed, latest
    return guaranteed, max(latest or 0.0, at + verdict.estimate_seconds)


def _confidence(verdict: _Verdict, previous: Optional[CacheEvent], carried: bool, write: int) -> tuple[str, str]:
    """A pure read keeps the carried proof (the entry it hit was written at that tier); a write on
    top of carried coverage is only as sure as the weaker of the two."""
    if not carried or previous is None:
        return verdict.confidence, verdict.reason
    if write == 0:
        return previous.confidence, previous.reason
    return min(verdict.confidence, previous.confidence, key=_CONFIDENCE_RANK.__getitem__), verdict.reason


# ── Identity ────────────────────────────────────────────────────────────────


def _route_key(provider: Any, base_url: Any, model: Any) -> tuple[str, str, str]:
    from hermes_cli.route_identity import normalize_route_base_url
    return (str(provider or ""), normalize_route_base_url(str(base_url or "")), str(model or ""))


def _display_route(agent: Any) -> tuple[str, str, str]:
    return _route_key(getattr(agent, "provider", ""), getattr(agent, "base_url", ""), getattr(agent, "model", ""))


def _compression_count(agent: Any) -> int:
    """This process's witness of context rewrites: compressions, plus the in-place archives
    (prune, micro-compaction) a worker store counts, which leave ``compression_count`` alone."""
    rewrites = getattr(getattr(agent, "_session_db", None), "context_rewrites", 0)
    return (int(getattr(getattr(agent, "context_compressor", None), "compression_count", 0) or 0)
            + (rewrites if isinstance(rewrites, int) else 0))


def _storage_errors() -> tuple[type[BaseException], ...]:
    """What a store read can raise: SQLite and the filesystem on the owner, the owner RPC in a worker."""
    from agent.runtime_session_store import WorkerPersistenceError
    return sqlite3.Error, OSError, WorkerPersistenceError


def current_generation(db: Any, session_id: Any) -> Optional[int]:
    """The session's durable context generation; None when the store has none (no session db),
    :data:`STORAGE_UNAVAILABLE` when the read failed."""
    reader = getattr(db, "prompt_cache_generation", None)
    if not session_id or not callable(reader):
        return None
    try:
        value = reader(str(session_id))
    except _storage_errors() as exc:
        logger.debug("prompt-cache clock: generation read failed for %s (%s); window unknown", session_id, exc)
        return STORAGE_UNAVAILABLE
    return value if isinstance(value, int) else None


def _invalid_reason(event: CacheEvent, *, session_id: str, generation: Optional[int],
                    route: Optional[tuple[str, str, str]]) -> Optional[str]:
    """Why *event* no longer describes the cache the next request reaches, or None."""
    if event.session_id and event.session_id != session_id:
        return "session_rotated"
    if STORAGE_UNAVAILABLE in (generation, event.generation):
        return "storage_unavailable"
    if generation is not None and event.generation != generation:
        return "compacted"
    if route is not None and event.route != route:
        return "route_changed"
    return None


def _agent_generation(agent: Any) -> Optional[int]:
    return current_generation(getattr(agent, "_session_db", None), getattr(agent, "session_id", None))


def _agent_invalid_reason(agent: Any, event: CacheEvent, generation: Optional[int]) -> Optional[str]:
    # No durable generation (a worker store): this process's compressor count is the only witness.
    if generation is None and getattr(agent, "_prompt_cache_compression_mark", 0) != _compression_count(agent):
        return "compacted"
    return _invalid_reason(event, session_id=str(getattr(agent, "session_id", "") or ""), generation=generation,
                           route=_display_route(agent))


def _set_event(agent: Any, event: Optional[CacheEvent]) -> None:
    agent._prompt_cache_clock = event
    agent._prompt_cache_compression_mark = _compression_count(agent)


# ── Response side ───────────────────────────────────────────────────────────


def _valid_previous(agent: Any, generation: Optional[int]) -> Optional[CacheEvent]:
    """The event this response folds onto: the saved one on a resumed agent (restored first), and
    only while it still describes the same cache."""
    if not getattr(agent, "_prompt_cache_clock_restored", False):
        restore_cache_clock(agent)
    previous = getattr(agent, "_prompt_cache_clock", None)
    if not isinstance(previous, CacheEvent) or previous.confidence == UNKNOWN:
        return None
    return None if _agent_invalid_reason(agent, previous, generation) else previous


def _moa_destination(agent: Any) -> tuple[Optional[dict], Optional[dict]]:
    """``(route, attempt)`` of the MoA aggregator request the client really sent, or ``(None, None)``."""
    client = getattr(agent, "client", None)
    route = getattr(client, "last_aggregator_route", None)
    if not isinstance(route, dict):
        return None, None
    attempt = getattr(client, "last_aggregator_attempt", None)
    return route, attempt if isinstance(attempt, dict) else None


def record_cache_response(agent: Any, usage: Any, *, model: str, provider: str, base_url: str,
                          api_mode: str) -> Optional[CacheEvent]:
    """Fold one response into the clock. ``model``/``provider``/``base_url``/``api_mode`` are the
    request's destination; a MoA client's resolved aggregator route and payload win over them.
    Returns the new event, or None when the response touched no cache (the previous window stands)."""
    read, write = int(getattr(usage, "cache_read_tokens", 0) or 0), int(getattr(usage, "cache_write_tokens", 0) or 0)
    if read + write <= 0:
        return None
    moa_route, moa_attempt = _moa_destination(agent)
    if moa_route is not None:
        model, provider = moa_route.get("model") or model, moa_route.get("provider") or provider
        base_url, api_mode = moa_route.get("base_url") or "", moa_route.get("api_mode") or ""
    generation = _agent_generation(agent)
    previous = _valid_previous(agent, generation)
    attempt = moa_attempt or getattr(agent, "_prompt_cache_attempt", None) or {}
    ctx = _Ctx(
        route_class=classify_route(provider=provider, base_url=base_url, api_mode=api_mode or "", model=model or "",
                                   has_markers=attempt.get("marker_ttl") is not None,
                                   retention=attempt.get("retention")),
        requested_ttl=attempt.get("marker_ttl") or attempt.get("openai_ttl"), retention=attempt.get("retention"),
        write_tiers=dict(getattr(usage, "cache_write_by_ttl", ()) or ()), write_tokens=write,
        ceiling=route_ttl_ceiling(provider, base_url),
    )
    verdict = _RULES[ctx.route_class](ctx)
    destination = _route_key(provider, base_url, model)
    if previous is not None and (previous.destination != destination or previous.route_class != ctx.route_class):
        previous = None
    at = float(attempt.get("sent_at") or time.time())
    coverage, carried = _fold_coverage(previous, verdict, ctx.route_class, at, read, write)
    guaranteed, estimated = _ends(coverage, verdict, at)
    confidence, reason = _confidence(verdict, previous, carried, write)
    event = CacheEvent(
        route=_display_route(agent), destination=destination, route_class=ctx.route_class,
        requested_ttl=ctx.requested_ttl, sent_at=at, read_tokens=read, write_tokens=write,
        tier=min(coverage, key=coverage.__getitem__) if coverage else verdict.tier,
        confidence=confidence if estimated is not None else UNKNOWN, reason=reason, coverage=coverage,
        guaranteed_until=guaranteed, estimated_until=estimated, session_id=str(getattr(agent, "session_id", "") or ""),
        generation=generation,
    )
    agent._prompt_cache_clock_restored = True  # a live event supersedes a store that could not be read
    _set_event(agent, event)
    return event


def record_unobserved_cache_response(agent: Any, usage: Any, *, reason: str) -> Optional[CacheEvent]:
    """A runtime whose requests Hermes does not build (the Codex app-server sets its own cache
    options): the cache moved but its window cannot be known, so it is recorded ``unknown`` and no
    older window outlives it."""
    read, write = int(getattr(usage, "cache_read_tokens", 0) or 0), int(getattr(usage, "cache_write_tokens", 0) or 0)
    if read + write <= 0:
        return None
    route = _display_route(agent)
    event = CacheEvent(
        route=route, destination=route, route_class="unobserved", requested_ttl=None, sent_at=time.time(),
        read_tokens=read, write_tokens=write, tier=None, confidence=UNKNOWN, reason=reason,
        session_id=str(getattr(agent, "session_id", "") or ""), generation=_agent_generation(agent),
    )
    agent._prompt_cache_clock_restored = True
    _set_event(agent, event)
    return event


# ── Persistence ─────────────────────────────────────────────────────────────


def event_to_json(event: CacheEvent) -> str:
    return json.dumps(asdict(event), separators=(",", ":"))


def event_from_json(raw: Any) -> Optional[CacheEvent]:
    try:
        data: Any = json.loads(raw) if isinstance(raw, str) else None
        if not isinstance(data, dict):
            return None
        return CacheEvent(**{**data, "route": tuple(data["route"]), "destination": tuple(data["destination"])})
    except (TypeError, ValueError, KeyError):  # a column written by a different build: no clock
        return None


def _fallback_event(agent: Any, last_assistant_at: float, generation: Optional[int]) -> CacheEvent:
    """No stored clock (a session from before this column): anchor on the newest assistant row and
    apply the current route's rule with no response evidence. Never a guarantee, at most an end."""
    from agent.prompt_caching import effective_cache_ttl
    provider, base_url = str(getattr(agent, "provider", "") or ""), str(getattr(agent, "base_url", "") or "")
    model = str(getattr(agent, "model", "") or "")
    caching = bool(getattr(agent, "_use_prompt_caching", False))
    requested = effective_cache_ttl(getattr(agent, "_cache_ttl", None), provider=provider, model=model,
                                    base_url=base_url) if caching else None
    route_class = classify_route(provider=provider, base_url=base_url, api_mode=str(getattr(agent, "api_mode", "") or ""),
                                 model=model, has_markers=caching, retention=None)
    verdict = _RULES[route_class](_Ctx(route_class, requested, None, {}, 0, route_ttl_ceiling(provider, base_url)))
    seconds = max(max((TTL_SECONDS[t] for t in verdict.tiers), default=0), verdict.estimate_seconds or 0)
    route = _display_route(agent)
    return CacheEvent(
        route=route, destination=route, route_class=route_class, requested_ttl=requested, sent_at=last_assistant_at,
        read_tokens=0, write_tokens=0, tier=verdict.tiers[-1] if verdict.tiers else verdict.tier,
        confidence=ASSUMED if seconds else UNKNOWN, reason=f"restored_from_transcript:{verdict.reason}",
        estimated_until=last_assistant_at + seconds if seconds else None,
        session_id=str(getattr(agent, "session_id", "") or ""), generation=generation,
    )


def restore_cache_clock(agent: Any) -> Optional[CacheEvent]:
    """Load the session's stored clock once per agent (resume, gateway re-creation). The saved
    generation is kept as saved: a compaction committed since then invalidates it. A failed read
    restores nothing and is retried on the next consumer."""
    db, session_id = getattr(agent, "_session_db", None), getattr(agent, "session_id", None)
    reader = getattr(db, "get_prompt_cache_clock", None)
    if not session_id or not callable(reader):
        agent._prompt_cache_clock_restored = True
        return None
    try:
        stored: Any = reader(session_id) or {}
    except _storage_errors() as exc:
        logger.debug("prompt-cache clock: restore failed for %s (%s); window unknown", session_id, exc)
        return None
    agent._prompt_cache_clock_restored = True
    event = event_from_json(stored.get("clock"))
    if event is not None:
        event = replace(event, session_id=event.session_id or str(session_id))
    elif isinstance(stored.get("last_assistant_at"), (int, float)):
        event = _fallback_event(agent, float(stored["last_assistant_at"]), stored.get("generation"))
    _set_event(agent, event)
    return event


# ── The answer every consumer reads ─────────────────────────────────────────


def _unknown(reason: str, event: Optional[CacheEvent] = None) -> CacheWindow:
    return CacheWindow(None, None, None, UNKNOWN, reason, event.sent_at if event else None)


def _window(event: CacheEvent) -> CacheWindow:
    if event.confidence == UNKNOWN or event.estimated_until is None:
        return _unknown(event.reason, event)
    return CacheWindow(event.guaranteed_until, event.estimated_until, event.tier, event.confidence, event.reason,
                       event.sent_at, dict(event.coverage))


def cache_window(agent: Any) -> Optional[CacheWindow]:
    """The live route's cache window, or None when this session never touched a cache."""
    event = getattr(agent, "_prompt_cache_clock", None)
    if event is None and not getattr(agent, "_prompt_cache_clock_restored", False):
        event = restore_cache_clock(agent)
    if not isinstance(event, CacheEvent):
        # A restore that could not read the store: the session may have a window nobody can see.
        return None if getattr(agent, "_prompt_cache_clock_restored", False) else _unknown("storage_unavailable")
    reason = _agent_invalid_reason(agent, event, _agent_generation(agent))
    return _unknown(reason, event) if reason else _window(event)


def _saved_route_changed(event: CacheEvent, saved: tuple[Any, Any, Any]) -> bool:
    """Whether the session's persisted route (what ``session.mutate`` / ``/model`` wrote, and what
    a resume starts on) differs from the route the event was recorded on. Only persisted parts
    count: a row without a provider or base_url says nothing about them. A ``custom`` agent
    names the transport class while the row names the entry, so its endpoint is the base_url."""
    provider, base_url, model = _route_key(*saved)
    ev_provider, ev_url, ev_model = event.route
    return bool((model and model != ev_model) or (base_url and base_url != ev_url)
                or (provider and ev_provider != "custom" and provider != ev_provider))


def stored_cache_window(raw_clock: Any, *, session_id: str, generation: Optional[int],
                        saved_route: tuple[Any, Any, Any] = (None, None, None)) -> Optional[dict[str, Any]]:
    """:func:`cache_window_payload` for a session with no live agent, from its saved column, checked
    against the session's persisted ``(provider, base_url, model)``."""
    event = event_from_json(raw_clock)
    if event is None:
        return None
    reason = _invalid_reason(event, session_id=session_id, generation=generation, route=None)
    if reason is None and _saved_route_changed(event, saved_route):
        reason = "route_changed"
    return asdict(_unknown(reason, event) if reason else _window(event))


def cache_window_payload(agent: Any) -> Optional[dict[str, Any]]:
    """``cache_window`` as the JSON shape surfaces carry (``PromptCacheWindow`` contract)."""
    window = cache_window(agent)
    return asdict(window) if window is not None else None


def _field(window: Any, name: str) -> Any:
    return window.get(name) if isinstance(window, dict) else getattr(window, name, None)


def cache_certainly_warm(window: Any, *, now: Optional[float] = None) -> bool:
    """True only before the guaranteed end (a documented minimum or a proven tier)."""
    guaranteed = _field(window, "guaranteed_until") if window is not None else None
    return guaranteed is not None and (time.time() if now is None else now) < float(guaranteed)


def cache_certainly_cold(window: Any, *, now: Optional[float] = None) -> bool:
    """True only past the estimated end (Anthropic's exact tier, OpenAI's retention ceiling), never
    past only a minimum. An unknown window is never certainly cold."""
    if window is None or _field(window, "confidence") == UNKNOWN:
        return False
    estimated = _field(window, "estimated_until")
    return estimated is not None and (time.time() if now is None else now) >= float(estimated)


# ── /context line (surface absorbed from #142387) ──────────────────────────

_UNKNOWN_LINES = {"session_rotated": "cache_unknown_moved", "compacted": "cache_unknown_moved",
                  "route_changed": "cache_unknown_moved", "codex_app_server": "cache_unknown_runtime",
                  "storage_unavailable": "cache_unknown_storage"}


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s" if secs else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


def _clock(ts: Any) -> str:
    return time.strftime("%H:%M", time.localtime(float(ts)))


def render_prompt_cache_line(window: Optional[dict[str, Any]], *, now: Optional[float] = None) -> Optional[str]:
    """One ``/context`` line: warm until the guaranteed end, may still be warm until the estimated
    end, expired (only past the estimated end) and how long ago, or unknown."""
    from agent.i18n import t

    if not isinstance(window, dict):
        return None
    estimated = window.get("estimated_until")
    if window.get("confidence") == UNKNOWN or estimated is None:
        line = _UNKNOWN_LINES.get(str(window.get("reason") or ""), "cache_unknown_route")
        return t(f"gateway.context.{line}")
    now = time.time() if now is None else now
    facts = {"ttl": window.get("tier") or "", "confidence": t(f"gateway.context.cache_confidence.{window.get('confidence')}")}
    if cache_certainly_warm(window, now=now):
        guaranteed = float(window["guaranteed_until"])
        return t("gateway.context.cache_warm", until=_clock(guaranteed), remaining=_duration(guaranteed - now), **facts)
    if not cache_certainly_cold(window, now=now):
        return t("gateway.context.cache_maybe_warm", until=_clock(estimated), remaining=_duration(float(estimated) - now),
                 **facts)
    return t("gateway.context.cache_expired", ago=_duration(now - float(estimated)), **facts)

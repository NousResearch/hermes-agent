"""Session-cumulative usage for a local viewer (wave A2 / CALLMAP W3).

Main's HermesCLI read its status bar (context, ``◷`` latency, ``t/s``, the spinner's ``↓ N tok``),
``/usage`` and the exit summary straight off the in-process agent's ``session_*`` counters,
``context_compressor`` and ``get_rate_limit_state()``. On the session authority the agent lives in
the gateway, so the same numbers are projected here: published as ``session.usage`` after every API
call and at turn end of a local-viewer turn, and returned by the ``session.stats`` verb.
"""
from __future__ import annotations

import contextlib
import dataclasses
import logging
import time
from types import SimpleNamespace

logger = logging.getLogger(__name__)

_COUNTERS = (("input_tokens", "session_input_tokens"), ("output_tokens", "session_output_tokens"),
             ("cache_read_tokens", "session_cache_read_tokens"),
             ("cache_write_tokens", "session_cache_write_tokens"),
             ("prompt_tokens", "session_prompt_tokens"), ("completion_tokens", "session_completion_tokens"),
             ("total_tokens", "session_total_tokens"), ("reasoning_tokens", "session_reasoning_tokens"),
             ("api_calls", "session_api_calls"))
# The output-token count at the start of the viewer's current turn (``↓ N tok`` on main's spinner).
TURN_BASE_ATTR = "_viewer_turn_output_base"
# Providers whose own usage API reports normalized window percentages — the same shape a Portal
# usage-window plan sends, so the status bar can carry one. Window label (first word, lowercased) →
# (gauge id, status-bar word): a known id (``week``/``month``) is localized by the Desktop client, any
# other id renders the word as its own title (``gaugeTitle`` in apps/shared/src/plan-usage.ts).
# Insertion order is the status bar's preference — shortest window first, the one that burns while a
# turn runs (Go has no daily window; its 5-hour rolling one is the closest thing).
_PROVIDER_WINDOW_GAUGES: dict[str, dict[str, tuple[str, str]]] = {
    "opencode-go": {"rolling": ("rolling", "5h"), "weekly": ("week", "Weekly"), "monthly": ("month", "Monthly")},
}
# A window this spent is about to block, so it outranks the short one as the status bar's figure
# (the same 80% the Desktop paints red).
_PROVIDER_WINDOW_ALERT_BP = 8_000
# How often the status bar may ask its provider. REFRESH_AFTER_S (2 min) is a picker's cadence; the
# number the user watches mid-turn moves in whole percentage points, so one minute keeps it alive
# without spending a request per usage frame.
_PROVIDER_WINDOW_REFRESH_S = 60.0


def _finite(value):
    return round(float(value), 2) if value is not None and value == value and 0 < value < 1e6 else None


def _context(agent, compressor) -> dict:
    """Main's status-bar context figures (``_get_status_bar_snapshot``), clamped the same way."""
    if compressor is None:
        return {"last_prompt_tokens": 0, "context_tokens": 0, "context_length": None,
                "config_context_length": None, "context_source": None, "compression_count": 0}
    from agent.context_breakdown import context_display_source
    last = max(0, getattr(compressor, "last_prompt_tokens", 0) or 0)
    tokens = last
    with contextlib.suppress(Exception):
        from agent.usage_anchor import anchored_context_tokens
        messages = getattr(agent, "_session_messages", None)
        anchored = anchored_context_tokens(messages if isinstance(messages, list) else [],
                                           getattr(agent, "_turn_base_usage_anchor", None),
                                           charge_stale_thinking=False)
        if anchored is not None and anchored > 0:
            tokens = anchored
    length = max(0, getattr(compressor, "context_length", 0) or 0)
    return {"last_prompt_tokens": last, "context_tokens": min(tokens, length) if length else tokens,
            "context_length": length or None,
            "config_context_length": getattr(compressor, "_config_context_length", None),
            "context_source": context_display_source(compressor),
            "compression_count": getattr(compressor, "compression_count", 0) or 0}


def _rate_limit(agent):
    with contextlib.suppress(Exception):
        state = agent.get_rate_limit_state()
        if state is not None and dataclasses.is_dataclass(state) and not isinstance(state, type):
            return dataclasses.asdict(state)
    return None


def plan_gauge(agent) -> dict | None:
    """Window plans only: the allowance the status bar shows at a glance (today's, else the one that binds),
    from the reply headers the agent already tracks — no extra round trip, None for everyone else."""
    gauge = _portal_plan_gauge(agent)
    if gauge is None:  # a provider that reports its own windows (opencode-go) carries the same slot
        try:
            gauge = _provider_plan_gauge(agent)
        except Exception:  # a status-bar readout must never break usage reporting
            logger.debug("provider plan gauge failed; this frame carries no allowance", exc_info=True)
    return gauge


def _portal_route(agent) -> bool:
    """Whether the agent's *intended* route is a Nous one — the only route that sends plan headers.

    ``_credits_state`` is adopted from those headers and outlives an in-place model switch
    (``agent.provider`` changes, nothing clears the state), so a Go turn after a Nous one would keep
    showing the Nous allowance and never the Go window. The primary runtime is the route a fallback
    turn restores, so a live fallback away from Nous still shows the plan it is spending.
    """
    from agent.rate_limit_credits import _is_nous_agent

    primary = getattr(agent, "_primary_runtime", None) or {}
    route = SimpleNamespace(provider=primary.get("provider"), base_url=primary.get("base_url")) if primary else agent
    return _is_nous_agent(route)


def _portal_plan_gauge(agent) -> dict | None:
    """A Nous Portal usage-window plan: today's allowance, else the one that binds, off the reply headers."""
    with contextlib.suppress(Exception):
        if not _portal_route(agent):
            return None  # the plan state belongs to the route that produced it
        return _portal_state_gauge(getattr(agent, "_credits_state", None))
    return None


def _portal_state_gauge(state) -> dict | None:
    """A passed reset is no longer current: the bar would name an allowance that already refilled."""
    if getattr(state, "is_usage_window", False) is not True:  # `is True`: a Mock agent's attribute is not a plan
        return None
    gauge = _portal_shown_window(state)
    reset = getattr(gauge, "resets_at_ms", None)
    if gauge is None or (isinstance(reset, int) and 0 < reset <= int(time.time() * 1000)):
        return None
    return {"id": gauge.id, "label": gauge.label, "used_bp": int(gauge.used_bp), "resets_at_ms": gauge.resets_at_ms}


def _portal_shown_window(state):
    """A full allowance is what blocks, so it wins over today's all-scope window."""
    binding = state.binding_gauge
    if binding is not None and binding.full:
        return binding
    return next((g for g in state.gauges if g.id == "day" and g.scope == "all"), None) or binding


def _reset_ms(window) -> int | None:
    """A usage window's reset as epoch milliseconds; None when the fetcher left it out."""
    with contextlib.suppress(AttributeError, OSError, OverflowError, TypeError, ValueError):
        return int(window.reset_at.timestamp() * 1000)
    return None


def _pooled_credentials(agent) -> bool:
    """Whether this provider serves from more than one credential.

    The provider-wide fetch below reads whichever key the resolver hands back, so on a pool its
    percentage may be another account's — ``hermes_cli/inventory.py`` refuses the provider-wide
    figure for a pooled provider for the same reason. One account (the common case) is exact.
    ``CredentialPool.entries`` is a METHOD (guarded by the pool's lock), not an attribute."""
    pool = getattr(agent, "_credential_pool", None)
    entries = getattr(pool, "entries", None)
    if callable(entries):
        entries = entries()
    return len(entries or ()) > 1


def _provider_plan_gauge(agent) -> dict | None:
    """A provider that reports its own usage windows — the same gauge a Portal plan sends.

    ``opencode-go``'s ``GET /zen/go/v1/usage`` answers ``{"usage": {"rolling"|"weekly"|"monthly":
    {"percent", "resetsAt"}}}``; its provider profile already normalizes that into an
    ``AccountUsageSnapshot`` for ``/usage``. The status bar carries the provider's shortest window
    (the one that moves while a turn runs), yielding to any window at ``_PROVIDER_WINDOW_ALERT_BP``.
    Read from the account-usage cache — the refresh rides a daemon thread, so a usage frame never
    waits on the network."""
    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    windows = _PROVIDER_WINDOW_GAUGES.get(provider)
    if not windows or _pooled_credentials(agent):
        return None
    from agent.account_usage_cache import cached_account_usage, refresh_account_usage_async, snapshot_is_stale

    # Throttled per provider: a turn's frames cannot storm the provider's API.
    refresh_account_usage_async([provider], interval=_PROVIDER_WINDOW_REFRESH_S)
    snapshot = cached_account_usage(provider)
    if snapshot is None or snapshot_is_stale(snapshot):
        return None  # nothing trustworthy yet; the refresh above lands in a later frame
    return _status_bar_window(snapshot, windows)


def _window_key(window) -> str:
    """A window's stable key, else the first word of its label (fetchers that send no key)."""
    key = str(getattr(window, "key", "") or "").strip().lower()
    return key or str(getattr(window, "label", "") or "").strip().lower().split(" ")[0]


def _window_gauge(window, windows, now_ms: int) -> dict | None:
    """One snapshot window as the status bar's gauge, or None when it has nothing to render."""
    words = windows.get(_window_key(window))
    used, reset = getattr(window, "used_percent", None), _reset_ms(window)
    if words is None or isinstance(used, bool) or not isinstance(used, (int, float)):
        return None
    if reset is not None and reset <= now_ms:
        used, reset = 0.0, None  # Go keeps the last percent until the window refills: a passed reset is empty
    return {"id": words[0], "label": words[1],
            "used_bp": max(0, min(10_000, round(float(used) * 100))), "resets_at_ms": reset}


def _status_bar_window(snapshot, windows) -> dict | None:
    """The provider's shortest window — unless one is about to block, then that one.

    ``windows`` is the provider's table, whose order is the preference (shortest first), so the
    snapshot's own ordering cannot decide the figure. A window at ``_PROVIDER_WINDOW_ALERT_BP`` is
    the news the user needs; of several, the most used wins."""
    now_ms = int(time.time() * 1000)
    gauges = [g for g in (_window_gauge(w, windows, now_ms) for w in snapshot.windows) if g is not None]
    if not gauges:
        return None
    alerting = [g for g in gauges if g["used_bp"] >= _PROVIDER_WINDOW_ALERT_BP]
    if alerting:
        return max(alerting, key=lambda g: g["used_bp"])
    rank = {gauge_id: index for index, (gauge_id, _label) in enumerate(windows.values())}
    return min(gauges, key=lambda g: rank[g["id"]])


def usage_stats(agent) -> dict:
    """The frozen ``session.usage`` / ``session.stats`` dict; zeros before the first turn built an agent."""
    stats: dict = {key: int(getattr(agent, attr, 0) or 0) for key, attr in _COUNTERS}
    stats.update(_context(agent, getattr(agent, "context_compressor", None)))
    avg_latency = tps = None
    with contextlib.suppress(Exception):
        latencies = list(getattr(agent, "_api_latency_history", None) or [])
        outputs = list(getattr(agent, "_api_output_history", None) or [])
        if n := min(len(latencies), len(outputs)):
            total = sum(latencies[-n:])
            avg_latency = _finite(total / n)
            tps = _finite(sum(outputs[-n:]) / total) if total > 0 else None
    active = 0
    with contextlib.suppress(Exception):
        from tools.async_delegation import active_count
        active = active_count()
    base = getattr(agent, TURN_BASE_ATTR, None)
    stats.update(avg_latency_s=avg_latency, tokens_per_s=tps,
                 last_turn_tokens=max(0, stats["output_tokens"] - base) if isinstance(base, int) else 0,
                 rate_limit=_rate_limit(agent) if agent is not None else None, active_subagents=active,
                 model=str(getattr(agent, "model", "") or ""), provider=str(getattr(agent, "provider", "") or ""),
                 fallback_model=(str(getattr(agent, "model", "") or "") or None)
                 if getattr(agent, "_fallback_activated", False) is True else None)
    if (gauge := plan_gauge(agent)) is not None:  # absent off a window plan, as on the embedded path
        stats["plan_gauge"] = gauge
    return stats


def carried_compression_count(live, agent=None) -> int:
    """Compactions of this conversation: main's agent counted them on its compressor for the
    session's life. A canonical compress evicts the gateway's agent (a session boundary), so the
    count rides the live session and seeds the successor (``seed_compression_count``)."""
    own = getattr(getattr(agent, "context_compressor", None), "compression_count", 0)
    carried = getattr(live, "compression_count", 0)
    return max(carried if isinstance(carried, int) else 0, own if isinstance(own, int) else 0)


def note_compression(authority, ref, live) -> None:
    """A committed (never an exact-retried) compress: one more compaction than the evicted agent saw."""
    agent = None
    with contextlib.suppress(AttributeError, KeyError):  # a runner without an agent cache has none
        agent = authority.agent(ref)
    live.compression_count = carried_compression_count(live, agent) + 1


def seed_compression_count(live, agent) -> None:
    compressor = getattr(agent, "context_compressor", None)
    if live is not None and compressor is not None:
        with contextlib.suppress(Exception):
            compressor.compression_count = carried_compression_count(live, agent)


def tui_usage(stats: dict) -> dict:
    """``stats`` in main's TUI ``Usage`` shape (``tui_gateway/server.py::_get_usage``): the fields its
    status bar (context used/max/bar, ``◷`` latency, ``t/s``, ``cmp`` badge, cache hit) and ``/usage``
    read off ``message.complete.usage``. Absent figures are omitted, never fabricated as zeros."""
    usage = {"model": stats.get("model") or "", "input": stats.get("input_tokens") or stats.get("prompt_tokens") or 0,
             "output": stats.get("output_tokens") or stats.get("completion_tokens") or 0,
             "reasoning": stats.get("reasoning_tokens") or 0, "prompt": stats.get("prompt_tokens") or 0,
             "completion": stats.get("completion_tokens") or 0, "total": stats.get("total_tokens") or 0,
             "calls": stats.get("api_calls") or 0}
    used, maximum = stats.get("last_prompt_tokens") or 0, stats.get("context_length") or 0
    if used and maximum:
        # ``agent/context_breakdown.py::context_usage_fields``: current occupancy, clamped.
        used = min(used, maximum)
        source = stats.get("context_source") or "provider_usage"
        usage.update(context_used=used, context_max=maximum,
                     context_percent=max(0, min(100, round(used / maximum * 100))),
                     context_source=source, context_estimated=source != "provider_usage")
    if stats.get("context_source") is not None:  # a compressor, sized or not (``_get_usage``'s ``if comp``)
        usage["compressions"] = stats.get("compression_count") or 0
    prompt_total, cache_read = stats.get("prompt_tokens") or 0, stats.get("cache_read_tokens") or 0
    if prompt_total > 0 and cache_read > 0:
        usage["cache_hit_pct"] = max(0, min(100, round(cache_read / prompt_total * 100)))
    for key, source_key in (("avg_latency_s", "avg_latency_s"), ("avg_tps", "tokens_per_s")):
        if isinstance(stats.get(source_key), (int, float)):
            usage[key] = round(float(stats[source_key]), 1)
    if isinstance(stats.get("active_subagents"), int):
        usage["active_subagents"] = stats["active_subagents"]
    if isinstance(stats.get("plan_gauge"), dict):
        usage["plan_gauge"] = stats["plan_gauge"]
    return usage


def usage_frame(stats: dict) -> dict:
    """The ``session.usage`` event payload: BOTH views of the same counters, in one frame.

    ``SessionUsagePayload`` (``tui_gateway/contracts/events.py``) allows either view — the TUI
    ``usage`` object (``server._start_usage_ticker``) or the session-cumulative counters at the top
    level — and the readers on the far end want different halves. The classic CLI
    (``hermes_cli.gateway_connection.SessionState.apply_event``) keeps the top-level counters
    whenever they are present and ignores the nested view; ACP (``acp_adapter.gateway_server``)
    reads only the top level; a Desktop bundle older than ``usageFromTick`` reads only
    ``payload.usage``, so its context / cache-hit / tokens-per-second meters render their empty
    placeholder for as long as the frame carries the counters alone. Publishing both serves every
    viewer from one frame, and leaves the counters untouched for every existing consumer.

    ``usage`` is the standard status-bar projection of the embedded ticker's ``server._get_usage``
    view, with two exceptions: ``dev_credits_spent_micros`` (``HERMES_DEV_CREDITS`` only) is not
    carried, and ``avg_latency_s`` / ``avg_tps`` are rounded from the counters' two decimals rather
    than the raw values, so they can differ by 0.1 at a rounding boundary. A Desktop bundle that
    reads only ``payload.usage`` fills its meters here too. The
    top-level counters on this event are deprecated: new viewers read ``usage`` (or call
    ``session.stats`` for the counters). Before they can be dropped, these readers must move off
    them: the classic CLI's ``SessionState.apply_event`` → ``GatewayAgent.apply_session_usage``,
    ACP's ``_project_usage`` (``usage_update`` / ``turn_usage``), and the flat branch of
    ``usageFromTick`` (Desktop and Ink).
    """
    return {"usage": tui_usage(stats), **stats}


def publish_usage(turn, agent, *, record: bool = False) -> None:
    """One ``session.usage`` frame on the viewer's stream; never breaks the turn. At turn end
    (``record``) the same dict rides the captured execution result, so the turn's
    ``message.complete`` carries main's ``usage`` (``session_settlement_recovery.completion_payload``)."""
    if agent is None:
        return
    with contextlib.suppress(Exception):
        stats = usage_stats(agent)
        turn._publish_execution("session.usage", usage_frame(stats))
        if record:
            from gateway.session_results import execution_result
            captured = execution_result.get()
            if captured is not None:
                captured["session_usage"] = stats


_STORED_COUNTERS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")


def _stored_stats(row: dict) -> dict:
    """A session row's cumulative counters in ``usage_stats``' shape. The row's ``input_tokens`` excludes cache
    reads and writes, so the prompt total adds them back for the cache hit rate."""
    counts = {key: int(row.get(key) or 0) for key in _STORED_COUNTERS}
    return {**counts, "prompt_tokens": counts["input_tokens"] + counts["cache_read_tokens"] + counts["cache_write_tokens"],
            "completion_tokens": counts["output_tokens"], "total_tokens": counts["input_tokens"] + counts["output_tokens"],
            "api_calls": int(row.get("api_call_count") or 0), "model": row.get("model") or ""}


def snapshot_usage(authority, ref, agent) -> dict | None:
    """The status-bar ``usage`` a resume / ``session.info`` snapshot carries, so a viewer switching to the
    session paints its own cache hit and ``t/s`` instead of the last-viewed session's. An agent that has
    called the model answers as its ``session.usage`` frames do; otherwise (no agent since a restart, or one
    not used yet) the stored row answers, with no ``t/s`` since latency is not stored."""
    if agent is not None and getattr(agent, "session_api_calls", 0):
        result = usage_stats(agent)
        result["compression_count"] = carried_compression_count(authority.sessions.get(ref.session_id), agent)
        return tui_usage(result)
    from gateway.session_persistence import authority_persistence
    row = authority_persistence(authority).sessions.get(ref.session_id)
    return tui_usage(_stored_stats(row.as_session_row())) if row else None


async def account_usage(connection, ref, params):
    """``session.account_usage {session_id}``: the CLI's /usage account blocks for the route this
    session runs (``_session_runtime``: its agent's, else the one its next turn resolves, so
    ``provider: auto`` is the saved login), fetched off the loop in the owning profile's scope:
    the provider's plan limits, then the Nous section whenever Nous is signed in (``nous_usage_view``), with
    the same read's dollar model on a Nous route (``nous_usage``: the classic CLI's two-bar block)."""
    import asyncio
    from agent.account_usage import account_limit_lines
    from agent.account_usage_nous import nous_usage_view
    from agent.billing_usage import usage_model_wire
    from agent.billing_links import is_nous_inference_route
    from gateway.session_authorities import owner_scope
    from gateway.session_llm_oneshot import _session_runtime
    from hermes_cli.auth import AuthError
    await connection.authority.resolve(connection.actor, ref)
    authority = connection.authority
    runtime = _session_runtime(authority, ref)
    credits_state = getattr(authority.agent(ref), "_credits_state", None)  # the last reply's Nous gauges

    def fetch():
        with owner_scope(authority):
            try:
                route = runtime() or {}
            except AuthError:  # signed out: no route to report limits or a Nous balance for
                route = {}
            provider, base_url = route.get("provider") or "", route.get("base_url") or ""
            lines = account_limit_lines(provider, base_url=base_url or None, api_key=route.get("api_key"))
            nous_route = is_nous_inference_route(provider, base_url)
            nous, usage = nous_usage_view(nous_route=nous_route, credits_state=credits_state)
        dollars = usage_model_wire(usage)
        return {"lines": lines, "nous_lines": nous, "nous_route": nous_route} | ({"nous_usage": dollars} if dollars else {})
    return await asyncio.to_thread(fetch)


async def stats(connection, ref, params):
    """``session.stats {session_id}``: the same dict ``session.usage`` carries."""
    from persistence.core.errors import AuthorityRefusal
    if set(params) - {"session_id", "profile"}:
        raise AuthorityRefusal("invalid_params")
    await connection.authority.resolve(connection.actor, ref)
    agent = connection.authority.agent(ref)
    result = usage_stats(agent)
    result["compression_count"] = carried_compression_count(connection.authority.sessions.get(ref.session_id), agent)
    return result

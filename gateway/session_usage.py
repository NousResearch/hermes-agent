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
import time

_COUNTERS = (("input_tokens", "session_input_tokens"), ("output_tokens", "session_output_tokens"),
             ("cache_read_tokens", "session_cache_read_tokens"),
             ("cache_write_tokens", "session_cache_write_tokens"),
             ("prompt_tokens", "session_prompt_tokens"), ("completion_tokens", "session_completion_tokens"),
             ("total_tokens", "session_total_tokens"), ("reasoning_tokens", "session_reasoning_tokens"),
             ("api_calls", "session_api_calls"))
# Cost statuses that mean "this session's spend is known". A real agent initializes the counter to
# 0.0 with status "unknown"; reporting that as a price is a fabricated $0.00.
_PRICED_COST_STATUSES = ("estimated", "actual", "included")
# The output-token count at the start of the viewer's current turn (``↓ N tok`` on main's spinner).
TURN_BASE_ATTR = "_viewer_turn_output_base"


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
    with contextlib.suppress(Exception):
        state = getattr(agent, "_credits_state", None)
        if getattr(state, "is_usage_window", False) is True:  # `is True`: a Mock agent's attribute is not a plan
            binding = state.binding_gauge
            day = next((g for g in state.gauges if g.id == "day" and g.scope == "all"), None)
            # A full allowance is what blocks, so it wins over today's; a passed reset is no longer current.
            gauge = binding if binding is not None and binding.full else (day or binding)
            reset = getattr(gauge, "resets_at_ms", None)
            if gauge is not None and not (isinstance(reset, int) and 0 < reset <= int(time.time() * 1000)):
                return {"id": gauge.id, "label": gauge.label, "used_bp": int(gauge.used_bp),
                        "resets_at_ms": gauge.resets_at_ms}
    return None


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
    # Money: the agent's own per-call accumulation (``agent/turn_usage.py`` prices every response
    # through ``estimate_usage_cost``, cache reads/writes at their own rates).
    #
    # Only a route that actually priced the session may report a figure. A real agent starts with
    # ``session_estimated_cost_usd = 0.0`` and ``session_cost_status = "unknown"``
    # (``agent/agent_init.py``), so testing ``isinstance`` alone would emit 0.0 for a fresh session
    # and for a model with no pricing entry — and a client would render "$0.0000", the exact
    # fabrication this contract forbids. ``included`` is a known answer (a subscription) and rides
    # through with its own label on the client.
    cost = getattr(agent, "session_estimated_cost_usd", None)
    cost_status = str(getattr(agent, "session_cost_status", "") or "") or None
    estimated_cost = (round(float(cost), 6)
                      if cost_status in _PRICED_COST_STATUSES
                      and isinstance(cost, (int, float)) and not isinstance(cost, bool) else None)
    stats.update(estimated_cost_usd=estimated_cost,
                 cost_status=cost_status,
                 cost_source=str(getattr(agent, "session_cost_source", "") or "") or None)
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
    # The raw cache counts ride along so a client can show "8.4k cached / 1.2k written" beside the
    # percentage; omitted at zero, like the hit rate.
    for key, source_key in (("cache_read", "cache_read_tokens"), ("cache_write", "cache_write_tokens")):
        if stats.get(source_key):
            usage[key] = stats[source_key]
    # Session spend (USD, estimated from the provider's published rates) — absent when the route
    # has no pricing entry, so a client shows no money figure instead of a false $0.
    cost = stats.get("estimated_cost_usd")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool):
        usage["cost_usd"] = round(float(cost), 6)
    if stats.get("cost_status"):
        usage["cost_status"] = stats["cost_status"]
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
    reads and writes, so the prompt total adds them back for the cache hit rate. A priced row also carries its
    spend: a resumed session's meter must not read zero until the next model call."""
    counts = {key: int(row.get(key) or 0) for key in _STORED_COUNTERS}
    stats = {**counts, "prompt_tokens": counts["input_tokens"] + counts["cache_read_tokens"] + counts["cache_write_tokens"],
             "completion_tokens": counts["output_tokens"], "total_tokens": counts["input_tokens"] + counts["output_tokens"],
             "api_calls": int(row.get("api_call_count") or 0), "model": row.get("model") or ""}
    cost = row.get("estimated_cost_usd")
    status = str(row.get("cost_status") or "")
    if status in _PRICED_COST_STATUSES and isinstance(cost, (int, float)) and not isinstance(cost, bool):
        stats.update(estimated_cost_usd=round(float(cost), 6), cost_status=status,
                     cost_source=row.get("cost_source") or None)
    return stats


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


# A session's stored counters → the agent attributes the status bar reads. ``input_tokens`` in the
# session row excludes cache reads/writes (the prompt total adds them back), matching CanonicalUsage.
# ``api_call_count`` rides along because the gateway's cumulative writer sets it absolutely — leaving
# it out would rewrite the session's call count with this process's few calls. The first-response
# branches keyed on ``session_api_calls == 1`` are unaffected in practice: a resumed session is not
# on its first response, and a fresh one seeds nothing.
_STORED_TO_AGENT = (
    ("input_tokens", "session_input_tokens"),
    ("output_tokens", "session_output_tokens"),
    ("cache_read_tokens", "session_cache_read_tokens"),
    ("cache_write_tokens", "session_cache_write_tokens"),
    ("reasoning_tokens", "session_reasoning_tokens"),
    ("api_call_count", "session_api_calls"),
)


def seed_session_totals(agent, row) -> bool:
    """Add a resumed agent's session history to the counters it has accumulated since this process
    started, so the status bar reports the CONVERSATION (what the session has spent in total), not
    just what happened after the last gateway restart.

    Both halves are added together and stay consistent: a token count and a spend that describe the
    same span. Seeding only the cost would show a few thousand tokens next to $1.41.

    Idempotent per agent (``_session_totals_seeded``), and safe against double counting because the
    destination is additive only once per process while the persistence path writes absolute totals
    from these attributes — history + every increment since the restart, which is exactly the sum.

    Derived fields (``session_prompt_tokens`` / ``session_completion_tokens`` / ``session_total_tokens``)
    are recomputed from the seeded parts rather than added, so they cannot drift. ``api_calls`` is
    seeded too (see ``_STORED_TO_AGENT``) because the cumulative writer stores it absolutely.
    """
    if agent is None or not isinstance(row, dict) or getattr(agent, "_session_totals_seeded", False):
        return False
    stored = {key: int(row.get(key) or 0) for key in
              ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
               "reasoning_tokens", "api_call_count")}
    cost = row.get("estimated_cost_usd")
    stored_cost = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else 0.0
    if not any(stored.values()) and stored_cost <= 0:
        agent._session_totals_seeded = True
        return False
    for row_key, attr in _STORED_TO_AGENT:
        if stored[row_key] > 0:
            setattr(agent, attr, int(getattr(agent, attr, 0) or 0) + stored[row_key])
    prompt = sum(int(getattr(agent, attr, 0) or 0)
                 for attr in ("session_input_tokens", "session_cache_read_tokens", "session_cache_write_tokens"))
    completion = int(getattr(agent, "session_output_tokens", 0) or 0)
    agent.session_prompt_tokens = prompt
    agent.session_completion_tokens = completion
    agent.session_total_tokens = prompt + completion
    if stored_cost > 0:
        agent.session_estimated_cost_usd = float(getattr(agent, "session_estimated_cost_usd", 0.0) or 0.0) + stored_cost
        agent.session_cost_status = row.get("cost_status") or "estimated"
        agent.session_cost_source = row.get("cost_source") or None
    agent._session_totals_seeded = True
    return True

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
    if maximum:
        # ``agent/context_breakdown.py::context_usage_fields``: the window is reported even before
        # anything is counted in it (a model switch adopts the new window and zeroes occupancy);
        # occupancy figures stay absent until there is a reading, never substituted.
        usage["context_max"] = maximum
        if used:
            used = min(used, maximum)
            source = stats.get("context_source") or "provider_usage"
            usage.update(context_used=used,
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


def publish_switch_usage(authority, session_id: str, live, context_length: int | None) -> None:
    """A committed model switch: put the new window on the status bar now, not at the next turn.

    The pick evicts the cached agent (``session_local.publish_local_policy``), so the ``session.info``
    that follows reads a mirror that cannot size the new model's window: without this frame the
    previous model's window (1M) and occupancy would stand until a prompt is priced in the new one
    (500k). Occupancy stays absent -- ``update_model`` zeroed the old model's reading and nothing has
    been counted in the new window yet -- so ``tui_usage`` never turns "unknown" into ``0``. A ``None``
    window is carried as an explicit ``context_length`` null for the same reason: every viewer merges
    shallowly, and an omitted field would keep the window of the model the session just left.

    *session_id* is the committed receipt's session (a live session carries its route and source, not
    its id); *live* only answers whether this session has a local viewer to paint.
    """
    from gateway.config import Platform
    sid = session_id or None
    # A local viewer's projection only: platform chats get their figures with the turn's own replies.
    if not sid or getattr(getattr(live, "source", None), "platform", None) != Platform.LOCAL:
        return
    with contextlib.suppress(Exception):  # an informational frame never fails a committed switch
        maximum = max(0, int(context_length or 0))
        stats = _stored_stats(authority.db.get_session(sid) or {})
        stats.update(last_prompt_tokens=0, context_length=maximum or None)
        frame = usage_frame(stats)
        frame["context_reset"] = True
        authority.sessions[sid].event_stream.publish(sid, frame, event_type="session.usage")


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
    row = authority.db.get_session(ref.session_id)
    return tui_usage(_stored_stats(row)) if row else None


async def stats(connection, ref, params):
    """``session.stats {session_id}``: the same dict ``session.usage`` carries."""
    from hermes_state_runtime import RuntimeStoreError
    if set(params) - {"session_id", "profile"}:
        raise RuntimeStoreError("invalid_params")
    await connection.authority.resolve(connection.actor, ref)
    agent = connection.authority.agent(ref)
    result = usage_stats(agent)
    result["compression_count"] = carried_compression_count(connection.authority.sessions.get(ref.session_id), agent)
    return result

"""The proactive tool-result prune's two call sites: after a tool round (every iteration where full
compression stands down) and at turn start (cold provider cache only, where most expiries happen:
a person comes back after a pause). Both share the safety gates and the commit contract.
Nothing here imports ``agent.conversation_loop`` at module level (cycle)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agent.context_compressor import _DB_PERSISTED_MARKER
from agent.conversation_compression import _reset_read_dedup_caches
from agent.prompt_cache_clock import DECLARED, PROVEN, cache_certainly_cold, cache_window

logger = logging.getLogger("agent.conversation_loop")


def prompt_cache_is_cold(agent: Any, *, now: float | None = None) -> bool:
    """Whether the provider prompt cache has certainly expired, so a history rewrite breaks nothing.

    The prune's ONLY cold-cache reading (both call sites), from the shared cache clock: past the
    window's estimated end only. An unknown window (no cache touched yet, an unreadable store, a
    route switch, or a compaction or prune since the last response) or one that may still be warm
    answers False. The clock restores a resumed session's window from the store itself.

    Only a proven or declared window prunes. An assumed one includes the window the clock rebuilds for a
    pre-clock session from TODAY's TTL and route: a session cached for 1h (or on another route) then
    reads cold while still warm. The /context line keeps showing that estimate.

    A route switch (``/model``, a fallback, any change of provider, endpoint or model since the last
    response) is cold too: the new route has never seen this prefix, so the first request there writes
    it whole and an offload before it breaks nothing."""
    window = cache_window(agent)
    if window is not None and window.reason == "route_changed":
        return True
    return window is not None and window.confidence in (PROVEN, DECLARED) and cache_certainly_cold(window, now=now)


def run_proactive_prune(
    agent: Any, messages: list[dict[str, Any]], current_tokens: int, effective_task_id: Any,
) -> list[dict[str, Any]]:
    """Run the post-tool prune helper once (same gates, suppression accounting and commit contract);
    reset the read dedup on a commit. Returns the list the loop continues with."""
    from agent.turn_preflight import _proactive_tool_result_prune

    pruned = _proactive_tool_result_prune(agent, agent.context_compressor, messages, current_tokens)
    if pruned is not messages:
        # A committed prune is a content-loss boundary like compaction: demoted skill_view / read_file bodies
        # survive only as one-line markers, so the repeat-read dedup must stop answering "unchanged" for them
        # or the reload the marker asks for is refused.
        _reset_read_dedup_caches(effective_task_id, session_id=agent.session_id or "")
    return pruned


def turn_start_cold_prune(agent: Any, out: Any, effective_task_id: Any) -> None:
    """Turn-start prune, only on an already-expired provider cache (the rewrite then costs no cache break).

    Runs on the history BEFORE this turn's user row and only while that row (and everything after it) is
    still unwritten: the commit then cannot touch it, so the turn-start persist and ``api_content`` stamp
    handle it exactly as on an unpruned turn. A row the TUI/Desktop already wrote at submit is skipped: the
    commit would clone it under a new id while the live dict keeps the old ``_row_id``. The prune replaces
    bodies in place and never drops rows, so ``current_turn_user_idx`` stays valid.
    """
    idx = out.current_turn_user_idx
    if (out.compressed or out.blocked or not agent.compression_enabled
            or not isinstance(idx, int) or not 0 < idx < len(out.messages)
            or any(isinstance(m, dict) and (m.get("_row_id") or m.get(_DB_PERSISTED_MARKER))
                   for m in out.messages[idx:])
            or not prompt_cache_is_cold(agent)):
        return
    from agent import turn_context as _tc

    prefix = out.messages[:idx]
    tokens = _tc._preflight_request_tokens(agent, out.messages, out.active_system_prompt or "")
    pruned = run_proactive_prune(agent, prefix, tokens, effective_task_id)
    if pruned is not prefix and len(pruned) == len(prefix):
        out.messages = pruned + out.messages[idx:]


@dataclass
class RouteOffloadVerdict:
    action: str
    messages: Any


def route_change_offload(
    agent: Any, *, messages: Any, current_turn_user_idx: Any, active_system_prompt: Any, effective_task_id: Any,
) -> RouteOffloadVerdict:
    """Before every request: the cold offload on a route that has not answered this prefix yet.

    A fallback switches provider/model inside the turn and its restart never reaches the turn-start prune,
    so without this the new route's first request writes all the old tool output and marks it warm. Only
    ``route_changed`` counts here (an expiry is the turn-start job). The running turn is never shortened
    (``turn_start``), so this is safe on any iteration. Each clock event is checked once per route: a retry
    on the same route before a response does not redo the estimate, and a commit makes the window
    ``compacted`` so the next iteration stands down."""
    from agent.prompt_cache_clock import _display_route

    window = cache_window(agent)
    if window is None or window.reason != "route_changed" or not agent.compression_enabled:
        return RouteOffloadVerdict("fallthrough", messages)
    seen = (getattr(agent, "_prompt_cache_clock", None), _display_route(agent))
    previous = getattr(agent, "_route_offload_checked", None)
    if previous is not None and previous[0] is seen[0] and previous[1] == seen[1]:
        return RouteOffloadVerdict("fallthrough", messages)
    agent._route_offload_checked = seen
    from agent import turn_context as _tc
    from agent.turn_preflight import _proactive_tool_result_prune

    tokens = _tc._preflight_request_tokens(agent, messages, active_system_prompt or "")
    pruned = _proactive_tool_result_prune(agent, agent.context_compressor, messages, tokens,
                                          turn_start=current_turn_user_idx)
    if pruned is messages:
        return RouteOffloadVerdict("fallthrough", messages)
    _reset_read_dedup_caches(effective_task_id, session_id=agent.session_id or "")
    return RouteOffloadVerdict("fallthrough", pruned)


def offload_cold_history(agent: Any, messages: list[dict[str, Any]], task_id: Any = None) -> list[dict[str, Any]]:
    """Step 3's offload on an idle chat whose cache is over (``prompt_cache_lifecycle.claim_offload``):
    the same cold prune a return after expiry runs (byte-exact archive + pointer, protected tail, the
    16k reclaim gate), run by the surface that owns the chat so it is already small when the person comes
    back. Returns the history to keep (the input object when nothing committed)."""
    if not messages or not agent.compression_enabled or not prompt_cache_is_cold(agent):
        return messages
    from agent.conversation_compression_manual import estimate_request_tokens

    pruned = run_proactive_prune(agent, messages, estimate_request_tokens(agent, messages),
                                 task_id or getattr(agent, "session_id", None) or "default")
    return pruned if pruned is not messages and len(pruned) == len(messages) else messages

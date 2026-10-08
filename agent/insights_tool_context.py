"""Tool-context advisor for ``hermes insights``: which eager tools pay their schema on
every request without being reached by most sessions, and which deferred tools are hot.

A tool's full schema sits in static context on every turn whether or not the
conversation ever calls it; the ``tools.tool_search.defer`` list moves cold tools behind
the ``tool_search`` bridge. The decision needs two numbers per tool that nothing surfaced
before: *reach* (share of tool-using sessions that invoked it at least once) and the
*schema cost* it adds to every request. Cursor published the same rule for its harness
(keep tools reached by most conversations eager; defer those under ~20%) and cut static
tool-description tokens by 60% with it.

The advisor is read-only: it never changes the served tool list. The runtime keeps the
tool list byte-stable across a conversation for the prompt cache, so usage-driven
re-inflation mid-session is off the table by design; the user applies the advice through
config, once, and the next conversation pays less.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Cursor's cut-off: tools needed in fewer than ~20% of conversations left static context.
DEFER_REACH_PCT = 20.0
# A deferred tool most sessions end up calling anyway costs a tool_search/tool_describe
# round trip each time; above this reach the stub is worse than the schema.
HOT_REACH_PCT = 50.0
# Below this many tool-using sessions the reach percentages are noise, not evidence.
MIN_SESSIONS = 20
# Ambient affordances that a live A/B showed must stay eager regardless of reach (deferring
# ``clarify`` collapsed structured-clarify usage 18/18 -> 7/18); never suggest them.
_NEVER_SUGGEST = frozenset({"clarify"})


def _schema_tokens(td: Dict[str, Any]) -> int:
    """Static cost of one tool-def, the same chars/4 estimate the bridge budgets with."""
    from tools.tool_search_catalog import CHARS_PER_TOKEN

    return int(len(json.dumps(td, separators=(",", ":"), ensure_ascii=False)) / CHARS_PER_TOKEN)


def _served_tools(platform: str) -> tuple[Dict[str, int], Dict[str, int]]:
    """``(eager, deferred)`` name -> schema tokens for the tools a ``platform`` session on
    this install serves, split exactly as the runtime splits them: the platform's resolved
    toolsets (``hermes tools`` config) through the raw enabled tool-defs (availability probes
    applied) and ``classify_tools`` with the effective ``tools.tool_search.defer`` set."""
    from hermes_cli.config import load_config
    from hermes_cli.tools_config import _get_platform_tools
    from model_tools import get_tool_definitions
    from tools.tool_search import classify_tools, load_config_readonly

    enabled = sorted(_get_platform_tools(load_config(), platform)) or None
    defs = get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True, skip_tool_search_assembly=True)
    visible, deferrable = classify_tools(defs, load_config_readonly().effective_defer_tools)
    by_name = lambda tds: {td["function"]["name"]: _schema_tokens(td) for td in tds}  # noqa: E731
    return by_name(visible), by_name(deferrable)


def tool_context_advice(tools: List[Dict[str, Any]], tool_sessions: int, platform: str = "cli") -> Dict[str, Any]:
    """Advice from a ranked ``tools`` list (entries carry ``tool``, ``sessions``,
    ``reach_pct``) over ``tool_sessions`` tool-using sessions, against the tool surface
    ``platform`` serves (the report's ``--source``; the CLI when unfiltered).

    Returns ``{"sample_sessions", "insufficient", "defer_candidates", "hot_deferred",
    "eager_schema_tokens"}``. ``defer_candidates`` are eager tools under
    ``DEFER_REACH_PCT`` reach ordered by schema cost (the biggest saving first);
    ``hot_deferred`` are deferrable tools at or over ``HOT_REACH_PCT``. Both lists are
    empty when the sample is too small or the registry is unavailable.
    """
    advice: Dict[str, Any] = {
        "platform": platform, "sample_sessions": tool_sessions, "insufficient": tool_sessions < MIN_SESSIONS,
        "defer_candidates": [], "hot_deferred": [], "eager_schema_tokens": 0,
    }
    if advice["insufficient"] or not tools:
        return advice
    try:
        eager, deferred = _served_tools(platform)
    except Exception:
        logger.debug("tool-context advice unavailable (registry/config)", exc_info=True)
        return advice
    reach = {t["tool"]: t for t in tools}

    def entry(name: str, tokens: int) -> Dict[str, Any]:
        seen = reach.get(name) or {}
        return {"tool": name, "reach_pct": float(seen.get("reach_pct", 0.0)),
                "sessions": int(seen.get("sessions", 0)), "schema_tokens": tokens}

    advice["eager_schema_tokens"] = sum(eager.values())
    advice["defer_candidates"] = [
        e for name, tokens in eager.items() if name not in _NEVER_SUGGEST
        if (e := entry(name, tokens))["reach_pct"] < DEFER_REACH_PCT]
    # A used tool this process cannot see (an MCP server's, a plugin's) is deferred by construction;
    # the bridge's own names are plumbing, not capabilities.
    from tools.tool_search_catalog import BRIDGE_TOOL_NAMES

    hot_names = (set(deferred) | (set(reach) - set(eager))) - BRIDGE_TOOL_NAMES
    advice["hot_deferred"] = [
        e for name in hot_names if (e := entry(name, deferred.get(name, 0)))["reach_pct"] >= HOT_REACH_PCT]
    advice["defer_candidates"].sort(key=lambda e: (-e["schema_tokens"], e["tool"]))
    advice["hot_deferred"].sort(key=lambda e: (-e["reach_pct"], e["tool"]))
    return advice


def format_tool_context(advice: Dict[str, Any], section: List[str]) -> List[str]:
    """Terminal lines for the advice (``section`` is the caller's rendered header)."""
    if advice.get("insufficient"):
        return []
    cands, hot = advice.get("defer_candidates") or [], advice.get("hot_deferred") or []
    if not cands and not hot:
        return []
    lines = list(section) + [
        f"  Eager tool schemas cost ~{advice.get('eager_schema_tokens', 0):,} tokens on every "
        f"{advice.get('platform', 'cli')} request ({advice.get('sample_sessions', 0)} tool-using sessions sampled)."]
    if cands:
        saving = sum(e["schema_tokens"] for e in cands)
        lines += [f"  Reached by <{DEFER_REACH_PCT:.0f}% of sessions yet eager (~{saving:,} tokens/request):",
                  f"  {'Tool':<28} {'Reach':>7} {'Tokens':>8}"]
        lines += [f"  {e['tool'][:28]:<28} {e['reach_pct']:>6.1f}% {e['schema_tokens']:>8,}" for e in cands[:8]]
        lines.append("  Add them to tools.tool_search.defer (the list replaces the default set).")
    if hot:
        lines += [f"  Deferred yet reached by ≥{HOT_REACH_PCT:.0f}% of sessions (each use costs a bridge round trip):"]
        lines += [f"  {e['tool'][:28]:<28} {e['reach_pct']:>6.1f}%" for e in hot[:8]]
    return lines + [""]

"""Per-turn tool surface for requests the active route cannot hold.

Tool definitions are the one part of a request that compaction cannot shrink: with 50+ MCP tools
the schemas are 20-30K tokens on their own, so a route whose window is smaller than the system
prompt plus that floor rejects every attempt however much of the transcript is summarized.

Progressive disclosure (``tools/tool_search.py``) is the configured answer — MCP and plugin schemas
are replaced by the tool_search/tool_describe/tool_call bridge. A session that still carries those
schemas inline (bridge off by config, or its assembly failed) has no way out. ``agent/turn_overflow.py``
re-renders such a surface through that same bridge when a provider-proven overflow shows the inline
schemas cannot fit, and retries the turn. Every tool stays reachable — the model loads a schema with
``tool_describe`` and invokes it with ``tool_call`` — so nothing is dropped.

The rendered surface is turn-scoped: the request builder substitutes it for ``agent.tools`` and the
turn prologue clears it, so the next turn starts from the configured surface again.
"""

from __future__ import annotations

from typing import Any, List, Optional

_AGENT_ATTR = "_overflow_deferred_tool_surface"


def deferred_tool_surface(agent: Any) -> Optional[List[Any]]:
    """Tool definitions recorded for the rest of this turn, or ``None`` when none are."""
    surface = getattr(agent, _AGENT_ATTR, None)
    return surface if isinstance(surface, list) and surface else None


def apply_deferred_tool_surface(agent: Any, tools: Any) -> Any:
    """``tools`` replaced by the recorded surface — the input list when none is recorded."""
    surface = deferred_tool_surface(agent)
    return surface if surface is not None else tools


def record_deferred_tool_surface(agent: Any, tool_defs: List[Any]) -> None:
    """Serve *tool_defs* in place of ``agent.tools`` for the remainder of this turn."""
    setattr(agent, _AGENT_ATTR, list(tool_defs))


def reset_deferred_tool_surface(agent: Any) -> None:
    """Clear the record so the next turn starts from the configured tool surface."""
    setattr(agent, _AGENT_ATTR, None)

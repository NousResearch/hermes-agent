"""Finalize one agent's effective tool surface after provider injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from tools.tool_search import AssemblyResult, ToolSearchConfig, assemble_tool_defs, load_config


def _deduplicate_tool_defs(tool_defs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for tool_def in tool_defs:
        if not isinstance(tool_def, dict):
            continue
        name = (tool_def.get("function") or {}).get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(tool_def)
    return result


@dataclass(frozen=True)
class AgentToolSurface:
    """One prepared, internally consistent session-local tool snapshot."""

    source_defs: tuple[Dict[str, Any], ...]
    config: ToolSearchConfig
    context_length: Optional[int]
    assembly: AssemblyResult
    visible_names: frozenset[str]


def prepare_agent_tool_surface(
    agent: Any,
    *,
    source_tool_defs: Iterable[Dict[str, Any]],
    config: Optional[ToolSearchConfig] = None,
    context_length: Optional[int] = None,
) -> AgentToolSurface:
    """Build a complete surface without mutating the live agent."""
    if config is None:
        config = load_config()
    source = tuple(_deduplicate_tool_defs(source_tool_defs))
    assembly = assemble_tool_defs(
        list(source),
        context_length=context_length,
        config=config,
    )
    visible_names = frozenset(
        name
        for tool in assembly.tool_defs
        if (name := (tool.get("function") or {}).get("name"))
    )
    return AgentToolSurface(
        source_defs=source,
        config=config,
        context_length=context_length,
        assembly=assembly,
        visible_names=visible_names,
    )


def publish_agent_tool_surface(agent: Any, surface: AgentToolSurface) -> None:
    """Publish every dependent tool-surface attribute from one snapshot."""
    assembly = surface.assembly
    agent._tool_search_config = surface.config
    agent._tool_search_context_length = surface.context_length
    agent._tool_search_source_defs = surface.source_defs
    if assembly.activated:
        agent._tool_search_catalog = tuple(assembly.catalog)
        agent._tool_search_allowed_names = frozenset(
            entry.name for entry in assembly.catalog
        )
    else:
        # ``None`` means no finalized bridge scope is active. Keep that distinct
        # from an authoritative empty catalog so legacy/manual bridge callers can
        # still use the scoped registry fallback when Tool Search is disabled.
        agent._tool_search_catalog = None
        agent._tool_search_allowed_names = None
    agent._tool_search_assembly = assembly
    agent.tools = list(assembly.tool_defs)
    agent.valid_tool_names = set(surface.visible_names)
    agent._tool_search_scope_cache = None


def finalize_agent_tool_surface(
    agent: Any,
    *,
    source_tool_defs: Optional[Iterable[Dict[str, Any]]] = None,
    config: Optional[ToolSearchConfig] = None,
    context_length: Optional[int] = None,
) -> AssemblyResult:
    """Publish visible tools and the deferred catalog from one complete snapshot."""
    source = _deduplicate_tool_defs(
        source_tool_defs if source_tool_defs is not None else (getattr(agent, "tools", None) or [])
    )
    surface = prepare_agent_tool_surface(
        agent,
        source_tool_defs=source,
        context_length=context_length,
        config=config,
    )
    publish_agent_tool_surface(agent, surface)
    return surface.assembly


__all__ = [
    "AgentToolSurface",
    "finalize_agent_tool_surface",
    "prepare_agent_tool_surface",
    "publish_agent_tool_surface",
]

"""Apply delegated read-only memory policy before provider startup."""

import logging

from agent.memory_manager_tools import memory_provider_tools_enabled, memory_provider_tools_exposed

logger = logging.getLogger(__name__)


def apply_memory_read_policy(agent, handed_in=None) -> None:
    """Never restrict an owner's shared manager; restrict fresh managers before startup."""
    tools = getattr(agent, "memory_read_tools", None)
    manager = agent._memory_manager
    if tools is None or manager is None:
        return
    agent._memory_read_spawn_denials = (
        frozenset(agent.disabled_toolsets or ()) if getattr(agent, "memory_read_grant", False) is True else None
    )
    if manager is handed_in:
        logger.warning("memory_read_tools set with a shared memory_manager; memory provider detached")
        agent._memory_manager = None
    else:
        manager.restrict_to_read_only(tools)


def initialize_memory_provider(agent, provider, **kwargs) -> bool:
    """Register, restrict, then start a fresh manager; keep startup policy out of agent_init."""
    manager = agent._memory_manager
    if provider is not None:
        manager.add_provider(provider)
    apply_memory_read_policy(agent)
    return bool(manager.providers) and manager.initialize_for_agent(say=agent._emit_startup_warning, **kwargs)


def retain_child_memory_toolset(parent, requested, enabled):
    """Keep a selected provider toolset after delegation strips the built-in write tool."""
    selected = requested if requested else getattr(parent, "enabled_toolsets", None)
    if ("memory" not in enabled and memory_provider_tools_exposed(parent)
            and memory_provider_tools_enabled(selected)):
        return [*enabled, "memory"]
    return enabled


def child_memory_identity(parent):
    """Copy memory-scoping users, never the parent's gateway routing or display identity."""
    return {name: getattr(parent, f"_{name}", None) for name in ("user_id", "user_id_alt")}

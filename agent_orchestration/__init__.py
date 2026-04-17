"""
Agent Orchestration Plugin
============================

Multi-agent orchestration for Hermes — spawn, communicate, and coordinate
sub-agents via workflow patterns (parallel, sequential, map_reduce, dag).

Ported from Claude Code's agent management patterns as an independent plugin.

Usage:
  This module is loaded by Hermes' plugin system.  The plugin manager calls
  ``register(ctx)`` which wires up tools and hooks via PluginContext.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Plugin entry point — called by Hermes PluginManager.

    Registers:
      - 4 tools: spawn_agent, agent_status, send_agent_message, orchestrate_task
      - 3 hooks: on_session_start, on_session_end, pre_tool_call
    """
    from agent_orchestration.config import load_orchestration_config

    config = load_orchestration_config()
    if not config.get("enabled", True):
        logger.info("Agent orchestration plugin disabled via config")
        return

    from agent_orchestration.tools import register_all as register_tools
    from agent_orchestration.hooks import register_all as register_hooks

    register_tools(ctx)
    register_hooks(ctx)

    logger.info(
        "Agent orchestration plugin loaded: 4 tools, 3 hooks "
        "(max_agents=%d)",
        config.get("max_agents", 5),
    )

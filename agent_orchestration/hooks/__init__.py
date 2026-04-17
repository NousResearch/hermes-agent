"""
Agent Orchestration Lifecycle Hooks
=====================================

Registered hooks:

  - on_session_start: Initialize AgentManager + MailboxHub
  - on_session_end:   Cleanup all agents and mailboxes
  - pre_tool_call:    Permission checks for orchestration-managed agents
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def on_session_start(**kwargs) -> None:
    """Initialize orchestration components when a new session begins."""
    from agent_orchestration.manager import get_manager
    from agent_orchestration.mailbox import MailboxHub, get_mailbox_hub, set_mailbox_hub
    from agent_orchestration.config import load_orchestration_config

    config = load_orchestration_config()
    if not config.get("enabled", True):
        logger.debug("Agent orchestration disabled in config")
        return

    # Initialize manager
    manager = get_manager()

    # If the hook receives an agent reference, wire it up
    agent = kwargs.get("agent")
    if agent is not None:
        manager.set_parent_agent(agent)

    # Initialize mailbox hub
    hub = MailboxHub(
        mailbox_dir=config.get("mailbox_dir", ""),
        session_id=getattr(agent, "session_id", "") if agent else "",
    )
    set_mailbox_hub(hub)

    logger.info(
        "Agent orchestration initialized (max_agents=%d)",
        config.get("max_agents", 5),
    )


def on_session_end(**kwargs) -> None:
    """Cleanup all orchestration resources when session ends."""
    from agent_orchestration.manager import reset_manager
    from agent_orchestration.mailbox import get_mailbox_hub

    # Terminate all running agents
    reset_manager()

    # Cleanup all mailboxes
    hub = get_mailbox_hub()
    hub.cleanup_all()

    logger.info("Agent orchestration cleaned up")


def pre_tool_call(**kwargs) -> Optional[Dict[str, Any]]:
    """Permission check for tools called by orchestration-managed agents.

    Returns a block directive if the tool call should be prevented:
        {"action": "block", "message": "reason"}

    Returns None to allow the call.
    """
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args", {})

    # Block orchestration tools from being called by sub-agents
    # (they should only be called by the parent agent)
    # This prevents infinite recursion
    orchestration_tools = {"spawn_agent", "agent_status", "send_agent_message", "orchestrate_task"}
    if tool_name in orchestration_tools:
        # Check if caller is a sub-agent (has _delegate_depth > 0)
        agent = kwargs.get("agent")
        if agent and getattr(agent, "_delegate_depth", 0) > 0:
            return {
                "action": "block",
                "message": f"Sub-agents cannot call orchestration tools ({tool_name})",
            }

    return None


def register_all(ctx) -> None:
    """Register all lifecycle hooks via PluginContext."""
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_end", on_session_end)
    ctx.register_hook("pre_tool_call", pre_tool_call)

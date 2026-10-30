"""Snapshot Kanban worker tool metadata during agent initialization."""

from __future__ import annotations

from typing import Any


def configure_kanban_tool_surface(agent: Any) -> None:
    """Bind terminal hand-off metadata to the tools exposed to this agent instance."""
    from tools.registry import registry

    valid_tool_names = getattr(agent, "valid_tool_names", set())
    agent.kanban_terminal_tool_names = registry.kanban_terminal_tool_names(valid_tool_names)
    agent.kanban_terminal_tool_guidance = registry.kanban_terminal_tool_guidance(valid_tool_names)

    # Kanban guidance is session-static for dispatcher-owned workers only. Profiles may expose
    # kanban_show interactively, and child/cron runs can inherit the task env without owning it.
    from agent.delegation_context import owned_kanban_task
    from agent.prompt_builder import KANBAN_GUIDANCE

    agent._kanban_worker_guidance = (
        KANBAN_GUIDANCE if owned_kanban_task() and "kanban_show" in valid_tool_names else ""
    )

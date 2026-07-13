"""Agent-local operational surface for the research-only mode route."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

TOOL_NAME = "route_research_mode"

# This immutable-by-convention schema is copied into each enabled agent once.
ROUTE_RESEARCH_MODE_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Delegate a substantial evidence-gathering or comparative analysis task "
            "to the read-only research-analysis mode and return its findings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Specific research question or evidence-gathering goal.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional relevant background for the research task.",
                },
            },
            "required": ["goal"],
            "additionalProperties": False,
        },
    },
}


def inject_research_mode_tool(tools: list, *, enabled: bool) -> list:
    """Return an agent-owned tool snapshot with the canonical local router.

    Disabled mode is an exact no-op, including object identity. Enabled mode
    copies the outer snapshot, removes every registry/plugin collision for the
    reserved name, and appends one private copy of the canonical schema.
    """
    if not enabled:
        return tools

    local_tools = [
        tool for tool in tools
        if tool.get("function", {}).get("name") != TOOL_NAME
    ]
    local_tools.append(deepcopy(ROUTE_RESEARCH_MODE_TOOL))
    return local_tools


def finalize_research_mode_tool(agent: Any) -> None:
    """Restore the canonical router after all initialization-time injections.

    Providers and context engines add schemas after the registry snapshot is
    built. In enabled mode, canonicalize that completed surface and rebuild
    validation names from it. Disabled mode deliberately touches nothing.
    """
    if not getattr(agent, "_mode_router_enabled", False):
        return

    agent.tools = inject_research_mode_tool(agent.tools, enabled=True)
    agent.valid_tool_names = {
        tool["function"]["name"] for tool in agent.tools
    }


def disable_research_mode_tool(agent: Any) -> None:
    """Disable and remove the automatic router from a leaf child agent."""
    agent._mode_router_enabled = False
    tools = getattr(agent, "tools", None)
    if isinstance(tools, list):
        agent.tools = [
            tool for tool in tools
            if tool.get("function", {}).get("name") != TOOL_NAME
        ]
    valid_tool_names = getattr(agent, "valid_tool_names", None)
    if isinstance(valid_tool_names, set):
        valid_tool_names.discard(TOOL_NAME)


def validate_arguments(arguments: dict[str, Any]) -> str | None:
    """Return a normal tool error for anything outside the fixed public contract."""
    unknown = sorted(set(arguments) - {"goal", "context"})
    goal = arguments.get("goal")
    context = arguments.get("context")
    if unknown:
        return json.dumps({"error": "Invalid tool arguments", "unknown_fields": unknown})
    if not isinstance(goal, str) or not goal.strip():
        return json.dumps({"error": "Invalid tool arguments", "message": "goal must be a non-empty string"})
    if context is not None and not isinstance(context, str):
        return json.dumps({"error": "Invalid tool arguments", "message": "context must be a string"})
    return None

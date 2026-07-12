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

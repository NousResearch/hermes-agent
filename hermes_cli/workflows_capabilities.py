"""Runtime capability registry for Hermes Workflows.

This module is the product-facing truth for what the engine/dispatcher can
execute today. `workflows_spec.py` may declare future primitives, but the
assistant and dashboard must default to this implemented subset.
"""
from __future__ import annotations

from typing import Any

DECLARED_TRIGGER_TYPES = {"manual", "schedule", "webhook", "kanban_event"}
DECLARED_NODE_TYPES = {
    "pass",
    "switch",
    "agent_task",
    "wait",
    "parallel",
    "join",
    "send_message",
    "fail",
    "subworkflow",
}

IMPLEMENTED_TRIGGER_TYPES = {"manual", "schedule"}
IMPLEMENTED_NODE_TYPES = {
    "pass",
    "switch",
    "agent_task",
    "wait",
    "parallel",
    "join",
    "fail",
}

UNSUPPORTED_TRIGGER_TYPES = DECLARED_TRIGGER_TYPES - IMPLEMENTED_TRIGGER_TYPES
UNSUPPORTED_NODE_TYPES = DECLARED_NODE_TYPES - IMPLEMENTED_NODE_TYPES


def workflow_capabilities() -> dict[str, Any]:
    """Return stable capability metadata for API/tool/dashboard consumers."""
    return {
        "triggers": {
            "declared": sorted(DECLARED_TRIGGER_TYPES),
            "implemented": sorted(IMPLEMENTED_TRIGGER_TYPES),
            "unsupported": sorted(UNSUPPORTED_TRIGGER_TYPES),
        },
        "nodes": {
            "declared": sorted(DECLARED_NODE_TYPES),
            "implemented": sorted(IMPLEMENTED_NODE_TYPES),
            "unsupported": sorted(UNSUPPORTED_NODE_TYPES),
        },
        "assistant": {
            "allowed_triggers": sorted(IMPLEMENTED_TRIGGER_TYPES),
            "allowed_nodes": sorted(IMPLEMENTED_NODE_TYPES),
        },
    }

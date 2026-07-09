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


def implemented_primitive_errors(spec: Any) -> list[str]:
    """Return errors for workflow primitives not implemented by today's dispatcher."""
    errors: list[str] = []
    for trigger in getattr(spec, "triggers", []) or []:
        trigger_type = getattr(trigger, "type", None)
        if trigger_type not in IMPLEMENTED_TRIGGER_TYPES:
            errors.append(f"unsupported trigger type: {trigger_type}")

    nodes = getattr(spec, "nodes", {}) or {}
    for node_id, node in nodes.items():
        node_type = getattr(node, "type", None)
        if node_type not in IMPLEMENTED_NODE_TYPES:
            errors.append(f"unsupported node type: {node_type} on node {node_id}")
        # `workspace` (cwd/env) is declared for forward compatibility but no
        # runtime consumes it yet — reject it so user intent never silently
        # no-ops. The implemented per-node knobs are workspace_kind/workspace_path.
        if getattr(node, "workspace", None) is not None:
            errors.append(
                f"unsupported node field: workspace on node {node_id} "
                "(not implemented; use workspace_kind/workspace_path)"
            )
    return errors


def require_implemented_primitives(spec: Any) -> None:
    """Raise ValueError when a spec uses primitives unavailable at runtime."""
    errors = implemented_primitive_errors(spec)
    if errors:
        raise ValueError("; ".join(errors))

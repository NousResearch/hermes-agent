"""Pure in-memory workflow graph runner."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from hermes_cli.workflows_expr import eval_condition, resolve_path
from hermes_cli.workflows_spec import EdgeSpec, WorkflowSpec, validate_graph

_TEMPLATE_RE = re.compile(r"^\$\{\s*([^}]+?)\s*\}$")
_WAITING_NODE_TYPES = {
    "agent_task",
    "wait",
    "parallel",
    "join",
    "send_message",
    "subworkflow",
}


@dataclass
class EngineResult:
    status: Literal["succeeded", "waiting", "failed"]
    context: dict[str, Any]
    waiting_nodes: list[str]
    error: dict[str, Any] | None = None


def initial_context(input_data: dict[str, Any], spec: WorkflowSpec) -> dict[str, Any]:
    return {
        "input": input_data,
        "workflow": {"id": spec.id, "version": spec.version},
        "node": {},
    }


def render_template(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        match = _TEMPLATE_RE.match(value)
        if not match:
            return value
        path = match.group(1).strip()
        if not path.startswith("$."):
            path = f"$.{path}"
        return resolve_path(context, path)
    if isinstance(value, list):
        return [render_template(item, context) for item in value]
    if isinstance(value, dict):
        return {key: render_template(item, context) for key, item in value.items()}
    return value


def next_edges(spec: WorkflowSpec, node_id: str, port: str | None = None) -> list[EdgeSpec]:
    source = node_id if port is None else f"{node_id}.{port}"
    return [edge for edge in spec.edges if edge.from_ == source]


def _initial_nodes(spec: WorkflowSpec) -> list[str]:
    incoming = {edge.to for edge in spec.edges}
    incoming.update(node.default for node in spec.nodes.values() if node.default)
    incoming.update(node.catch for node in spec.nodes.values() if node.catch)
    return [node_id for node_id in spec.nodes if node_id not in incoming]


def _switch_port(node_id: str, cases: list[Any], context: dict[str, Any]) -> str:
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError(f"switch case for {node_id} must be a mapping")
        name = case.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"switch case for {node_id} requires name")
        if eval_condition(case.get("when"), context):
            return name
    return "default"


def run_in_memory_until_waiting(
    spec: WorkflowSpec,
    input_data: dict[str, Any],
    completed_wait_nodes: set[str] | None = None,
    completed_node_outputs: dict[str, Any] | None = None,
    catch_failed_nodes: set[str] | None = None,
    error_context: dict[str, Any] | None = None,
) -> EngineResult:
    validate_graph(spec)
    completed_wait_nodes = completed_wait_nodes or set()
    completed_node_outputs = completed_node_outputs or {}
    catch_failed_nodes = catch_failed_nodes or set()
    context = initial_context(input_data, spec)
    runnable = deque(_initial_nodes(spec))
    # ponytail: cheap cycle guard; real scheduler can track runs.
    max_steps = max(1, len(spec.nodes) * 10)
    steps = 0

    while runnable:
        steps += 1
        if steps > max_steps:
            return EngineResult(
                status="failed",
                context=context,
                waiting_nodes=[],
                error={"message": "workflow exceeded max in-memory steps"},
            )

        node_id = runnable.popleft()
        node = spec.nodes[node_id]

        if node_id in completed_node_outputs:
            context["node"][node_id] = {"output": completed_node_outputs[node_id]}
            runnable.extend(edge.to for edge in next_edges(spec, node_id))
            continue

        if node.type == "pass":
            context["node"][node_id] = {"output": render_template(node.output, context)}
            runnable.extend(edge.to for edge in next_edges(spec, node_id))
            continue

        if node.type == "switch":
            port = _switch_port(node_id, node.cases, context)
            edges = next_edges(spec, node_id, port)
            if edges:
                runnable.extend(edge.to for edge in edges)
            elif port == "default" and node.default:
                runnable.append(node.default)
            continue

        if node.type == "fail":
            error = {"node": node_id, "type": "fail", "output": render_template(node.output, context)}
            if node_id in catch_failed_nodes and node.catch:
                context["error"] = error_context or error
                runnable.append(node.catch)
                continue
            return EngineResult(
                status="failed",
                context=context,
                waiting_nodes=[],
                error=error,
            )

        if node.type == "wait" and node_id in completed_wait_nodes:
            context["node"][node_id] = {"output": {"waited": True}}
            runnable.extend(edge.to for edge in next_edges(spec, node_id))
            continue

        if node.type in _WAITING_NODE_TYPES:
            return EngineResult(status="waiting", context=context, waiting_nodes=[node_id])

        return EngineResult(
            status="failed",
            context=context,
            waiting_nodes=[],
            error={"node": node_id, "message": f"unsupported node type: {node.type}"},
        )

    return EngineResult(status="succeeded", context=context, waiting_nodes=[])

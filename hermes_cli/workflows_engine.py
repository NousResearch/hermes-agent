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


def _can_reach(spec: WorkflowSpec, start_id: str, target_id: str) -> bool:
    stack = [start_id]
    seen: set[str] = set()
    while stack:
        node_id = stack.pop()
        if node_id == target_id:
            return True
        if node_id in seen:
            continue
        seen.add(node_id)
        node = spec.nodes[node_id]
        for edge in spec.edges:
            if edge.from_.split(".", 1)[0] == node_id:
                stack.append(edge.to)
        if node.default:
            stack.append(node.default)
        if node.catch:
            stack.append(node.catch)
    return False


def _parallel_branches_reaching_join(
    spec: WorkflowSpec,
    parallel_id: str,
    join_id: str,
) -> set[str]:
    branches: set[str] = set()
    for edge in spec.edges:
        source_base, _, branch = edge.from_.partition(".")
        if source_base == parallel_id and branch and _can_reach(spec, edge.to, join_id):
            branches.add(branch)
    return branches


def _record_branch_output(
    context: dict[str, Any],
    completed_branch_by_node: dict[str, tuple[str, str]],
    node_id: str,
    branch_key: tuple[str, str] | None,
) -> None:
    if branch_key is None:
        return
    parallel_id, branch = branch_key
    output = context["node"].get(node_id, {}).get("output")
    context.setdefault("branches", {}).setdefault(parallel_id, {})[branch] = output
    completed_branch_by_node[node_id] = branch_key


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
    runnable: deque[tuple[str, tuple[str, str] | None]] = deque()
    max_steps = spec.max_node_runs
    steps = 0
    waiting_nodes: list[str] = []
    scheduled_branch_by_node: dict[str, tuple[str, str] | None] = {}
    completed_branch_by_node: dict[str, tuple[str, str]] = {}
    join_branch_cache: dict[tuple[str, str], set[str]] = {}

    def enqueue(node_id: str, branch_key: tuple[str, str] | None) -> None:
        queued_branch = None if spec.nodes[node_id].type == "join" else branch_key
        scheduled_branch_by_node.setdefault(node_id, queued_branch)
        runnable.append((node_id, queued_branch))

    for node_id in _initial_nodes(spec):
        enqueue(node_id, None)

    while runnable:
        steps += 1
        if steps > max_steps:
            return EngineResult(
                status="failed",
                context=context,
                waiting_nodes=[],
                error={"message": "workflow exceeded max node runs"},
            )

        node_id, branch_key = runnable.popleft()
        node = spec.nodes[node_id]

        if node.type == "join" and node_id in context["node"]:
            continue

        if node_id in completed_node_outputs:
            context["node"][node_id] = {"output": completed_node_outputs[node_id]}
            _record_branch_output(context, completed_branch_by_node, node_id, branch_key)
            for edge in next_edges(spec, node_id):
                enqueue(edge.to, branch_key)
            continue

        if node.type == "pass":
            context["node"][node_id] = {"output": render_template(node.output, context)}
            _record_branch_output(context, completed_branch_by_node, node_id, branch_key)
            for edge in next_edges(spec, node_id):
                enqueue(edge.to, branch_key)
            continue

        if node.type == "switch":
            port = _switch_port(node_id, node.cases, context)
            edges = next_edges(spec, node_id, port)
            if edges:
                for edge in edges:
                    enqueue(edge.to, branch_key)
            elif port == "default" and node.default:
                enqueue(node.default, branch_key)
            continue

        if node.type == "parallel":
            branch_edges = [
                (edge.from_.split(".", 1)[1], edge)
                for edge in spec.edges
                if "." in edge.from_ and edge.from_.split(".", 1)[0] == node_id
            ]
            context.setdefault("branches", {}).setdefault(node_id, {})
            context["node"][node_id] = {"output": {"branches": [branch for branch, _ in branch_edges]}}
            for branch, edge in branch_edges:
                enqueue(edge.to, (node_id, branch))
            continue

        if node.type == "join":
            incoming = [edge for edge in spec.edges if edge.to == node_id]
            branches = {}
            expected_labels: set[str] = set()
            active_parallel_ids: set[str] = set()
            for edge in incoming:
                source_base, _, port = edge.from_.partition(".")
                if source_base not in scheduled_branch_by_node and source_base not in context["node"]:
                    continue
                owner = completed_branch_by_node.get(source_base)
                if owner is None:
                    owner = scheduled_branch_by_node.get(source_base)
                if owner:
                    active_parallel_ids.add(owner[0])
                label = owner[1] if owner else (port or source_base)
                expected_labels.add(label)
                node_context = context["node"].get(source_base, {})
                if source_base in context["node"]:
                    output = node_context.get("output") if isinstance(node_context, dict) else None
                    branches[label] = output
            for parallel_id in active_parallel_ids:
                cache_key = (parallel_id, node_id)
                if cache_key not in join_branch_cache:
                    join_branch_cache[cache_key] = _parallel_branches_reaching_join(
                        spec, parallel_id, node_id
                    )
                expected_labels.update(join_branch_cache[cache_key])
            if expected_labels - branches.keys():
                if runnable:
                    enqueue(node_id, branch_key)
                elif not waiting_nodes:
                    return EngineResult(status="waiting", context=context, waiting_nodes=[node_id])
                continue
            context["node"][node_id] = {"output": {"branches": branches}}
            for edge in next_edges(spec, node_id):
                enqueue(edge.to, None)
            continue

        if node.type == "fail":
            error = {"node": node_id, "type": "fail", "output": render_template(node.output, context)}
            if node_id in catch_failed_nodes and node.catch:
                context["error"] = error_context or error
                enqueue(node.catch, branch_key)
                continue
            return EngineResult(
                status="failed",
                context=context,
                waiting_nodes=[],
                error=error,
            )

        if node.type == "wait" and node_id in completed_wait_nodes:
            context["node"][node_id] = {"output": {"waited": True}}
            _record_branch_output(context, completed_branch_by_node, node_id, branch_key)
            for edge in next_edges(spec, node_id):
                enqueue(edge.to, branch_key)
            continue

        if node.type in _WAITING_NODE_TYPES:
            if node_id not in waiting_nodes:
                waiting_nodes.append(node_id)
            continue

        return EngineResult(
            status="failed",
            context=context,
            waiting_nodes=[],
            error={"node": node_id, "message": f"unsupported node type: {node.type}"},
        )

    if waiting_nodes:
        return EngineResult(status="waiting", context=context, waiting_nodes=waiting_nodes)
    return EngineResult(status="succeeded", context=context, waiting_nodes=[])

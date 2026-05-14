"""Deterministic workflow status reports for no-agent/cron use."""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict, deque
from typing import Any

from .api import get_workflow_dag

_TERMINAL_GATE_STATUSES = {"approved", "rejected", "skipped"}
_NODE_COUNT_ORDER = ("done", "running", "blocked", "waiting")


def workflow_status_report(conn: sqlite3.Connection, workflow_id: str) -> dict[str, Any]:
    """Return deterministic status facts for a workflow without LLM synthesis."""

    dag_payload = get_workflow_dag(conn, workflow_id)
    dag_facts = dag_payload["facts"]
    nodes = dag_facts["nodes"]
    gates = dag_facts["gates"]
    facts = {
        "workflow": dag_facts["workflow"],
        "nodeCounts": _node_counts(nodes),
        "criticalPath": _critical_path(nodes, dag_facts["edges"]),
        "humanActionRequired": _human_actions(gates),
        "controlActions": dag_facts.get("controlActions", []),
    }
    return {"facts": facts, "insights": None}


def render_workflow_status_report(report: dict[str, Any]) -> str:
    """Render a stable plain-text workflow status report."""

    facts = report["facts"]
    workflow = facts["workflow"]
    counts = facts["nodeCounts"]
    lines = [
        f"Workflow: {workflow['id']} {workflow.get('title') or ''}".rstrip(),
        f"State: {workflow.get('status') or 'unknown'}",
        _render_node_counts(counts),
        "Critical path: " + (" → ".join(facts.get("criticalPath") or []) or "unknown"),
        "Human action required: " + _render_human_actions(facts.get("humanActionRequired") or []),
    ]
    return "\n".join(lines)


def _node_counts(nodes: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter((node.get("status") or "unknown") for node in nodes)
    ordered: dict[str, int] = {"total": len(nodes)}
    for status in _NODE_COUNT_ORDER:
        if status in counts:
            ordered[status] = counts[status]
    for status in sorted(counts):
        if status not in ordered:
            ordered[status] = counts[status]
    return ordered


def _critical_path(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    if not nodes:
        return []
    node_ids = [node["id"] for node in nodes]
    node_set = set(node_ids)
    children: dict[str, list[str]] = defaultdict(list)
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in node_set and target in node_set:
            children[source].append(target)
            indegree[target] += 1
    for child_list in children.values():
        child_list.sort()

    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    best_path = {node_id: [node_id] for node_id in node_ids}
    while queue:
        node_id = queue.popleft()
        for child in children[node_id]:
            candidate = [*best_path[node_id], child]
            if len(candidate) > len(best_path[child]) or (len(candidate) == len(best_path[child]) and candidate < best_path[child]):
                best_path[child] = candidate
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    return max(best_path.values(), key=lambda path: (len(path), tuple(reversed(path))))


def _human_actions(gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for gate in gates:
        if gate.get("status") in _TERMINAL_GATE_STATUSES:
            continue
        actions.append(
            {
                "gateId": gate.get("id"),
                "nodeId": gate.get("nodeId"),
                "gateType": gate.get("gateType"),
                "requiredActor": gate.get("requiredActor"),
                "reason": gate.get("reason"),
            }
        )
    return actions


def _render_node_counts(counts: dict[str, int]) -> str:
    total = counts.get("total", 0)
    parts = [f"{total} total"]
    for status, count in counts.items():
        if status == "total":
            continue
        parts.append(f"{count} {status}")
    return "Nodes: " + ", ".join(parts)


def _render_human_actions(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "none"
    rendered = []
    for action in actions:
        reason = action.get("reason") or "No reason recorded."
        rendered.append(
            f"{action.get('gateId')} on {action.get('nodeId')} ({action.get('requiredActor')}) — {reason}"
        )
    return "; ".join(rendered)

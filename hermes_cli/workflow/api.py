"""Read-only serialization helpers for persisted workflow state."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .store import get_workflow, list_artifacts, list_gates, list_workflows


def list_workflow_summaries(
    conn: sqlite3.Connection,
    *,
    board: str | None = None,
    status: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    _validate_limit(limit)
    workflows = list_workflows(conn, board=board, status=status, limit=limit)
    return _response({"workflows": [workflow.to_dict() for workflow in workflows], "count": len(workflows)})


def get_workflow_dag(conn: sqlite3.Connection, workflow_id: str) -> dict[str, Any]:
    workflow = _require_workflow(conn, workflow_id)
    nodes = _node_rows(conn, workflow_id)
    edges = _edge_rows(conn, workflow_id)
    parent_map, child_map = _adjacency(edges)
    facts = {
        "workflow": workflow.to_dict(),
        "nodes": [_serialize_node(node, parent_map, child_map) for node in nodes],
        "edges": [_serialize_edge(edge) for edge in edges],
        "gates": [gate.to_dict() for gate in list_gates(conn, workflow_id)],
        "artifacts": [artifact.to_dict() for artifact in list_artifacts(conn, workflow_id)],
    }
    return _response(facts)


def get_workflow_node(conn: sqlite3.Connection, workflow_id: str, node_id: str) -> dict[str, Any]:
    _require_workflow(conn, workflow_id)
    edges = _edge_rows(conn, workflow_id)
    parent_map, child_map = _adjacency(edges)
    row = conn.execute(
        "SELECT * FROM workflow_nodes WHERE workflow_id = ? AND node_id = ?",
        (workflow_id, node_id),
    ).fetchone()
    if row is None:
        raise ValueError(f"workflow node not found: {node_id}")
    artifact_ids = _artifact_ids_for_node(conn, workflow_id, node_id)
    facts = {
        "workflowId": workflow_id,
        "node": _serialize_node(row, parent_map, child_map),
        "gates": _node_gates(conn, workflow_id, node_id),
        "events": _node_events(conn, workflow_id, node_id),
        "artifacts": [
            artifact.to_dict()
            for artifact in list_artifacts(conn, workflow_id)
            if artifact.id in artifact_ids
        ],
    }
    return _response(facts)


def get_workflow_events(conn: sqlite3.Connection, workflow_id: str, *, limit: int = 100) -> dict[str, Any]:
    _validate_limit(limit)
    _require_workflow(conn, workflow_id)
    rows = conn.execute(
        "SELECT * FROM workflow_events WHERE workflow_id = ? ORDER BY created_at DESC, id DESC LIMIT ?",
        (workflow_id, limit),
    ).fetchall()
    return _response({"workflowId": workflow_id, "events": [_serialize_event(row) for row in rows]})


def get_workflow_artifacts(
    conn: sqlite3.Connection,
    workflow_id: str,
    *,
    kind: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    _validate_limit(limit)
    _require_workflow(conn, workflow_id)
    artifacts = list_artifacts(conn, workflow_id, kind=kind, limit=limit)
    return _response({"workflowId": workflow_id, "artifacts": [artifact.to_dict() for artifact in artifacts]})


def _response(facts: dict[str, Any]) -> dict[str, Any]:
    return {"facts": facts, "insights": None}


def _validate_limit(limit: int) -> None:
    if limit < 1:
        raise ValueError("limit must be positive")


def _require_workflow(conn: sqlite3.Connection, workflow_id: str):
    workflow = get_workflow(conn, workflow_id)
    if workflow is None:
        raise ValueError(f"workflow not found: {workflow_id}")
    return workflow


def _node_rows(conn: sqlite3.Connection, workflow_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM workflow_nodes WHERE workflow_id = ? ORDER BY created_at ASC, node_id ASC",
        (workflow_id,),
    ).fetchall()


def _edge_rows(conn: sqlite3.Connection, workflow_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM workflow_edges WHERE workflow_id = ? ORDER BY parent_node_id ASC, child_node_id ASC",
        (workflow_id,),
    ).fetchall()


def _adjacency(edges: list[sqlite3.Row]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    parent_map: dict[str, list[str]] = {}
    child_map: dict[str, list[str]] = {}
    for edge in edges:
        parent = edge["parent_node_id"]
        child = edge["child_node_id"]
        parent_map.setdefault(child, []).append(parent)
        child_map.setdefault(parent, []).append(child)
    return parent_map, child_map


def _serialize_edge(row: sqlite3.Row) -> dict[str, Any]:
    return {"source": row["parent_node_id"], "target": row["child_node_id"], "kind": row["kind"]}


def _serialize_event(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "workflowId": row["workflow_id"],
        "nodeId": row["node_id"],
        "eventType": row["event_type"],
        "actorType": row["actor_type"],
        "actorId": row["actor_id"],
        "message": row["message"],
        "data": _loads(row["data_json"], {}),
        "createdAt": row["created_at"],
    }


def _serialize_node(row: sqlite3.Row, parent_map: dict[str, list[str]], child_map: dict[str, list[str]]) -> dict[str, Any]:
    node_id = row["node_id"]
    return {
        "id": node_id,
        "title": row["title"],
        "role": row["role"],
        "profile": row["profile"],
        "status": row["status"],
        "parents": parent_map.get(node_id, []),
        "children": child_map.get(node_id, []),
        "gateLevel": row["gate_level"],
        "gateType": row["gate_type"],
        "kanbanTaskId": row["kanban_task_id"],
        "workspace": {
            "kind": "worktree" if row["worktree_path"] else "scratch",
            "branch": row["branch"],
            "worktreePath": row["worktree_path"],
            "baseRef": row["base_ref"],
        },
        "definitionOfDone": _loads(row["definition_of_done_json"], []),
        "scope": _loads(row["scope_json"], {}),
        "evidence": _loads(row["evidence_json"], {}),
        "metadata": _loads(row["metadata_json"], {}),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _artifact_ids_for_node(conn: sqlite3.Connection, workflow_id: str, node_id: str) -> set[str]:
    rows = conn.execute(
        """
        SELECT artifact_id FROM workflow_gates
        WHERE workflow_id = ? AND node_id = ? AND artifact_id IS NOT NULL
        """,
        (workflow_id, node_id),
    ).fetchall()
    return {row["artifact_id"] for row in rows}


def _node_gates(conn: sqlite3.Connection, workflow_id: str, node_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
    _validate_limit(limit)
    rows = conn.execute(
        """
        SELECT * FROM workflow_gates
        WHERE workflow_id = ? AND node_id = ?
        ORDER BY level ASC, id ASC
        LIMIT ?
        """,
        (workflow_id, node_id, limit),
    ).fetchall()
    return [_serialize_gate(row) for row in rows]


def _node_events(conn: sqlite3.Connection, workflow_id: str, node_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
    _validate_limit(limit)
    rows = conn.execute(
        """
        SELECT * FROM workflow_events
        WHERE workflow_id = ? AND node_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (workflow_id, node_id, limit),
    ).fetchall()
    return [_serialize_event(row) for row in rows]


def _serialize_gate(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "workflowId": row["workflow_id"],
        "nodeId": row["node_id"],
        "gateType": row["gate_type"],
        "level": row["level"],
        "status": row["status"],
        "requiredActor": row["required_actor"],
        "verdict": row["verdict"],
        "resolvedBy": row["resolved_by"],
        "artifactId": row["artifact_id"],
        "metadata": _loads(row["metadata_json"], {}),
        "createdAt": row["created_at"],
        "resolvedAt": row["resolved_at"],
    }


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    return json.loads(raw)

"""Read-only serialization helpers for persisted workflow state."""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from .dag import normalize_dag
from .materialize import materialize_workflow
from .policy import DEFAULT_POLICY
from .store import (
    add_event,
    add_gate,
    create_workflow,
    get_inbox_item,
    get_workflow,
    list_artifacts,
    list_gates,
    list_inbox_items,
    list_workflows,
    resolve_gate,
    save_dag,
    update_inbox_item,
    update_workflow_status,
)


def list_inbox_item_summaries(
    conn: sqlite3.Connection,
    *,
    status: str | None = None,
    source: str | None = None,
    classification: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    _validate_limit(limit)
    items = list_inbox_items(conn, status=status, source=source, classification=classification, limit=limit)
    return _response({"inboxItems": [item.to_dict() for item in items], "count": len(items)})


def get_inbox_item_detail(conn: sqlite3.Connection, inbox_item_id: str) -> dict[str, Any]:
    item = get_inbox_item(conn, inbox_item_id)
    if item is None:
        raise ValueError(f"workflow inbox item not found: {inbox_item_id}")
    return _response({"inboxItem": item.to_dict()})


def update_inbox_item_triage(
    conn: sqlite3.Connection,
    inbox_item_id: str,
    *,
    status: str | None = None,
    classification: str | None = None,
    workspace_path: str | None = None,
    assigned_workflow_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    updates: dict[str, Any] = {"status": status if status is not None else "triaged", "metadata": metadata, "now": now}
    if classification is not None:
        updates["classification"] = classification
    if workspace_path is not None:
        updates["workspace_path"] = workspace_path
    if assigned_workflow_id is not None:
        updates["assigned_workflow_id"] = assigned_workflow_id
    item = update_inbox_item(conn, inbox_item_id, **updates)
    if item is None:
        raise ValueError(f"workflow inbox item not found: {inbox_item_id}")
    return _response({"inboxItem": item.to_dict()})


def shape_inbox_item_as_draft_workflow(
    conn: sqlite3.Connection,
    inbox_item_id: str,
    *,
    workflow_id: str | None = None,
    title: str | None = None,
    description: str | None = None,
    board: str = "default",
    scale: str = "medium",
    user_intent: str | None = None,
    profile_hints: dict[str, str] | None = None,
) -> dict[str, Any]:
    item = get_inbox_item(conn, inbox_item_id)
    if item is None:
        raise ValueError(f"workflow inbox item not found: {inbox_item_id}")

    resolved_workflow_id = (workflow_id or item.assigned_workflow_id or _workflow_id_from_inbox_item_id(inbox_item_id)).strip()
    resolved_title = (title or item.title or resolved_workflow_id).strip()
    resolved_description = description if description is not None else item.body
    body = item.body or "Shape and implement the selected inbox item."
    resolved_user_intent = _clean_optional_text(user_intent)
    resolved_profile_hints = _clean_profile_hints(profile_hints)
    shape_context = resolved_user_intent or body
    draft_dag = {
        "schema_version": 1,
        "workflow_id": resolved_workflow_id,
        "name": resolved_title,
        "scale": scale,
        "nodes": _draft_nodes_for_shape(
            resolved_title,
            body=body,
            shape_context=shape_context,
            scale=scale,
            profile_hints=resolved_profile_hints,
        ),
    }
    normalized = normalize_dag(draft_dag, policy=DEFAULT_POLICY)
    if not normalized.ok or normalized.dag is None:
        first = normalized.errors[0].message if normalized.errors else "unknown validation error"
        raise ValueError(f"workflow draft DAG is invalid: {first}")
    return _response(
        {
            "inboxItem": item.to_dict(),
            "draftWorkflow": {
                "id": resolved_workflow_id,
                "title": resolved_title,
                "description": resolved_description,
                "workspacePath": item.workspace_path,
                "board": board,
                "scale": scale,
                "sourceInboxItemId": inbox_item_id,
                "shapeIntent": {
                    "userIntent": resolved_user_intent,
                    "profileHints": resolved_profile_hints,
                },
            },
            "draftDag": normalized.dag,
        }
    )


def _draft_nodes_for_shape(
    resolved_title: str,
    *,
    body: str,
    shape_context: str,
    scale: str,
    profile_hints: dict[str, str],
) -> list[dict[str, Any]]:
    planner_profile = _profile_for_role("planner", profile_hints)
    engineer_profile = _profile_for_role("engineer", profile_hints)
    nodes: list[dict[str, Any]] = [
        {
            "id": "shape-plan",
            "title": "Shape plan",
            "role": "planner",
            "profile": planner_profile,
            "scope": {"summary": f"Shape inbox request: {shape_context}"},
        },
    ]
    if scale not in {"large", "xl"}:
        nodes.append(
            {
                "id": "build-slice",
                "title": "Build first slice",
                "role": "engineer",
                "profile": engineer_profile,
                "parents": ["shape-plan"],
                "definition_of_done": ["Targeted tests pass."],
                "scope": {"summary": f"Implement the first useful slice for: {resolved_title}"},
            }
        )
        return nodes

    nodes.extend(
        [
            {
                "id": "design-architecture",
                "title": "Design architecture",
                "role": "architect",
                "profile": _profile_for_role("architect", profile_hints),
                "parents": ["shape-plan"],
                "scope": {"summary": f"Design the workflow architecture for: {shape_context}"},
            },
            {
                "id": "build-foundation",
                "title": "Build foundation",
                "role": "engineer",
                "profile": engineer_profile,
                "parents": ["design-architecture"],
                "definition_of_done": ["Targeted tests pass.", "Core behavior is covered by regression tests."],
                "scope": {"summary": f"Build the foundation for: {resolved_title}"},
            },
            {
                "id": "build-integration",
                "title": "Build integration",
                "role": "engineer",
                "profile": engineer_profile,
                "parents": ["design-architecture"],
                "definition_of_done": ["Targeted tests pass.", "Integration behavior is covered by regression tests."],
                "scope": {"summary": f"Build the integration path for: {resolved_title}"},
            },
            {
                "id": "integrate-workflow",
                "title": "Integrate workflow",
                "role": "integrator",
                "profile": _profile_for_role("integrator", profile_hints),
                "parents": ["build-foundation", "build-integration"],
                "scope": {"summary": f"Integrate and verify the shaped workflow: {body}"},
            },
        ]
    )
    return nodes


def _clean_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _clean_profile_hints(profile_hints: dict[str, str] | None) -> dict[str, str]:
    if not isinstance(profile_hints, dict):
        return {}
    cleaned: dict[str, str] = {}
    for role, profile in profile_hints.items():
        role_name = str(role).strip()
        profile_name = str(profile).strip()
        if role_name and profile_name:
            cleaned[role_name] = profile_name
    return cleaned


def _profile_for_role(role: str, profile_hints: dict[str, str]) -> str:
    return profile_hints.get(role) or role


def promote_inbox_item_to_workflow(
    conn: sqlite3.Connection,
    inbox_item_id: str,
    *,
    workflow_id: str,
    title: str,
    description: str = "",
    board: str = "default",
    scale: str = "medium",
    draft_dag: dict[str, Any],
    workspace_path: str | None = None,
    actor_id: str = "webui",
    now: float | None = None,
) -> dict[str, Any]:
    item = get_inbox_item(conn, inbox_item_id)
    if item is None:
        raise ValueError(f"workflow inbox item not found: {inbox_item_id}")
    if item.status == "promoted":
        raise ValueError(f"workflow inbox item already promoted: {inbox_item_id}")
    normalized = normalize_dag(draft_dag, policy=DEFAULT_POLICY)
    if not normalized.ok or normalized.dag is None:
        first = normalized.errors[0].message if normalized.errors else "unknown validation error"
        raise ValueError(f"workflow draft DAG is invalid: {first}")
    if normalized.dag.get("workflow_id") != workflow_id:
        raise ValueError("workflow draft DAG workflow_id must match promoted workflow")

    resolved_workspace_path = workspace_path if workspace_path is not None else item.workspace_path
    workflow = create_workflow(
        conn,
        workflow_id=workflow_id,
        title=title,
        description=description,
        workspace_path=resolved_workspace_path,
        board=board,
        scale=scale,
        status="dag_draft",
        created_by=actor_id,
        metadata={"sourceInboxItemId": inbox_item_id},
        now=now,
    )
    save_dag(conn, workflow_id=workflow_id, normalized_dag=normalized.dag, now=now)
    updated_item = update_inbox_item(
        conn,
        inbox_item_id,
        status="promoted",
        assigned_workflow_id=workflow_id,
        metadata={"promotedBy": actor_id},
        now=now,
    )
    if updated_item is None:
        raise ValueError(f"workflow inbox item not found: {inbox_item_id}")
    add_event(
        conn,
        workflow_id=workflow_id,
        event_type="inbox_promoted",
        actor_type="human" if actor_id == "webui" else "agent",
        actor_id=actor_id,
        message=f"Inbox item {inbox_item_id} promoted to draft workflow",
        data={"inboxItemId": inbox_item_id},
        now=now,
    )
    return _response({"workflow": workflow.to_dict(), "inboxItem": updated_item.to_dict(), "dag": normalized.dag})


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
    facts["controlActions"] = _control_actions(workflow_id, facts["workflow"], facts["gates"])
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
    events = [_serialize_event(row) for row in rows]
    return _response({"workflowId": workflow_id, "events": events, "count": len(events)})


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
    artifact_rows = [artifact.to_dict() for artifact in artifacts]
    return _response({"workflowId": workflow_id, "artifacts": artifact_rows, "count": len(artifact_rows)})


def materialize_workflow_to_kanban(
    conn: sqlite3.Connection,
    workflow_id: str,
    *,
    kanban_conn: sqlite3.Connection | None = None,
    actor_id: str = "webui",
    now: float | None = None,
) -> dict[str, Any]:
    result = materialize_workflow(conn, workflow_id, kanban_conn=kanban_conn, actor_id=actor_id, now=now)
    return _response(result.to_dict())


def resolve_workflow_gate_control(
    conn: sqlite3.Connection,
    workflow_id: str,
    gate_id: str,
    *,
    status: str,
    verdict: str | None = None,
    resolved_by: str = "webui",
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    workflow = _require_workflow(conn, workflow_id)
    gate = next((item for item in list_gates(conn, workflow_id) if item.id == gate_id), None)
    if gate is None:
        raise ValueError(f"workflow gate not found: {gate_id}")
    if gate.status in {"approved", "rejected", "skipped"}:
        raise ValueError(f"workflow gate is already resolved: {gate_id}")
    if status not in {"approved", "rejected", "skipped"}:
        raise ValueError(f"invalid gate resolution status: {status}")
    resolved = resolve_gate(
        conn,
        gate_id=gate_id,
        status=status,
        verdict=verdict or status,
        resolved_by=resolved_by,
        reason=reason,
        metadata=metadata,
        now=now,
    )
    add_event(
        conn,
        workflow_id=workflow_id,
        node_id=resolved.node_id,
        event_type="gate_resolved",
        actor_type="human" if resolved_by == "webui" else "agent",
        actor_id=resolved_by,
        message=f"Gate {gate_id} resolved as {status}",
        data={"gateId": gate_id, "status": status, "verdict": verdict or status},
        now=now,
    )
    workflow = get_workflow(conn, workflow_id) or workflow
    gates = [item.to_dict() for item in list_gates(conn, workflow_id)]
    return _response({"workflow": workflow.to_dict(), "gate": resolved.to_dict(), "controlActions": _control_actions(workflow_id, workflow.to_dict(), gates)})


def approve_workflow_for_materialization(
    conn: sqlite3.Connection,
    workflow_id: str,
    *,
    actor_id: str = "webui",
    now: float | None = None,
) -> dict[str, Any]:
    workflow = _require_workflow(conn, workflow_id)
    unresolved = [gate for gate in list_gates(conn, workflow_id) if gate.status not in {"approved", "rejected", "skipped"}]
    if unresolved:
        raise ValueError("workflow has unresolved gates")
    nodes = _node_rows(conn, workflow_id)
    if not nodes:
        raise ValueError("workflow DAG has no persisted nodes")
    if workflow.status == "materialized":
        raise ValueError("workflow is already materialized")
    if workflow.status != "dag_approved":
        updated = update_workflow_status(conn, workflow_id, status="dag_approved", current_gate=None, now=now)
        if updated is None:
            raise ValueError(f"workflow not found: {workflow_id}")
        workflow = updated
        add_event(
            conn,
            workflow_id=workflow_id,
            event_type="workflow_approved",
            actor_type="human" if actor_id == "webui" else "agent",
            actor_id=actor_id,
            message="Workflow approved for materialization",
            data={"status": "dag_approved"},
            now=now,
        )
    gates = [gate.to_dict() for gate in list_gates(conn, workflow_id)]
    return _response({"workflow": workflow.to_dict(), "controlActions": _control_actions(workflow_id, workflow.to_dict(), gates)})


def seed_actionable_workflow_fixture(
    conn: sqlite3.Connection,
    *,
    workflow_id: str = "wf_actionable_fixture",
    title: str = "Actionable workflow fixture",
    board: str = "core",
    workspace_path: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Seed a small persisted workflow with pending gates for live UI QA."""

    workflow = create_workflow(
        conn,
        workflow_id=workflow_id,
        title=title,
        description="Seeded workflow with pending Core control actions for WebUI verification.",
        workspace_path=workspace_path,
        board=board,
        scale="medium",
        status="dag_validated",
        created_by="workflow-fixture",
        now=now,
    )
    save_dag(
        conn,
        workflow_id=workflow.id,
        normalized_dag={
            "schema_version": 1,
            "workflow_id": workflow.id,
            "name": title,
            "nodes": [
                {
                    "id": "review-plan",
                    "title": "Review plan",
                    "role": "planner",
                    "profile": "planner",
                    "scope": {"summary": "Confirm the workflow plan is ready for implementation."},
                    "definition_of_done": ["Plan is accepted or explicitly rejected."],
                },
                {
                    "id": "build-slice",
                    "title": "Build slice",
                    "role": "engineer",
                    "profile": "engineer",
                    "parents": ["review-plan"],
                    "scope": {"summary": "Build the first useful workflow slice after approval."},
                    "definition_of_done": ["Implementation is complete.", "Targeted tests pass."],
                },
            ],
        },
        now=now,
    )
    add_gate(
        conn,
        workflow_id=workflow.id,
        node_id="review-plan",
        gate_id="gate_actionable_review",
        gate_type="dag_review",
        level=1,
        required_actor="human",
        status="pending",
        reason="Seeded pending gate for WebUI action verification.",
        metadata={"fixture": True, "actionable": True},
    )
    add_event(
        conn,
        workflow_id=workflow.id,
        node_id="review-plan",
        event_type="workflow_actionable_fixture_seeded",
        actor_type="system",
        actor_id="workflow-fixture",
        message="Seeded actionable workflow fixture.",
        data={"workflowId": workflow.id},
        now=now,
    )
    return get_workflow_dag(conn, workflow.id)



def _control_actions(workflow_id: str, workflow: dict[str, Any], gates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unresolved = [gate for gate in gates if gate.get("status") not in {"approved", "rejected", "skipped"}]
    actions: list[dict[str, Any]] = []
    for gate in unresolved:
        gate_id = gate.get("id")
        gate_type = gate.get("gateType") or "workflow"
        endpoint = f"/api/workflows/{workflow_id}/gates/{gate_id}/resolve"
        actions.extend(
            [
                {
                    "id": f"approve-gate:{gate_id}",
                    "type": "resolve_gate",
                    "label": f"Approve {gate_type} gate",
                    "method": "POST",
                    "endpoint": endpoint,
                    "gateId": gate_id,
                    "status": "approved",
                    "verdict": "approved",
                    "enabled": True,
                },
                {
                    "id": f"reject-gate:{gate_id}",
                    "type": "resolve_gate",
                    "label": f"Reject {gate_type} gate",
                    "method": "POST",
                    "endpoint": endpoint,
                    "gateId": gate_id,
                    "status": "rejected",
                    "verdict": "rejected",
                    "enabled": True,
                },
                {
                    "id": f"skip-gate:{gate_id}",
                    "type": "resolve_gate",
                    "label": f"Skip {gate_type} gate",
                    "method": "POST",
                    "endpoint": endpoint,
                    "gateId": gate_id,
                    "status": "skipped",
                    "verdict": "skipped",
                    "enabled": True,
                },
            ]
        )
    if not unresolved and workflow.get("status") not in {"dag_approved", "materialized"}:
        actions.append(
            {
                "id": "approve-workflow",
                "type": "approve_workflow",
                "label": "Approve workflow for materialization",
                "method": "POST",
                "endpoint": f"/api/workflows/{workflow_id}/approve",
                "enabled": True,
            }
        )
    if not unresolved and workflow.get("status") == "dag_approved":
        actions.append(
            {
                "id": "materialize-workflow",
                "type": "materialize_workflow",
                "label": "Materialize to Kanban",
                "method": "POST",
                "endpoint": f"/api/workflows/{workflow_id}/materialize",
                "enabled": True,
            }
        )
    return actions


def _workflow_id_from_inbox_item_id(inbox_item_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", inbox_item_id).strip("_")
    return f"wf_{slug or 'inbox_item'}"


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
        "verdict": row["verdict"],
        "requiredActor": row["required_actor"],
        "resolvedBy": row["resolved_by"],
        "resolvedAt": row["resolved_at"],
        "artifactId": row["artifact_id"],
        "reason": row["reason"],
        "metadata": _loads(row["metadata_json"], {}),
    }


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    return json.loads(raw)

"""SQLite workflow store.

The workflow DB is the authoritative source for workflow records, graph nodes,
gates, Kanban mappings, artifacts, and audit events. This module intentionally
starts with small CRUD primitives that later DAG/materialization/API layers can
compose without parsing Kanban task prose.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_default_hermes_root

SCHEMA_VERSION = 1
VALID_WORKFLOW_SCALES = {"small", "medium", "large", "xl"}


def workflow_home() -> Path:
    """Return the shared Hermes root for workflow state.

    ``HERMES_WORKFLOW_HOME`` exists for tests and unusual deployments. Normal
    installs use the profile-collapsed Hermes root, matching Kanban's shared
    coordination semantics.
    """

    override = os.environ.get("HERMES_WORKFLOW_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    return get_default_hermes_root()


def workflow_dir() -> Path:
    return workflow_home() / "workflow"


def workflow_db_path() -> Path:
    override = os.environ.get("HERMES_WORKFLOW_DB", "").strip()
    if override:
        return Path(override).expanduser()
    return workflow_dir() / "workflow.db"


def artifact_root() -> Path:
    return workflow_dir() / "artifacts"


@dataclass(frozen=True)
class WorkflowRecord:
    id: str
    title: str
    description: str
    workspace_path: str | None
    board: str
    scale: str
    status: str
    current_gate: str | None
    policy_path: str | None
    policy_snapshot: dict[str, Any]
    created_at: float
    updated_at: float
    created_by: str | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "workspacePath": self.workspace_path,
            "board": self.board,
            "scale": self.scale,
            "status": self.status,
            "currentGate": self.current_gate,
            "policyPath": self.policy_path,
            "policySnapshot": self.policy_snapshot,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "createdBy": self.created_by,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class WorkflowInboxItem:
    id: str
    title: str
    body: str
    source: str
    status: str
    classification: str | None
    workspace_path: str | None
    assigned_workflow_id: str | None
    created_at: float
    updated_at: float
    created_by: str | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "source": self.source,
            "status": self.status,
            "classification": self.classification,
            "workspacePath": self.workspace_path,
            "assignedWorkflowId": self.assigned_workflow_id,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "createdBy": self.created_by,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class WorkflowGate:
    id: str
    workflow_id: str
    node_id: str | None
    gate_type: str
    level: int
    status: str
    verdict: str | None
    required_actor: str
    resolved_by: str | None
    resolved_at: float | None
    artifact_id: str | None
    reason: str | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflowId": self.workflow_id,
            "nodeId": self.node_id,
            "gateType": self.gate_type,
            "level": self.level,
            "status": self.status,
            "verdict": self.verdict,
            "requiredActor": self.required_actor,
            "resolvedBy": self.resolved_by,
            "resolvedAt": self.resolved_at,
            "artifactId": self.artifact_id,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class WorkflowArtifact:
    id: str
    workflow_id: str
    kind: str
    path: str | None
    sha256: str | None
    mime_type: str | None
    schema_version: int
    status: str
    created_at: float
    created_by: str | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflowId": self.workflow_id,
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "mimeType": self.mime_type,
            "schemaVersion": self.schema_version,
            "status": self.status,
            "createdAt": self.created_at,
            "createdBy": self.created_by,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class WorkflowEvent:
    id: str
    workflow_id: str
    node_id: str | None
    event_type: str
    actor_type: str
    actor_id: str | None
    message: str
    data: dict[str, Any]
    created_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflowId": self.workflow_id,
            "nodeId": self.node_id,
            "eventType": self.event_type,
            "actorType": self.actor_type,
            "actorId": self.actor_id,
            "message": self.message,
            "data": self.data,
            "createdAt": self.created_at,
        }


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open a workflow DB connection and initialize schema."""

    db_path = Path(path).expanduser() if path is not None else workflow_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create or migrate workflow DB schema."""

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS workflows (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          description TEXT DEFAULT '',
          workspace_path TEXT,
          board TEXT NOT NULL DEFAULT 'default',
          scale TEXT NOT NULL CHECK (scale IN ('small','medium','large','xl')),
          status TEXT NOT NULL,
          current_gate TEXT,
          policy_path TEXT,
          policy_snapshot_json TEXT NOT NULL,
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL,
          created_by TEXT,
          metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS workflow_inbox_items (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          body TEXT NOT NULL DEFAULT '',
          source TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'new',
          classification TEXT,
          workspace_path TEXT,
          assigned_workflow_id TEXT,
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL,
          created_by TEXT,
          metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS workflow_artifacts (
          id TEXT PRIMARY KEY,
          workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
          kind TEXT NOT NULL,
          path TEXT,
          sha256 TEXT,
          mime_type TEXT,
          schema_version INTEGER NOT NULL DEFAULT 1,
          status TEXT NOT NULL DEFAULT 'active',
          created_at REAL NOT NULL,
          created_by TEXT,
          metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS workflow_nodes (
          workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
          node_id TEXT NOT NULL,
          title TEXT NOT NULL,
          role TEXT NOT NULL,
          profile TEXT,
          status TEXT NOT NULL,
          gate_level INTEGER NOT NULL DEFAULT 1,
          gate_type TEXT,
          kanban_task_id TEXT,
          branch TEXT,
          worktree_path TEXT,
          base_ref TEXT,
          definition_of_done_json TEXT NOT NULL DEFAULT '[]',
          scope_json TEXT NOT NULL DEFAULT '{}',
          evidence_json TEXT NOT NULL DEFAULT '{}',
          metadata_json TEXT NOT NULL DEFAULT '{}',
          created_at REAL NOT NULL,
          updated_at REAL NOT NULL,
          PRIMARY KEY (workflow_id, node_id)
        );

        CREATE TABLE IF NOT EXISTS workflow_edges (
          workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
          parent_node_id TEXT NOT NULL,
          child_node_id TEXT NOT NULL,
          kind TEXT NOT NULL DEFAULT 'depends_on',
          PRIMARY KEY (workflow_id, parent_node_id, child_node_id)
        );

        CREATE TABLE IF NOT EXISTS workflow_gates (
          id TEXT PRIMARY KEY,
          workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
          node_id TEXT,
          gate_type TEXT NOT NULL,
          level INTEGER NOT NULL,
          status TEXT NOT NULL,
          verdict TEXT,
          required_actor TEXT NOT NULL,
          resolved_by TEXT,
          resolved_at REAL,
          artifact_id TEXT,
          reason TEXT,
          metadata_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS workflow_events (
          id TEXT PRIMARY KEY,
          workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
          node_id TEXT,
          event_type TEXT NOT NULL,
          actor_type TEXT NOT NULL,
          actor_id TEXT,
          message TEXT NOT NULL DEFAULT '',
          data_json TEXT NOT NULL DEFAULT '{}',
          created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS workflow_kanban_mappings (
          workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
          node_id TEXT NOT NULL,
          board TEXT NOT NULL,
          task_id TEXT NOT NULL,
          materialized_at REAL NOT NULL,
          PRIMARY KEY (workflow_id, node_id),
          UNIQUE (board, task_id)
        );

        CREATE INDEX IF NOT EXISTS idx_workflows_board_status ON workflows(board, status);
        CREATE INDEX IF NOT EXISTS idx_workflow_inbox_status_source ON workflow_inbox_items(status, source);
        CREATE INDEX IF NOT EXISTS idx_workflow_events_workflow_created ON workflow_events(workflow_id, created_at);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def create_inbox_item(
    conn: sqlite3.Connection,
    *,
    title: str,
    body: str = "",
    source: str = "manual",
    status: str = "new",
    classification: str | None = None,
    workspace_path: str | Path | None = None,
    assigned_workflow_id: str | None = None,
    created_by: str | None = None,
    metadata: dict[str, Any] | None = None,
    inbox_item_id: str | None = None,
    now: float | None = None,
) -> WorkflowInboxItem:
    """Capture a raw, not-yet-shaped workflow intake item."""

    if not title.strip():
        raise ValueError("inbox item title must be non-empty")
    if not source.strip():
        raise ValueError("inbox item source must be non-empty")
    ts = time.time() if now is None else now
    iid = inbox_item_id or f"inbox_{secrets.token_hex(8)}"
    conn.execute(
        """
        INSERT INTO workflow_inbox_items (
          id, title, body, source, status, classification, workspace_path,
          assigned_workflow_id, created_at, updated_at, created_by, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            iid,
            title.strip(),
            body,
            source.strip(),
            status,
            classification,
            str(workspace_path) if workspace_path is not None else None,
            assigned_workflow_id,
            ts,
            ts,
            created_by,
            _json(metadata or {}),
        ),
    )
    conn.commit()
    return get_inbox_item(conn, iid)  # type: ignore[return-value]


def get_inbox_item(conn: sqlite3.Connection, inbox_item_id: str) -> WorkflowInboxItem | None:
    row = conn.execute("SELECT * FROM workflow_inbox_items WHERE id = ?", (inbox_item_id,)).fetchone()
    return _inbox_item_from_row(row) if row else None


def list_inbox_items(
    conn: sqlite3.Connection,
    *,
    status: str | None = None,
    source: str | None = None,
    classification: str | None = None,
    limit: int = 100,
) -> list[WorkflowInboxItem]:
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if source:
        clauses.append("source = ?")
        params.append(source)
    if classification:
        clauses.append("classification = ?")
        params.append(classification)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM workflow_inbox_items {where} ORDER BY updated_at DESC, id DESC LIMIT ?",
        (*params, limit),
    ).fetchall()
    return [_inbox_item_from_row(row) for row in rows]


_UNSET = object()


def update_inbox_item(
    conn: sqlite3.Connection,
    inbox_item_id: str,
    *,
    title: str | object = _UNSET,
    body: str | object = _UNSET,
    source: str | object = _UNSET,
    status: str | object = _UNSET,
    classification: str | None | object = _UNSET,
    workspace_path: str | Path | None | object = _UNSET,
    assigned_workflow_id: str | None | object = _UNSET,
    metadata: dict[str, Any] | None = None,
    now: float | None = None,
) -> WorkflowInboxItem | None:
    """Update triage fields for an inbox item, preserving unspecified fields."""

    current = get_inbox_item(conn, inbox_item_id)
    if current is None:
        return None
    updates: list[str] = []
    params: list[Any] = []

    if title is not _UNSET:
        clean_title = str(title).strip()
        if not clean_title:
            raise ValueError("inbox item title must be non-empty")
        updates.append("title = ?")
        params.append(clean_title)
    if body is not _UNSET:
        updates.append("body = ?")
        params.append(str(body))
    if source is not _UNSET:
        clean_source = str(source).strip()
        if not clean_source:
            raise ValueError("inbox item source must be non-empty")
        updates.append("source = ?")
        params.append(clean_source)
    if status is not _UNSET:
        updates.append("status = ?")
        params.append(str(status))
    if classification is not _UNSET:
        updates.append("classification = ?")
        params.append(classification)
    if workspace_path is not _UNSET:
        updates.append("workspace_path = ?")
        params.append(str(workspace_path) if workspace_path is not None else None)
    if assigned_workflow_id is not _UNSET:
        updates.append("assigned_workflow_id = ?")
        params.append(assigned_workflow_id)
    if metadata is not None:
        merged_metadata = {**current.metadata, **metadata}
        updates.append("metadata_json = ?")
        params.append(_json(merged_metadata))

    ts = time.time() if now is None else now
    updates.append("updated_at = ?")
    params.append(ts)
    params.append(inbox_item_id)
    conn.execute(f"UPDATE workflow_inbox_items SET {', '.join(updates)} WHERE id = ?", tuple(params))
    conn.commit()
    return get_inbox_item(conn, inbox_item_id)


def create_workflow(
    conn: sqlite3.Connection,
    *,
    title: str,
    description: str = "",
    workspace_path: str | Path | None = None,
    board: str = "default",
    scale: str = "medium",
    status: str = "inbox",
    current_gate: str | None = None,
    policy_path: str | Path | None = None,
    policy_snapshot: dict[str, Any] | None = None,
    created_by: str | None = None,
    metadata: dict[str, Any] | None = None,
    workflow_id: str | None = None,
    now: float | None = None,
) -> WorkflowRecord:
    """Insert and return a workflow record."""

    if not title.strip():
        raise ValueError("workflow title must be non-empty")
    if scale not in VALID_WORKFLOW_SCALES:
        raise ValueError(f"invalid workflow scale: {scale!r}")
    ts = time.time() if now is None else now
    wid = workflow_id or f"wf_{secrets.token_hex(8)}"
    conn.execute(
        """
        INSERT INTO workflows (
          id, title, description, workspace_path, board, scale, status,
          current_gate, policy_path, policy_snapshot_json, created_at,
          updated_at, created_by, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            wid,
            title.strip(),
            description,
            str(workspace_path) if workspace_path is not None else None,
            board or "default",
            scale,
            status,
            current_gate,
            str(policy_path) if policy_path is not None else None,
            _json(policy_snapshot or {}),
            ts,
            ts,
            created_by,
            _json(metadata or {}),
        ),
    )
    conn.commit()
    return get_workflow(conn, wid)  # type: ignore[return-value]


def get_workflow(conn: sqlite3.Connection, workflow_id: str) -> WorkflowRecord | None:
    row = conn.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
    return _workflow_from_row(row) if row else None


def list_workflows(conn: sqlite3.Connection, *, board: str | None = None, status: str | None = None, limit: int = 100) -> list[WorkflowRecord]:
    clauses: list[str] = []
    params: list[Any] = []
    if board:
        clauses.append("board = ?")
        params.append(board)
    if status:
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"SELECT * FROM workflows {where} ORDER BY updated_at DESC LIMIT ?",
        (*params, limit),
    ).fetchall()
    return [_workflow_from_row(row) for row in rows]


def update_workflow_status(
    conn: sqlite3.Connection,
    workflow_id: str,
    *,
    status: str,
    current_gate: str | None | object = _UNSET,
    now: float | None = None,
) -> WorkflowRecord | None:
    """Update workflow status/current gate and return the workflow record."""

    if get_workflow(conn, workflow_id) is None:
        return None
    ts = time.time() if now is None else now
    updates = ["status = ?", "updated_at = ?"]
    params: list[Any] = [status, ts]
    if current_gate is not _UNSET:
        updates.append("current_gate = ?")
        params.append(current_gate)
    params.append(workflow_id)
    conn.execute(f"UPDATE workflows SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()
    return get_workflow(conn, workflow_id)


def add_gate(
    conn: sqlite3.Connection,
    *,
    workflow_id: str,
    gate_type: str,
    level: int,
    required_actor: str,
    status: str = "pending",
    node_id: str | None = None,
    verdict: str | None = None,
    resolved_by: str | None = None,
    resolved_at: float | None = None,
    artifact_id: str | None = None,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    gate_id: str | None = None,
) -> WorkflowGate:
    """Create a workflow gate record."""

    if get_workflow(conn, workflow_id) is None:
        raise ValueError(f"workflow not found: {workflow_id}")
    if not gate_type.strip():
        raise ValueError("gate_type must be non-empty")
    if level < 1:
        raise ValueError("gate level must be positive")
    if not required_actor.strip():
        raise ValueError("required_actor must be non-empty")
    if artifact_id is not None:
        artifact = conn.execute(
            "SELECT 1 FROM workflow_artifacts WHERE id = ? AND workflow_id = ?",
            (artifact_id, workflow_id),
        ).fetchone()
        if artifact is None:
            raise ValueError(f"artifact not found: {artifact_id}")
    gid = gate_id or f"gate_{secrets.token_hex(8)}"
    conn.execute(
        """
        INSERT INTO workflow_gates (
          id, workflow_id, node_id, gate_type, level, status, verdict,
          required_actor, resolved_by, resolved_at, artifact_id, reason,
          metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            gid,
            workflow_id,
            node_id,
            gate_type.strip(),
            level,
            status,
            verdict,
            required_actor.strip(),
            resolved_by,
            resolved_at,
            artifact_id,
            reason,
            _json(metadata or {}),
        ),
    )
    conn.commit()
    return _gate_from_row(conn.execute("SELECT * FROM workflow_gates WHERE id = ?", (gid,)).fetchone())


def list_gates(conn: sqlite3.Connection, workflow_id: str, *, status: str | None = None, limit: int = 100) -> list[WorkflowGate]:
    clauses = ["workflow_id = ?"]
    params: list[Any] = [workflow_id]
    if status:
        clauses.append("status = ?")
        params.append(status)
    rows = conn.execute(
        f"SELECT * FROM workflow_gates WHERE {' AND '.join(clauses)} ORDER BY level ASC, id ASC LIMIT ?",
        (*params, limit),
    ).fetchall()
    return [_gate_from_row(row) for row in rows]


def resolve_gate(
    conn: sqlite3.Connection,
    *,
    gate_id: str,
    status: str,
    verdict: str | None,
    resolved_by: str,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    now: float | None = None,
) -> WorkflowGate:
    """Resolve an existing workflow gate."""

    existing = conn.execute("SELECT * FROM workflow_gates WHERE id = ?", (gate_id,)).fetchone()
    if existing is None:
        raise ValueError(f"gate not found: {gate_id}")
    resolved_at = time.time() if now is None else now
    existing_metadata = _loads(existing["metadata_json"], {})
    merged_metadata = {**existing_metadata, **metadata} if metadata is not None else existing_metadata
    resolved_reason = existing["reason"] if reason is None else reason
    conn.execute(
        """
        UPDATE workflow_gates
        SET status = ?, verdict = ?, resolved_by = ?, resolved_at = ?, reason = ?, metadata_json = ?
        WHERE id = ?
        """,
        (status, verdict, resolved_by, resolved_at, resolved_reason, _json(merged_metadata), gate_id),
    )
    conn.commit()
    return _gate_from_row(conn.execute("SELECT * FROM workflow_gates WHERE id = ?", (gate_id,)).fetchone())


def add_artifact(
    conn: sqlite3.Connection,
    *,
    workflow_id: str,
    kind: str,
    path: str | Path,
    mime_type: str | None = None,
    schema_version: int = 1,
    status: str = "active",
    created_by: str | None = None,
    metadata: dict[str, Any] | None = None,
    artifact_id: str | None = None,
    now: float | None = None,
) -> WorkflowArtifact:
    """Record an auditable workflow artifact and compute its SHA-256."""

    if get_workflow(conn, workflow_id) is None:
        raise ValueError(f"workflow not found: {workflow_id}")
    if not kind.strip():
        raise ValueError("artifact kind must be non-empty")
    artifact_path = Path(path).expanduser()
    if not artifact_path.is_file():
        raise FileNotFoundError(str(artifact_path))
    ts = time.time() if now is None else now
    aid = artifact_id or f"art_{secrets.token_hex(8)}"
    digest = _sha256_file(artifact_path)
    conn.execute(
        """
        INSERT INTO workflow_artifacts (
          id, workflow_id, kind, path, sha256, mime_type, schema_version,
          status, created_at, created_by, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            aid,
            workflow_id,
            kind.strip(),
            str(artifact_path),
            digest,
            mime_type,
            schema_version,
            status,
            ts,
            created_by,
            _json(metadata or {}),
        ),
    )
    conn.commit()
    return _artifact_from_row(conn.execute("SELECT * FROM workflow_artifacts WHERE id = ?", (aid,)).fetchone())


def list_artifacts(conn: sqlite3.Connection, workflow_id: str, *, kind: str | None = None, limit: int = 100) -> list[WorkflowArtifact]:
    clauses = ["workflow_id = ?"]
    params: list[Any] = [workflow_id]
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    rows = conn.execute(
        f"SELECT * FROM workflow_artifacts WHERE {' AND '.join(clauses)} ORDER BY created_at ASC LIMIT ?",
        (*params, limit),
    ).fetchall()
    return [_artifact_from_row(row) for row in rows]


def save_dag(conn: sqlite3.Connection, *, workflow_id: str, normalized_dag: dict[str, Any], now: float | None = None) -> None:
    """Replace persisted workflow nodes and edges from a normalized DAG."""

    if get_workflow(conn, workflow_id) is None:
        raise ValueError(f"workflow not found: {workflow_id}")
    dag_workflow_id = normalized_dag.get("workflow_id")
    if dag_workflow_id is not None and dag_workflow_id != workflow_id:
        raise ValueError(f"normalized DAG workflow_id does not match {workflow_id}: {dag_workflow_id}")
    ts = time.time() if now is None else now
    nodes = normalized_dag.get("nodes", [])
    edges = normalized_dag.get("edges", [])
    with conn:
        conn.execute("DELETE FROM workflow_edges WHERE workflow_id = ?", (workflow_id,))
        conn.execute("DELETE FROM workflow_nodes WHERE workflow_id = ?", (workflow_id,))
        for node in nodes:
            workspace = node.get("workspace") if isinstance(node.get("workspace"), dict) else {}
            gate = node.get("gate") if isinstance(node.get("gate"), dict) else {}
            conn.execute(
                """
                INSERT INTO workflow_nodes (
                  workflow_id, node_id, title, role, profile, status, gate_level,
                  gate_type, kanban_task_id, branch, worktree_path, base_ref,
                  definition_of_done_json, scope_json, evidence_json, metadata_json,
                  created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow_id,
                    node["id"],
                    node.get("title", ""),
                    node["role"],
                    node.get("profile"),
                    node.get("status", "waiting"),
                    node.get("gate_level", 1),
                    gate.get("type"),
                    node.get("kanban_task_id"),
                    node.get("branch") or workspace.get("branch"),
                    node.get("worktree_path") or workspace.get("worktree_path"),
                    node.get("base_ref") or workspace.get("base_ref"),
                    _json(node.get("definition_of_done", [])),
                    _json(node.get("scope", {})),
                    _json(node.get("evidence", {})),
                    _json(node.get("metadata", {})),
                    ts,
                    ts,
                ),
            )
        for edge in edges:
            conn.execute(
                """
                INSERT INTO workflow_edges (workflow_id, parent_node_id, child_node_id, kind)
                VALUES (?, ?, ?, ?)
                """,
                (workflow_id, edge["source"], edge["target"], edge.get("kind", "depends_on")),
            )
        conn.execute("UPDATE workflows SET updated_at = ? WHERE id = ?", (ts, workflow_id))


def add_event(
    conn: sqlite3.Connection,
    *,
    workflow_id: str,
    event_type: str,
    actor_type: str,
    message: str = "",
    node_id: str | None = None,
    actor_id: str | None = None,
    data: dict[str, Any] | None = None,
    event_id: str | None = None,
    now: float | None = None,
) -> WorkflowEvent:
    """Append an audit event to a workflow."""

    if get_workflow(conn, workflow_id) is None:
        raise ValueError(f"workflow not found: {workflow_id}")
    ts = time.time() if now is None else now
    eid = event_id or f"evt_{secrets.token_hex(8)}"
    conn.execute(
        """
        INSERT INTO workflow_events (
          id, workflow_id, node_id, event_type, actor_type, actor_id,
          message, data_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (eid, workflow_id, node_id, event_type, actor_type, actor_id, message, _json(data or {}), ts),
    )
    conn.commit()
    return _event_from_row(conn.execute("SELECT * FROM workflow_events WHERE id = ?", (eid,)).fetchone())


def list_events(conn: sqlite3.Connection, workflow_id: str, *, limit: int = 100) -> list[WorkflowEvent]:
    rows = conn.execute(
        "SELECT * FROM workflow_events WHERE workflow_id = ? ORDER BY created_at ASC LIMIT ?",
        (workflow_id, limit),
    ).fetchall()
    return [_event_from_row(row) for row in rows]


def _inbox_item_from_row(row: sqlite3.Row) -> WorkflowInboxItem:
    return WorkflowInboxItem(
        id=row["id"],
        title=row["title"],
        body=row["body"],
        source=row["source"],
        status=row["status"],
        classification=row["classification"],
        workspace_path=row["workspace_path"],
        assigned_workflow_id=row["assigned_workflow_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        created_by=row["created_by"],
        metadata=_loads(row["metadata_json"], {}),
    )


def _workflow_from_row(row: sqlite3.Row) -> WorkflowRecord:
    return WorkflowRecord(
        id=row["id"],
        title=row["title"],
        description=row["description"],
        workspace_path=row["workspace_path"],
        board=row["board"],
        scale=row["scale"],
        status=row["status"],
        current_gate=row["current_gate"],
        policy_path=row["policy_path"],
        policy_snapshot=_loads(row["policy_snapshot_json"], {}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        created_by=row["created_by"],
        metadata=_loads(row["metadata_json"], {}),
    )


def _gate_from_row(row: sqlite3.Row) -> WorkflowGate:
    return WorkflowGate(
        id=row["id"],
        workflow_id=row["workflow_id"],
        node_id=row["node_id"],
        gate_type=row["gate_type"],
        level=row["level"],
        status=row["status"],
        verdict=row["verdict"],
        required_actor=row["required_actor"],
        resolved_by=row["resolved_by"],
        resolved_at=row["resolved_at"],
        artifact_id=row["artifact_id"],
        reason=row["reason"],
        metadata=_loads(row["metadata_json"], {}),
    )


def _artifact_from_row(row: sqlite3.Row) -> WorkflowArtifact:
    return WorkflowArtifact(
        id=row["id"],
        workflow_id=row["workflow_id"],
        kind=row["kind"],
        path=row["path"],
        sha256=row["sha256"],
        mime_type=row["mime_type"],
        schema_version=row["schema_version"],
        status=row["status"],
        created_at=row["created_at"],
        created_by=row["created_by"],
        metadata=_loads(row["metadata_json"], {}),
    )


def _event_from_row(row: sqlite3.Row) -> WorkflowEvent:
    return WorkflowEvent(
        id=row["id"],
        workflow_id=row["workflow_id"],
        node_id=row["node_id"],
        event_type=row["event_type"],
        actor_type=row["actor_type"],
        actor_id=row["actor_id"],
        message=row["message"],
        data=_loads(row["data_json"], {}),
        created_at=row["created_at"],
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _loads(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return fallback

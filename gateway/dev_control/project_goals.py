"""Durable Dev project goal hierarchy — vision → goal → milestone → subgoal."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from gateway.dev_control.acceptance_criteria import normalize_acceptance_criteria
from gateway.dev_control.project_scope import DEFAULT_PROJECT_ID, resolve_project_id
from hermes_state import DEFAULT_DB_PATH, apply_wal_with_fallback


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dev_project_goals (
    goal_id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    parent_goal_id TEXT,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    markdown TEXT NOT NULL,
    status TEXT NOT NULL,
    acceptance_criteria TEXT NOT NULL,
    plan_artifact_id TEXT,
    progress REAL NOT NULL DEFAULT 0.0,
    ordering INTEGER NOT NULL DEFAULT 0,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    achieved_at REAL,
    abandoned_at REAL
);

CREATE INDEX IF NOT EXISTS idx_dev_goals_project_kind
    ON dev_project_goals(project_id, kind, status, ordering);

CREATE INDEX IF NOT EXISTS idx_dev_goals_parent
    ON dev_project_goals(parent_goal_id, ordering);

CREATE INDEX IF NOT EXISTS idx_dev_goals_plan
    ON dev_project_goals(plan_artifact_id);
"""

GOAL_KINDS = ("vision", "goal", "milestone", "subgoal")
GOAL_STATUSES = ("proposed", "active", "blocked", "achieved", "abandoned")
_PARENT_KIND_FOR: Dict[str, Optional[str]] = {
    "vision": None,
    "goal": "vision",
    "milestone": "goal",
    "subgoal": "milestone",
}


@dataclass
class DevProjectGoalStore:
    """Persistence for durable Dev project goals."""

    db_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.db_path = self.db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        apply_wal_with_fallback(self._conn, db_label="state.db")
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self._conn.close()

    def create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = _normalize_create_payload(payload)
        _validate_parent_kind(self, normalized["kind"], normalized.get("parent_goal_id"))
        now = float(normalized.get("created_at") or time.time())
        normalized["created_at"] = now
        normalized["updated_at"] = now
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO dev_project_goals (
                    goal_id, project_id, parent_goal_id, kind, title, markdown,
                    status, acceptance_criteria, plan_artifact_id, progress,
                    ordering, payload, created_at, updated_at, achieved_at,
                    abandoned_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _row_values(normalized),
            )
        created = self.get(normalized["goal_id"]) or normalized
        parent_id = created.get("parent_goal_id")
        if parent_id:
            recompute_rollup(self, parent_id)
        return created

    def update(self, goal_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get(goal_id)
        if not current:
            raise KeyError(f"Project goal not found: {goal_id}")
        if current.get("status") == "abandoned":
            raise ValueError("Abandoned goals cannot be updated")
        updates = dict(updates)
        updates.pop("progress", None)
        if "acceptance_criteria" in updates:
            updates["acceptance_criteria"] = normalize_acceptance_criteria(
                updates.get("acceptance_criteria"),
            )
        payload = {**current, **updates, "updated_at": time.time()}
        if payload.get("status") == "achieved":
            payload["progress"] = 1.0
            if not payload.get("achieved_at"):
                payload["achieved_at"] = time.time()
        with self._conn:
            self._conn.execute(
                """
                UPDATE dev_project_goals
                SET project_id = ?, parent_goal_id = ?, kind = ?, title = ?,
                    markdown = ?, status = ?, acceptance_criteria = ?,
                    plan_artifact_id = ?, progress = ?, ordering = ?, payload = ?,
                    created_at = ?, updated_at = ?, achieved_at = ?, abandoned_at = ?
                WHERE goal_id = ?
                """,
                (*_row_values(payload)[1:], goal_id),
            )
        updated = self.get(goal_id) or payload
        if "status" in updates and updated.get("parent_goal_id"):
            recompute_rollup(self, updated["parent_goal_id"])
        return updated

    def append_judge_audit(self, goal_id: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        current = self.get(goal_id)
        if not current:
            raise KeyError(f"Project goal not found: {goal_id}")
        payload = dict(current.get("payload") or {})
        history = list(payload.get("judge_history") or [])
        history.append({**entry, "recorded_at": float(entry.get("recorded_at") or time.time())})
        payload["judge_history"] = history[-20:]
        return self.update(goal_id, {"payload": payload})

    def set_progress(self, goal_id: str, progress: float) -> Dict[str, Any]:
        """Internal rollup helper — bypasses the public no-direct-progress rule."""
        current = self.get(goal_id)
        if not current:
            raise KeyError(f"Project goal not found: {goal_id}")
        clamped = max(0.0, min(1.0, float(progress)))
        with self._conn:
            self._conn.execute(
                """
                UPDATE dev_project_goals
                SET progress = ?, updated_at = ?
                WHERE goal_id = ?
                """,
                (round(clamped, 4), time.time(), goal_id),
            )
        return self.get(goal_id) or current

    def get(self, goal_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM dev_project_goals WHERE goal_id = ?",
            (str(goal_id or "").strip(),),
        ).fetchone()
        return _row_to_payload(row) if row else None

    def list(
        self,
        *,
        project_id: Optional[str] = None,
        kind: Optional[str] = None,
        status: Optional[str] = None,
        parent_goal_id: Optional[str] = None,
        include_abandoned: bool = True,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if project_id:
            clauses.append("project_id = ?")
            params.append(str(project_id).strip())
        if kind:
            clauses.append("kind = ?")
            params.append(str(kind).strip())
        if status:
            clauses.append("status = ?")
            params.append(str(status).strip())
        if parent_goal_id is not None:
            if str(parent_goal_id).strip():
                clauses.append("parent_goal_id = ?")
                params.append(str(parent_goal_id).strip())
            else:
                clauses.append("parent_goal_id IS NULL")
        if not include_abandoned:
            clauses.append("status != 'abandoned'")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit or 200), 500)))
        rows = self._conn.execute(
            f"""
            SELECT *
            FROM dev_project_goals
            {where}
            ORDER BY ordering ASC, created_at ASC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [_row_to_payload(row) for row in rows]

    def tree(
        self,
        project_id: str,
        *,
        include_abandoned: bool = False,
    ) -> Dict[str, Any]:
        nodes = self.list(
            project_id=project_id,
            include_abandoned=include_abandoned,
            limit=500,
        )
        by_id = {node["goal_id"]: {**node, "children": []} for node in nodes}
        roots: List[Dict[str, Any]] = []
        for node in by_id.values():
            parent_id = node.get("parent_goal_id")
            if parent_id and parent_id in by_id:
                by_id[parent_id]["children"].append(node)
            elif not parent_id:
                roots.append(node)
        return {
            "object": "hermes.dev_project_goal_tree",
            "project_id": project_id,
            "roots": roots,
            "total": len(nodes),
        }


def create_project_goal(
    *,
    store: DevProjectGoalStore,
    kind: str,
    title: str,
    project_id: Optional[str] = None,
    parent_goal_id: Optional[str] = None,
    markdown: str = "",
    status: str = "proposed",
    acceptance_criteria: Optional[List[Any]] = None,
    plan_artifact_id: Optional[str] = None,
    ordering: int = 0,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return store.create({
        "goal_id": str(uuid.uuid4()),
        "project_id": resolve_project_id(project_id, default=DEFAULT_PROJECT_ID),
        "parent_goal_id": parent_goal_id,
        "kind": kind,
        "title": title,
        "markdown": markdown,
        "status": status,
        "acceptance_criteria": acceptance_criteria or [],
        "plan_artifact_id": plan_artifact_id,
        "ordering": ordering,
        "payload": payload or {},
    })


def list_project_goals(
    *,
    store: DevProjectGoalStore,
    project_id: Optional[str] = None,
    kind: Optional[str] = None,
    status: Optional[str] = None,
    parent_goal_id: Optional[str] = None,
    include_abandoned: bool = True,
    limit: int = 200,
) -> Dict[str, Any]:
    data = store.list(
        project_id=project_id,
        kind=kind,
        status=status,
        parent_goal_id=parent_goal_id,
        include_abandoned=include_abandoned,
        limit=limit,
    )
    return {"object": "list", "data": data, "total": len(data)}


def get_project_goal_tree(
    *,
    store: DevProjectGoalStore,
    project_id: str,
    include_abandoned: bool = False,
) -> Dict[str, Any]:
    return store.tree(project_id, include_abandoned=include_abandoned)


def abandon_project_goal(*, store: DevProjectGoalStore, goal_id: str) -> Dict[str, Any]:
    current = store.get(goal_id)
    if not current:
        raise KeyError(f"Project goal not found: {goal_id}")
    if current.get("status") == "abandoned":
        return current
    now = time.time()
    with store._conn:
        store._conn.execute(
            """
            UPDATE dev_project_goals
            SET status = 'abandoned', abandoned_at = ?, updated_at = ?
            WHERE goal_id = ?
            """,
            (now, now, goal_id),
        )
    abandoned = store.get(goal_id) or current
    parent_id = abandoned.get("parent_goal_id")
    if parent_id:
        recompute_rollup(store, parent_id)
    return abandoned


def recompute_rollup(store: DevProjectGoalStore, goal_id: str) -> None:
    node = store.get(goal_id)
    if not node or node.get("status") == "abandoned":
        return

    children = [
        child for child in store.list(parent_goal_id=goal_id, include_abandoned=False)
    ]

    if not children:
        progress = _effective_progress(node)
        store.set_progress(goal_id, progress)
        parent_id = node.get("parent_goal_id")
        if parent_id:
            recompute_rollup(store, parent_id)
        return

    weights = [_child_weight(child) for child in children]
    total_weight = sum(weights) or 1.0
    progress = sum(
        weight * _effective_progress(child)
        for weight, child in zip(weights, children)
    ) / total_weight

    statuses = [child.get("status") for child in children]
    new_status = node.get("status") or "proposed"
    if new_status != "abandoned":
        if all(status == "achieved" for status in statuses):
            new_status = "achieved"
        elif any(status == "blocked" for status in statuses):
            new_status = "blocked"
        elif new_status == "achieved":
            new_status = "active"

    updates: Dict[str, Any] = {}
    store.set_progress(goal_id, progress)
    if new_status != node.get("status"):
        updates["status"] = new_status
        if new_status == "achieved":
            updates["achieved_at"] = time.time()
        store.update(goal_id, updates)

    parent_id = node.get("parent_goal_id")
    if parent_id:
        recompute_rollup(store, parent_id)


def _child_weight(child: Dict[str, Any]) -> float:
    payload = child.get("payload") or {}
    try:
        weight = float(payload.get("weight", 1.0))
    except (TypeError, ValueError):
        weight = 1.0
    return max(0.0, weight)


def _effective_progress(node: Dict[str, Any]) -> float:
    if node.get("status") == "achieved":
        return 1.0
    return float(node.get("progress") or 0.0)


def _normalize_create_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(payload.get("kind") or "").strip().lower()
    if kind not in GOAL_KINDS:
        raise ValueError(f"kind must be one of: {', '.join(GOAL_KINDS)}")

    parent_goal_id = payload.get("parent_goal_id")
    parent_text = str(parent_goal_id).strip() if parent_goal_id is not None else ""
    expected_parent_kind = _PARENT_KIND_FOR[kind]

    if kind == "vision":
        if parent_text:
            raise ValueError("vision nodes must not have a parent_goal_id")
        parent_goal_id = None
    else:
        if not parent_text:
            raise ValueError(f"{kind} nodes require a parent_goal_id")
        parent_goal_id = parent_text

    status = str(payload.get("status") or "proposed").strip().lower()
    if status not in GOAL_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(GOAL_STATUSES)}")

    title = str(payload.get("title") or "").strip()
    if not title:
        raise ValueError("title is required")

    return {
        "goal_id": str(payload.get("goal_id") or uuid.uuid4()),
        "project_id": resolve_project_id(payload.get("project_id"), default=DEFAULT_PROJECT_ID),
        "parent_goal_id": parent_goal_id,
        "kind": kind,
        "title": title,
        "markdown": str(payload.get("markdown") or ""),
        "status": status,
        "acceptance_criteria": normalize_acceptance_criteria(payload.get("acceptance_criteria")),
        "plan_artifact_id": payload.get("plan_artifact_id"),
        "progress": 0.0,
        "ordering": int(payload.get("ordering") or 0),
        "payload": dict(payload.get("payload") or {}),
        "created_at": payload.get("created_at"),
        "updated_at": payload.get("updated_at"),
        "achieved_at": payload.get("achieved_at"),
        "abandoned_at": payload.get("abandoned_at"),
    }


def _validate_parent_kind(store: DevProjectGoalStore, kind: str, parent_goal_id: Optional[str]) -> None:
    if kind == "vision":
        return
    parent = store.get(str(parent_goal_id or ""))
    if not parent:
        raise ValueError(f"parent goal not found: {parent_goal_id}")
    expected = _PARENT_KIND_FOR.get(kind)
    if expected and parent.get("kind") != expected:
        raise ValueError(
            f"{kind} parent must be kind={expected}, got kind={parent.get('kind')}"
        )


def _row_values(payload: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        payload["goal_id"],
        payload["project_id"],
        payload.get("parent_goal_id"),
        payload["kind"],
        payload["title"],
        payload["markdown"],
        payload["status"],
        json.dumps(payload.get("acceptance_criteria") or [], ensure_ascii=False),
        payload.get("plan_artifact_id"),
        float(payload.get("progress") or 0.0),
        int(payload.get("ordering") or 0),
        json.dumps(payload.get("payload") or {}, ensure_ascii=False),
        float(payload["created_at"]),
        float(payload["updated_at"]),
        payload.get("achieved_at"),
        payload.get("abandoned_at"),
    )


def _row_to_payload(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "object": "hermes.dev_project_goal",
        "goal_id": row["goal_id"],
        "project_id": row["project_id"],
        "parent_goal_id": row["parent_goal_id"],
        "kind": row["kind"],
        "title": row["title"],
        "markdown": row["markdown"],
        "status": row["status"],
        "acceptance_criteria": json.loads(row["acceptance_criteria"] or "[]"),
        "plan_artifact_id": row["plan_artifact_id"],
        "progress": float(row["progress"] or 0.0),
        "ordering": int(row["ordering"] or 0),
        "payload": json.loads(row["payload"] or "{}"),
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "achieved_at": row["achieved_at"],
        "abandoned_at": row["abandoned_at"],
    }

"""Compact structured cleanup reports for Kanban maintenance runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_TERMINAL_PARENT_STATUSES = {"done", "archived"}


def _task_statuses(conn: sqlite3.Connection, task_ids: list[str]) -> dict[str, str]:
    if not task_ids:
        return {}
    placeholders = ",".join("?" for _ in task_ids)
    rows = conn.execute(
        f"SELECT id, status FROM tasks WHERE id IN ({placeholders})",
        task_ids,
    ).fetchall()
    return {str(row["id"]): str(row["status"]) for row in rows}


def _parent_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    return [
        str(row["parent_id"])
        for row in conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
            (task_id,),
        ).fetchall()
    ]


def _child_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    return [
        str(row["child_id"])
        for row in conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id",
            (task_id,),
        ).fetchall()
    ]


def _dependency_graph_ids(conn: sqlite3.Connection, root_task_id: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    queue = [root_task_id]
    while queue:
        task_id = queue.pop(0)
        if task_id in seen:
            continue
        seen.add(task_id)
        ordered.append(task_id)
        queue.extend(child for child in _child_ids(conn, task_id) if child not in seen)
    return ordered


def _inventory_ids(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row["id"])
        for row in conn.execute(
            "SELECT id FROM tasks ORDER BY created_at ASC, id ASC"
        ).fetchall()
    ]


def _cards(conn: sqlite3.Connection, task_ids: list[str]) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for task_id in task_ids:
        row = conn.execute(
            "SELECT id, title, status, assignee FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            continue
        cards.append(
            {
                "id": str(row["id"]),
                "title": str(row["title"]),
                "status": str(row["status"]),
                "assignee": row["assignee"],
                "parents": _parent_ids(conn, task_id),
                "children": _child_ids(conn, task_id),
            }
        )
    return cards


def _remaining_gated_items(conn: sqlite3.Connection, task_ids: list[str]) -> list[dict[str, Any]]:
    parent_cache: dict[str, list[str]] = {task_id: _parent_ids(conn, task_id) for task_id in task_ids}
    all_parent_ids = sorted({parent_id for parents in parent_cache.values() for parent_id in parents})
    parent_statuses = _task_statuses(conn, all_parent_ids)
    gated: list[dict[str, Any]] = []
    task_statuses = _task_statuses(conn, task_ids)
    for task_id in task_ids:
        waiting_on = [
            parent_id
            for parent_id in parent_cache[task_id]
            if parent_statuses.get(parent_id) not in _TERMINAL_PARENT_STATUSES
        ]
        if waiting_on:
            gated.append(
                {
                    "id": task_id,
                    "status": task_statuses.get(task_id, "unknown"),
                    "waiting_on": waiting_on,
                }
            )
    return gated


def build_cleanup_report(
    conn: sqlite3.Connection,
    *,
    board: str,
    root_task_id: str | None = None,
    inventory_only: bool = False,
) -> dict[str, Any]:
    """Build a compact, value-free report from structured Kanban state."""
    if root_task_id and inventory_only:
        raise ValueError("cleanup report accepts either --task or --inventory-only, not both")
    if not root_task_id and not inventory_only:
        raise ValueError("cleanup report requires --task or --inventory-only")

    if inventory_only:
        task_ids = _inventory_ids(conn)
        scope = {"type": "inventory_only", "root_task_id": None}
    else:
        assert root_task_id is not None
        if conn.execute("SELECT 1 FROM tasks WHERE id = ?", (root_task_id,)).fetchone() is None:
            raise ValueError(f"task {root_task_id} not found")
        task_ids = _dependency_graph_ids(conn, root_task_id)
        scope = {"type": "dependency_graph", "root_task_id": root_task_id}

    cards = _cards(conn, task_ids)
    checked_ids = [card["id"] for card in cards]
    missing_ids = [task_id for task_id in task_ids if task_id not in checked_ids]
    return {
        "board": board,
        "scope": scope,
        "cards_inspected": cards,
        "state_mutations": [],
        "comments_added": [],
        "verification_after_mutation": {
            "ok": not missing_ids,
            "checked_task_ids": checked_ids,
            "missing_task_ids": missing_ids,
        },
        "remaining_gated_items": _remaining_gated_items(conn, checked_ids),
    }


def write_cleanup_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path

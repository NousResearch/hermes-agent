"""Profile Tasks dashboard plugin — read-only backend API.

Mounted by the dashboard plugin system at ``/api/plugins/profile-tasks``.
This module intentionally exposes a narrow, read-only view over local Hermes
profiles and Kanban boards for a profile-scoped task dashboard.  It never calls
Kanban mutation helpers and opens board databases with SQLite ``mode=ro`` plus
``PRAGMA query_only=ON``.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Query

from hermes_cli import kanban_db
from hermes_cli import profiles as profile_mod

router = APIRouter()

_ACTIVE_COLUMNS = ("running", "blocked", "review")
_RECENT_DONE_LIMIT = 10
_SUMMARY_PREVIEW_CHARS = 240
_STALE_HEARTBEAT_SECONDS = 15 * 60


def _safe_preview(value: Any, limit: int = _SUMMARY_PREVIEW_CHARS) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _profile_dict(info: Any) -> dict[str, Any]:
    """Return non-secret profile metadata suitable for the dashboard."""
    return {
        "name": info.name,
        "is_default": bool(info.is_default),
        "gateway_running": bool(info.gateway_running),
        "model": info.model,
        "provider": info.provider,
        "has_env": bool(info.has_env),
        "skill_count": int(info.skill_count or 0),
        "description": info.description or "",
        "description_auto": bool(info.description_auto),
        "distribution_name": info.distribution_name,
        "distribution_version": info.distribution_version,
        "alias_name": info.alias_name,
    }


def _profiles_by_name() -> dict[str, Any]:
    return {p.name: p for p in profile_mod.list_profiles()}


def _normalize_existing_profile(profile: str) -> str:
    try:
        name = profile_mod.normalize_profile_name(profile)
        profile_mod.validate_profile_name(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Use list_profiles() as the local-profile source of truth, as requested.
    if name not in _profiles_by_name():
        raise HTTPException(status_code=404, detail=f"profile not found: {name}")
    return name


def _board_dict(meta: dict[str, Any]) -> dict[str, Any]:
    # Do not expose absolute db_path/default_workdir in the API response.
    return {
        "slug": meta.get("slug"),
        "name": meta.get("name") or meta.get("slug"),
        "description": meta.get("description") or "",
        "icon": meta.get("icon") or "",
        "color": meta.get("color") or "",
        "archived": bool(meta.get("archived", False)),
        "created_at": meta.get("created_at"),
        "db_exists": Path(str(meta.get("db_path", ""))).is_file(),
    }


def _board_slugs() -> set[str]:
    return {str(b.get("slug")) for b in kanban_db.list_boards(include_archived=True)}


def _normalize_board(board: str | None) -> str:
    slug = (board or kanban_db.DEFAULT_BOARD).strip() or kanban_db.DEFAULT_BOARD
    if slug not in _board_slugs():
        raise HTTPException(status_code=404, detail=f"board not found: {slug}")
    return slug


@contextmanager
def _open_board_readonly(board: str) -> Iterator[sqlite3.Connection]:
    db_path = kanban_db.kanban_db_path(board)
    if not db_path.is_file():
        raise FileNotFoundError(str(db_path))
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        yield conn
    finally:
        conn.close()


def _json_or_none(value: Any) -> Any:
    if value in (None, ""):
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def _latest_runs(conn: sqlite3.Connection, task_ids: list[str]) -> dict[str, sqlite3.Row]:
    if not task_ids:
        return {}
    placeholders = ",".join("?" for _ in task_ids)
    rows = conn.execute(
        f"""
        SELECT r.*
        FROM task_runs r
        JOIN (
          SELECT task_id, MAX(id) AS max_id
          FROM task_runs
          WHERE task_id IN ({placeholders})
          GROUP BY task_id
        ) latest ON latest.max_id = r.id
        """,
        task_ids,
    ).fetchall()
    return {str(r["task_id"]): r for r in rows}


def _task_warnings(row: sqlite3.Row, latest_run: sqlite3.Row | None, now: int) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    if row["status"] == "running":
        claim_expires = row["claim_expires"]
        if claim_expires is not None and int(claim_expires) < now:
            warnings.append({"kind": "stale_claim", "message": "Running claim has expired."})
        heartbeat = row["last_heartbeat_at"]
        if heartbeat is not None and int(heartbeat) < now - _STALE_HEARTBEAT_SECONDS:
            warnings.append({"kind": "stale_heartbeat", "message": "Worker heartbeat is stale."})
    if row["consecutive_failures"] and int(row["consecutive_failures"]) > 0:
        warnings.append({
            "kind": "recent_failures",
            "message": f"{int(row['consecutive_failures'])} consecutive failure(s).",
        })
    if latest_run is not None and latest_run["outcome"] in {"crashed", "timed_out", "spawn_failed", "gave_up"}:
        warnings.append({"kind": "last_run_failed", "message": "Latest run did not complete successfully."})
    return warnings


def _task_dict(row: sqlite3.Row, latest_run: sqlite3.Row | None, now: int) -> dict[str, Any]:
    skills = _json_or_none(row["skills"])
    if not isinstance(skills, list):
        skills = []
    summary_source = latest_run["summary"] if latest_run is not None and latest_run["summary"] else row["result"]
    safe: dict[str, Any] = {
        "id": row["id"],
        "title": row["title"],
        "status": row["status"],
        "assignee": row["assignee"],
        "priority": row["priority"] or 0,
        "created_by": row["created_by"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "tenant": row["tenant"],
        "workspace_kind": row["workspace_kind"],
        "branch_name": row["branch_name"],
        "claim_expires": row["claim_expires"],
        "last_heartbeat_at": row["last_heartbeat_at"],
        "current_run_id": row["current_run_id"],
        "consecutive_failures": row["consecutive_failures"] or 0,
        "workflow_template_id": row["workflow_template_id"],
        "current_step_key": row["current_step_key"],
        "skills_count": len(skills),
        "summary_preview": _safe_preview(summary_source),
        "warnings": [],
        "latest_run": None,
    }
    if latest_run is not None:
        safe["latest_run"] = {
            "id": latest_run["id"],
            "task_id": latest_run["task_id"],
            "profile": latest_run["profile"],
            "step_key": latest_run["step_key"],
            "status": latest_run["status"],
            "started_at": latest_run["started_at"],
            "ended_at": latest_run["ended_at"],
            "outcome": latest_run["outcome"],
            "summary_preview": _safe_preview(latest_run["summary"]),
        }
    safe["warnings"] = _task_warnings(row, latest_run, now)
    return safe


def _fetch_tasks_for_profile(
    conn: sqlite3.Connection,
    profile: str,
    *,
    include_ready: bool,
) -> dict[str, list[dict[str, Any]]]:
    statuses = list(_ACTIVE_COLUMNS) + ["done"]
    if include_ready:
        statuses.append("ready")
    placeholders = ",".join("?" for _ in statuses)
    rows = conn.execute(
        f"""
        SELECT * FROM tasks
        WHERE assignee = ? AND status IN ({placeholders})
        ORDER BY
          CASE status
            WHEN 'running' THEN 0
            WHEN 'blocked' THEN 1
            WHEN 'review' THEN 2
            WHEN 'ready' THEN 3
            WHEN 'done' THEN 4
            ELSE 5
          END,
          COALESCE(completed_at, started_at, created_at) DESC,
          priority DESC,
          created_at DESC
        """,
        [profile, *statuses],
    ).fetchall()
    latest_by_task = _latest_runs(conn, [r["id"] for r in rows])
    now = int(time.time())
    columns: dict[str, list[dict[str, Any]]] = {name: [] for name in _ACTIVE_COLUMNS}
    columns["recent_done"] = []
    if include_ready:
        columns["ready"] = []
    for row in rows:
        task = _task_dict(row, latest_by_task.get(row["id"]), now)
        status = str(row["status"])
        if status == "done":
            if len(columns["recent_done"]) < _RECENT_DONE_LIMIT:
                columns["recent_done"].append(task)
        elif status == "ready":
            if include_ready:
                columns["ready"].append(task)
        elif status in columns:
            columns[status].append(task)
    return columns


@router.get("/profiles")
def list_profiles() -> dict[str, Any]:
    profiles = [_profile_dict(p) for p in profile_mod.list_profiles()]
    return {"profiles": profiles, "count": len(profiles)}


@router.get("/boards")
def list_boards() -> dict[str, Any]:
    boards = [_board_dict(b) for b in kanban_db.list_boards(include_archived=True)]
    return {"boards": boards, "count": len(boards)}


@router.get("/tasks")
def tasks_for_profile(
    profile: str = Query(..., description="Local Hermes profile name"),
    board: str | None = Query(None, description="Kanban board slug"),
    include_ready: bool = Query(False, description="Include assigned ready tasks"),
) -> dict[str, Any]:
    profile_name = _normalize_existing_profile(profile)
    board_slug = _normalize_board(board)
    board_meta = next((b for b in kanban_db.list_boards(include_archived=True) if b.get("slug") == board_slug), None) or {"slug": board_slug}
    warnings: list[dict[str, Any]] = []
    try:
        with _open_board_readonly(board_slug) as conn:
            columns = _fetch_tasks_for_profile(conn, profile_name, include_ready=include_ready)
    except FileNotFoundError:
        columns = {name: [] for name in _ACTIVE_COLUMNS}
        columns["recent_done"] = []
        if include_ready:
            columns["ready"] = []
        warnings.append({"kind": "missing_board_db", "message": "Kanban database does not exist yet."})
    except sqlite3.DatabaseError as exc:
        raise HTTPException(status_code=500, detail=f"could not read kanban board: {exc}") from exc

    return {
        "profile": profile_name,
        "board": _board_dict(board_meta),
        "include_ready": include_ready,
        "columns": columns,
        "warnings": warnings,
        "read_only": True,
        "generated_at": int(time.time()),
    }


@router.get("/overview")
def overview(
    profile: str = Query(..., description="Local Hermes profile name"),
    board: str | None = Query(None, description="Kanban board slug"),
    include_ready: bool = Query(False, description="Include assigned ready tasks"),
) -> dict[str, Any]:
    profile_name = _normalize_existing_profile(profile)
    info = _profiles_by_name()[profile_name]
    tasks = tasks_for_profile(profile=profile_name, board=board, include_ready=include_ready)
    return {
        "profile": _profile_dict(info),
        "tasks": tasks,
    }

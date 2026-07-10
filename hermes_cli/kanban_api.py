"""Sanitized REST adapter for the shared Hermes Kanban store.

The dashboard plugin mounts this router at ``/api/plugins/kanban``.  The
adapter intentionally exposes workflow/task concepts rather than SQLite rows:
private task bodies, results, comments, workspace paths, claims, process data,
profile internals, and raw event payloads are never serialized.
"""

from __future__ import annotations

import re
import sqlite3
import time
from contextlib import contextmanager
from typing import Annotated, Any, Iterator, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field

from agent.redact import redact_sensitive_text
from hermes_cli import kanban_db

router = APIRouter()

_API_VERSION = "1"
_DEFAULT_LOG_TAIL = 8_192
_MAX_LOG_TAIL = 32_768
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w:])(?:[A-Za-z]:[\\/](?:[^\s\\/]+[\\/])*[^\s\\/]*|/(?:[^/\s]+/)+[^/\s]*)"
)
_AUTH_HEADER_RE = re.compile(
    r"(?im)^(authorization\s*:\s*(?:bearer|basic)\s+)[^\s]+"
)


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CreateTaskRequest(_RequestModel):
    title: str = Field(min_length=1, max_length=300)
    body: Optional[str] = Field(default=None, max_length=100_000)
    assignee: Optional[str] = Field(default=None, max_length=128)
    tenant: Optional[str] = Field(default=None, max_length=128)
    priority: int = Field(default=0, ge=-1_000_000, le=1_000_000)
    parents: list[str] = Field(default_factory=list, max_length=100)
    triage: bool = False
    idempotency_key: Optional[str] = Field(default=None, min_length=1, max_length=200)
    max_runtime_seconds: Optional[int] = Field(default=None, ge=1, le=31_536_000)
    skills: Optional[list[str]] = Field(default=None, max_length=50)


class UpdateTaskRequest(_RequestModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=300)
    body: Optional[str] = Field(default=None, max_length=100_000)
    assignee: Optional[str] = Field(default=None, max_length=128)
    priority: Optional[int] = Field(default=None, ge=-1_000_000, le=1_000_000)


class CommentRequest(_RequestModel):
    body: str = Field(min_length=1, max_length=20_000)
    author: str = Field(default="external-api", min_length=1, max_length=128)


class CompleteRequest(_RequestModel):
    summary: Optional[str] = Field(default=None, max_length=50_000)


class BlockRequest(_RequestModel):
    reason: Optional[str] = Field(default=None, max_length=20_000)
    kind: Optional[str] = None


@contextmanager
def _connection(board: Optional[str]) -> Iterator[sqlite3.Connection]:
    slug = _resolve_board_slug(board)
    kanban_db.init_db(board=slug)
    conn = kanban_db.connect(board=slug)
    try:
        yield conn
    finally:
        conn.close()


def _resolve_board_slug(board: Optional[str]) -> str:
    if board is None or not str(board).strip():
        return kanban_db.get_current_board()
    try:
        slug = kanban_db._normalize_board_slug(board)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not slug or not kanban_db.board_exists(slug):
        raise HTTPException(status_code=404, detail="board not found")
    return slug


def _resolve_board_id_or_name(value: str) -> str:
    candidate = value.strip()
    try:
        slug = kanban_db._normalize_board_slug(candidate)
    except ValueError:
        slug = None
    if slug and kanban_db.board_exists(slug):
        return slug

    matches = [
        item["slug"]
        for item in kanban_db.list_boards(include_archived=False)
        if str(item.get("name") or "").casefold() == candidate.casefold()
    ]
    if not matches:
        raise HTTPException(status_code=404, detail="board not found")
    if len(matches) > 1:
        raise HTTPException(status_code=409, detail="board name is ambiguous; use its id")
    return matches[0]


def _board_counts(slug: str) -> dict[str, int]:
    kanban_db.init_db(board=slug)
    conn = kanban_db.connect(board=slug)
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
        ).fetchall()
        return {row["status"]: int(row["count"]) for row in rows}
    finally:
        conn.close()


def _board_dto(meta: dict[str, Any], *, current: str) -> dict[str, Any]:
    slug = str(meta["slug"])
    counts = _board_counts(slug)
    return {
        "id": slug,
        "name": str(meta.get("name") or slug),
        "description": str(meta.get("description") or ""),
        "icon": str(meta.get("icon") or ""),
        "color": str(meta.get("color") or ""),
        "created_at": meta.get("created_at"),
        "is_current": slug == current,
        "counts": counts,
        "total": sum(counts.values()),
    }


def _task_dto(conn: sqlite3.Connection, task: kanban_db.Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "assignee": task.assignee,
        "status": task.status,
        "priority": task.priority,
        "tenant": task.tenant,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "workflow_template_id": task.workflow_template_id,
        "current_step_key": task.current_step_key,
        "block_kind": task.block_kind,
        "links": {
            "parents": kanban_db.parent_ids(conn, task.id),
            "children": kanban_db.child_ids(conn, task.id),
        },
    }


def _require_task(conn: sqlite3.Connection, task_id: str) -> kanban_db.Task:
    task = kanban_db.get_task(conn, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


def _task_response(conn: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    return {"task": _task_dto(conn, _require_task(conn, task_id))}


def _transition_response(
    conn: sqlite3.Connection,
    task_id: str,
    ok: bool,
    action: str,
) -> dict[str, Any]:
    if not ok:
        if kanban_db.get_task(conn, task_id) is None:
            raise HTTPException(status_code=404, detail="task not found")
        raise HTTPException(status_code=409, detail=f"task cannot be {action} from its current state")
    return _task_response(conn, task_id)


def _idempotency_key(
    payload_key: Optional[str],
    header_key: Optional[str],
) -> Optional[str]:
    body = (payload_key or "").strip()
    header = (header_key or "").strip()
    if body and header and body != header:
        raise HTTPException(status_code=409, detail="conflicting idempotency keys")
    key = header or body or None
    if key and len(key) > 200:
        raise HTTPException(status_code=422, detail="idempotency key is too long")
    return key


def _existing_idempotent_task(
    conn: sqlite3.Connection,
    key: Optional[str],
) -> Optional[str]:
    if not key:
        return None
    row = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (key,),
    ).fetchone()
    return str(row["id"]) if row else None


def _sanitize_log(content: str) -> str:
    redacted = redact_sensitive_text(content, force=True)
    redacted = _AUTH_HEADER_RE.sub(r"\1[REDACTED]", redacted)
    return _ABSOLUTE_PATH_RE.sub("[PATH]", redacted)


@router.get("/health")
def health() -> dict[str, Any]:
    current = kanban_db.get_current_board()
    try:
        with _connection(current) as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="kanban store unavailable") from exc
    return {
        "status": "ok",
        "service": "hermes-kanban",
        "api_version": _API_VERSION,
        "current_board": current,
    }


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    return {
        "api_version": _API_VERSION,
        "boards": {"read": True, "write": False},
        "tasks": {"read": True, "create": True, "update": True},
        "actions": ["comment", "complete", "block", "unblock", "archive"],
        "links": {"create": True, "delete": True},
        "observability": ["events", "runs", "sanitized_log_excerpt"],
        "task_statuses": sorted(kanban_db.VALID_STATUSES),
        "block_kinds": sorted(kanban_db.VALID_BLOCK_KINDS),
        "idempotent_task_creation": True,
        "profile_execution": False,
        "profiles_api": False,
    }


@router.get("/boards")
def list_boards() -> dict[str, Any]:
    current = kanban_db.get_current_board()
    boards = [
        _board_dto(meta, current=current)
        for meta in kanban_db.list_boards(include_archived=False)
    ]
    return {"boards": boards, "current": current, "count": len(boards)}


@router.get("/boards/{board_id_or_name}")
def get_board(board_id_or_name: str) -> dict[str, Any]:
    slug = _resolve_board_id_or_name(board_id_or_name)
    current = kanban_db.get_current_board()
    return {"board": _board_dto(kanban_db.read_board_metadata(slug), current=current)}


@router.get("/tasks")
def list_tasks(
    board: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    assignee: Optional[str] = Query(default=None),
    tenant: Optional[str] = Query(default=None),
    include_archived: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=200),
) -> dict[str, Any]:
    try:
        with _connection(board) as conn:
            tasks = kanban_db.list_tasks(
                conn,
                status=status,
                assignee=assignee,
                tenant=tenant,
                include_archived=include_archived,
                limit=limit,
            )
            items = [_task_dto(conn, task) for task in tasks]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"tasks": items, "count": len(items), "limit": limit}


@router.post("/tasks", status_code=201)
def create_task(
    payload: CreateTaskRequest,
    response: Response,
    board: Optional[str] = Query(default=None),
    idempotency_header: Annotated[Optional[str], Header(alias="Idempotency-Key")] = None,
) -> dict[str, Any]:
    key = _idempotency_key(payload.idempotency_key, idempotency_header)
    with _connection(board) as conn:
        existing = _existing_idempotent_task(conn, key)
        if existing:
            response.status_code = 200
            return {**_task_response(conn, existing), "created": False}
        try:
            task_id = kanban_db.create_task(
                conn,
                title=payload.title,
                body=payload.body,
                assignee=payload.assignee,
                created_by="external-api",
                workspace_kind="scratch",
                tenant=payload.tenant,
                priority=payload.priority,
                parents=payload.parents,
                triage=payload.triage,
                idempotency_key=key,
                max_runtime_seconds=payload.max_runtime_seconds,
                skills=payload.skills,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        created = existing is None
        return {**_task_response(conn, task_id), "created": created}


@router.get("/tasks/{task_id}")
def get_task(task_id: str, board: Optional[str] = Query(default=None)) -> dict[str, Any]:
    with _connection(board) as conn:
        return _task_response(conn, task_id)


@router.patch("/tasks/{task_id}")
def update_task(
    task_id: str,
    payload: UpdateTaskRequest,
    board: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    fields = payload.model_fields_set
    if not fields:
        raise HTTPException(status_code=400, detail="at least one field is required")
    with _connection(board) as conn:
        _require_task(conn, task_id)
        if "assignee" in fields:
            try:
                if not kanban_db.assign_task(conn, task_id, payload.assignee or None):
                    raise HTTPException(status_code=404, detail="task not found")
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        scalar_fields: list[tuple[str, Any]] = []
        if "title" in fields:
            scalar_fields.append(("title", payload.title))
        if "body" in fields:
            scalar_fields.append(("body", payload.body))
        if "priority" in fields:
            scalar_fields.append(("priority", payload.priority))
        if scalar_fields:
            with kanban_db.write_txn(conn):
                assignments = ", ".join(f"{name} = ?" for name, _ in scalar_fields)
                conn.execute(
                    f"UPDATE tasks SET {assignments} WHERE id = ?",
                    [value for _, value in scalar_fields] + [task_id],
                )
                kanban_db._append_event(
                    conn,
                    task_id,
                    "edited",
                    {"fields": [name for name, _ in scalar_fields]},
                )
        return _task_response(conn, task_id)


@router.post("/tasks/{task_id}/comment", status_code=201)
def comment_task(
    task_id: str,
    payload: CommentRequest,
    board: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        try:
            comment_id = kanban_db.add_comment(conn, task_id, payload.author, payload.body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        row = conn.execute(
            "SELECT created_at FROM task_comments WHERE id = ?", (comment_id,)
        ).fetchone()
        return {
            "comment": {
                "id": comment_id,
                "task_id": task_id,
                "created_at": row["created_at"] if row else int(time.time()),
            }
        }


@router.post("/tasks/{task_id}/complete")
def complete_task(
    task_id: str,
    payload: Optional[CompleteRequest] = None,
    board: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        return _transition_response(
            conn,
            task_id,
            kanban_db.complete_task(conn, task_id, summary=payload.summary if payload else None),
            "completed",
        )


@router.post("/tasks/{task_id}/block")
def block_task(
    task_id: str,
    payload: Optional[BlockRequest] = None,
    board: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        try:
            ok = kanban_db.block_task(
                conn,
                task_id,
                reason=payload.reason if payload else None,
                kind=payload.kind if payload else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _transition_response(conn, task_id, ok, "blocked")


@router.post("/tasks/{task_id}/unblock")
def unblock_task(task_id: str, board: Optional[str] = Query(default=None)) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        return _transition_response(
            conn, task_id, kanban_db.unblock_task(conn, task_id), "unblocked"
        )


@router.post("/tasks/{task_id}/archive")
def archive_task(task_id: str, board: Optional[str] = Query(default=None)) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        return _transition_response(
            conn, task_id, kanban_db.archive_task(conn, task_id), "archived"
        )


@router.post("/tasks/{parent_id}/links/{child_id}")
def link_tasks(
    parent_id: str,
    child_id: str,
    board: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    with _connection(board) as conn:
        try:
            kanban_db.link_tasks(conn, parent_id, child_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"parent_id": parent_id, "child_id": child_id, "linked": True}


@router.delete("/tasks/{parent_id}/links/{child_id}")
def unlink_tasks(
    parent_id: str,
    child_id: str,
    board: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, parent_id)
        _require_task(conn, child_id)
        removed = kanban_db.unlink_tasks(conn, parent_id, child_id)
        return {"parent_id": parent_id, "child_id": child_id, "removed": removed}


@router.get("/tasks/{task_id}/events")
def task_events(
    task_id: str,
    board: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        events = kanban_db.list_events(conn, task_id)[-limit:]
        items = [
            {
                "id": event.id,
                "task_id": event.task_id,
                "kind": event.kind,
                "created_at": event.created_at,
                "run_id": event.run_id,
            }
            for event in events
        ]
        return {"events": items, "count": len(items)}


@router.get("/tasks/{task_id}/runs")
def task_runs(
    task_id: str,
    board: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        runs = kanban_db.list_runs(conn, task_id)[-limit:]
        items = [
            {
                "id": run.id,
                "status": run.status,
                "outcome": run.outcome,
                "started_at": run.started_at,
                "ended_at": run.ended_at,
            }
            for run in runs
        ]
        return {"runs": items, "count": len(items)}


@router.get("/tasks/{task_id}/log")
def task_log(
    task_id: str,
    board: Optional[str] = Query(default=None),
    tail_bytes: int = Query(default=_DEFAULT_LOG_TAIL, ge=1, le=_MAX_LOG_TAIL),
) -> dict[str, Any]:
    slug = _resolve_board_slug(board)
    with _connection(slug) as conn:
        _require_task(conn, task_id)
    content = kanban_db.read_worker_log(task_id, tail_bytes=tail_bytes, board=slug)
    size = 0
    log_path = kanban_db.worker_log_path(task_id, board=slug)
    try:
        size = log_path.stat().st_size
    except OSError:
        pass
    return {
        "task_id": task_id,
        "exists": content is not None,
        "size_bytes": size,
        "tail_bytes": tail_bytes,
        "truncated": size > tail_bytes,
        "excerpt": _sanitize_log(content or ""),
    }

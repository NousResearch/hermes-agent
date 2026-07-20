"""Bounded, revision-aware Kanban records for the API-server bearer surface.

This module deliberately does not reuse the dashboard plugin API.  It exposes
only a small server-to-server contract and funnels writes through the existing
Kanban database helpers so CLI, dashboard, gateway, and API callers share the
same state-machine invariants.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import sqlite3
from typing import Any

from hermes_cli import kanban_db


KANBAN_API_VERSION = 1
MAX_BOARDS = 100
MAX_PROFILES = 200
MAX_TASKS = 200
MAX_COMMENTS = 100
MAX_RUNS = 100
MAX_EVENTS = 200
MAX_TEXT = 20_000
_BOARD_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_TASK_RE = re.compile(r"t_[a-f0-9]{8}\Z")
_PROFILE_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_IDEMPOTENCY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{7,127}\Z")
_REVISION_RE = re.compile(r"kanbanrev_[0-9a-f]{64}\Z")


class KanbanApiError(ValueError):
    """A deliberate public API failure with no reflected database details."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _text(value: Any, limit: int = MAX_TEXT) -> str:
    value = str(value or "").replace("\x00", "")
    value = "".join(ch for ch in value if ch.isprintable() or ch in "\n\t")
    return value[:limit]


def _optional_text(value: Any, limit: int) -> str | None:
    text = _text(value, limit).strip()
    return text or None


def _required_text(value: Any, label: str, limit: int) -> str:
    text = _optional_text(value, limit)
    if text is None:
        raise KanbanApiError(400, "invalid_request", f"{label} is required")
    return text


def _board(value: Any) -> str:
    board = _text(value, 64).strip().lower()
    if not _BOARD_RE.fullmatch(board):
        raise KanbanApiError(400, "invalid_board", "Board identifier is invalid")
    if board != kanban_db.DEFAULT_BOARD and not kanban_db.board_exists(board):
        raise KanbanApiError(404, "board_not_found", "Board was not found")
    return board


def _task_id(value: Any) -> str:
    task_id = _text(value, 16).strip()
    if not _TASK_RE.fullmatch(task_id):
        raise KanbanApiError(400, "invalid_task", "Task identifier is invalid")
    return task_id


def _profile(value: Any, *, allow_none: bool = True) -> str | None:
    if value is None and allow_none:
        return None
    profile = _text(value, 64).strip().lower()
    if not _PROFILE_RE.fullmatch(profile):
        raise KanbanApiError(400, "invalid_profile", "Profile identifier is invalid")
    return profile


def _idempotency_key(value: Any) -> str:
    key = _text(value, 128).strip()
    if not _IDEMPOTENCY_RE.fullmatch(key):
        raise KanbanApiError(400, "invalid_idempotency_key", "Idempotency key is invalid")
    return key


def _revision(value: Any) -> str:
    revision = _text(value, 80).strip()
    if not _REVISION_RE.fullmatch(revision):
        raise KanbanApiError(400, "invalid_revision", "Revision is invalid")
    return revision


def _canonical_hash(value: Any, *, prefix: str) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()


def _only_keys(payload: dict[str, Any], allowed: set[str]) -> None:
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise KanbanApiError(400, "invalid_request", "Request contains unsupported fields")


def _task_record(task: kanban_db.Task, *, include_body: bool) -> dict[str, Any]:
    return {
        "id": task.id,
        "object": "hermes.kanban.task",
        "title": _text(task.title, 500),
        **({"body": _text(task.body, MAX_TEXT)} if include_body else {}),
        "assignee": task.assignee,
        "status": task.status,
        "priority": int(task.priority),
        "created_by": _optional_text(task.created_by, 80),
        "created_at": int(task.created_at),
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "result": _text(task.result, MAX_TEXT) if include_body else None,
        "block_kind": task.block_kind,
        "block_recurrences": int(task.block_recurrences),
    }


def _comment_record(comment: kanban_db.Comment) -> dict[str, Any]:
    return {
        "id": int(comment.id),
        "author": _text(comment.author, 64),
        "body": _text(comment.body, 4_000),
        "created_at": int(comment.created_at),
    }


def _run_record(run: kanban_db.Run) -> dict[str, Any]:
    return {
        "id": int(run.id),
        "profile": _optional_text(run.profile, 64),
        "status": _text(run.status, 32),
        "outcome": _optional_text(run.outcome, 32),
        "summary": _text(run.summary, 4_000),
        "error": _text(run.error, 2_000),
        "started_at": int(run.started_at),
        "ended_at": run.ended_at,
    }


def _event_record(event: kanban_db.Event) -> dict[str, Any]:
    # Event payloads can contain worker paths, output, or arbitrary user text.
    # The public contract deliberately exposes only a bounded audit identity.
    return {
        "id": int(event.id),
        "kind": _text(event.kind, 80),
        "created_at": int(event.created_at),
        "run_id": event.run_id,
    }


def _task_edges(conn: sqlite3.Connection, task_id: str) -> dict[str, list[dict[str, Any]]]:
    parents = conn.execute(
        "SELECT t.id, t.status FROM tasks t JOIN task_links l ON l.parent_id = t.id "
        "WHERE l.child_id = ? ORDER BY t.id ASC LIMIT ?",
        (task_id, MAX_TASKS + 1),
    ).fetchall()
    children = conn.execute(
        "SELECT t.id, t.status FROM tasks t JOIN task_links l ON l.child_id = t.id "
        "WHERE l.parent_id = ? ORDER BY t.id ASC LIMIT ?",
        (task_id, MAX_TASKS + 1),
    ).fetchall()
    if len(parents) > MAX_TASKS or len(children) > MAX_TASKS:
        raise KanbanApiError(
            409,
            "task_relationship_limit",
            "Task relationships exceed the safe API limit",
        )
    return {
        "parents": [{"id": row["id"], "status": row["status"]} for row in parents],
        "children": [{"id": row["id"], "status": row["status"]} for row in children],
    }


def task_snapshot(conn: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    """Return the exact bounded state used for reads and mutation revisions."""
    task = kanban_db.get_task(conn, task_id)
    if task is None:
        raise KanbanApiError(404, "task_not_found", "Task was not found")
    comments = kanban_db.list_comments(conn, task_id)
    runs = kanban_db.list_runs(conn, task_id)
    events = kanban_db.list_events(conn, task_id)
    comments_truncated = len(comments) > MAX_COMMENTS
    runs_truncated = len(runs) > MAX_RUNS
    events_truncated = len(events) > MAX_EVENTS
    snapshot = {
        "object": "hermes.kanban.task_detail",
        "version": KANBAN_API_VERSION,
        "task": _task_record(task, include_body=True),
        "comments": [_comment_record(value) for value in comments[-MAX_COMMENTS:]],
        "runs": [_run_record(value) for value in runs[-MAX_RUNS:]],
        "events": [_event_record(value) for value in events[-MAX_EVENTS:]],
        "truncated": {
            "comments": comments_truncated,
            "runs": runs_truncated,
            "events": events_truncated,
        },
        **_task_edges(conn, task_id),
    }
    snapshot["revision"] = _canonical_hash(snapshot, prefix="kanbanrev_")
    return snapshot


def _request_hash(operation: str, board: str, task_id: str | None, payload: dict[str, Any]) -> str:
    return _canonical_hash(
        {"operation": operation, "board": board, "task_id": task_id, "payload": payload},
        prefix="sha256:",
    )


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _idempotency_replay(
    conn: sqlite3.Connection, *, key: str, request_hash: str
) -> tuple[bool, str | None]:
    row = conn.execute(
        "SELECT request_hash, task_id FROM kanban_api_idempotency WHERE key_hash = ?",
        (_key_hash(key),),
    ).fetchone()
    if row is None:
        return False, None
    if not hmac.compare_digest(str(row["request_hash"]), request_hash):
        raise KanbanApiError(409, "idempotency_conflict", "Idempotency key was already used for different input")
    return True, str(row["task_id"])


def _record_idempotency(conn: sqlite3.Connection, *, key: str, request_hash: str, task_id: str) -> None:
    conn.execute(
        "INSERT INTO kanban_api_idempotency (key_hash, request_hash, task_id, created_at) "
        "VALUES (?, ?, ?, strftime('%s','now'))",
        (_key_hash(key), request_hash, task_id),
    )


def list_boards() -> dict[str, Any]:
    boards = kanban_db.list_boards(include_archived=True)
    if len(boards) > MAX_BOARDS:
        raise KanbanApiError(409, "board_limit_exceeded", "Too many boards to return safely")
    return {
        "object": "list",
        "version": KANBAN_API_VERSION,
        "complete": True,
        "data": [
            {
                "id": _board(value.get("slug")),
                "object": "hermes.kanban.board",
                "name": _text(value.get("name") or value.get("slug"), 160),
                "archived": bool(value.get("archived")),
                "is_current": bool(value.get("is_current")),
            }
            for value in boards
        ],
    }


def list_profiles(board_value: Any) -> dict[str, Any]:
    board = _board(board_value)
    with kanban_db.connect_closing(board=board) as conn:
        profiles = kanban_db.known_assignees(conn)
    if len(profiles) > MAX_PROFILES:
        raise KanbanApiError(409, "profile_limit_exceeded", "Too many profiles to return safely")
    return {
        "object": "list",
        "version": KANBAN_API_VERSION,
        "complete": True,
        "board": board,
        "data": [
            {
                "id": _profile(value.get("name"), allow_none=False),
                "object": "hermes.kanban.profile",
                "available": bool(value.get("on_disk")),
                "counts": {
                    _text(status, 32): max(0, int(count))
                    for status, count in dict(value.get("counts") or {}).items()
                },
            }
            for value in profiles
        ],
    }


def list_tasks(board_value: Any, *, status: Any = None, assignee: Any = None, limit: Any = None) -> dict[str, Any]:
    board = _board(board_value)
    try:
        requested_limit = MAX_TASKS if limit is None else int(limit)
    except (TypeError, ValueError):
        raise KanbanApiError(400, "invalid_limit", "Task limit is invalid") from None
    if requested_limit < 1 or requested_limit > MAX_TASKS:
        raise KanbanApiError(400, "invalid_limit", "Task limit is invalid")
    status_value = _optional_text(status, 32)
    if status_value is not None and status_value not in kanban_db.VALID_STATUSES:
        raise KanbanApiError(400, "invalid_status", "Task status is invalid")
    with kanban_db.connect_closing(board=board) as conn:
        tasks = kanban_db.list_tasks(
            conn,
            status=status_value,
            assignee=_profile(assignee) if assignee is not None else None,
            include_archived=status_value == "archived",
            limit=requested_limit + 1,
        )
    truncated = len(tasks) > requested_limit
    return {
        "object": "list",
        "version": KANBAN_API_VERSION,
        "complete": not truncated,
        "board": board,
        "data": [_task_record(task, include_body=False) for task in tasks[:requested_limit]],
    }


def get_task(board_value: Any, task_value: Any) -> dict[str, Any]:
    board, task_id = _board(board_value), _task_id(task_value)
    with kanban_db.connect_closing(board=board) as conn:
        conn.execute("BEGIN")
        try:
            snapshot = task_snapshot(conn, task_id)
        finally:
            conn.execute("ROLLBACK")
    snapshot["board"] = board
    return snapshot


def create_task(board_value: Any, payload: dict[str, Any]) -> dict[str, Any]:
    board = _board(board_value)
    _only_keys(payload, {"title", "body", "assignee", "workspace_kind", "priority", "idempotency_key"})
    title = _required_text(payload.get("title"), "Title", 500)
    body = _optional_text(payload.get("body"), MAX_TEXT)
    assignee = _profile(payload.get("assignee"))
    workspace_kind = _text(payload.get("workspace_kind") or "scratch", 16).strip()
    if workspace_kind not in {"scratch", "worktree"}:
        raise KanbanApiError(400, "invalid_workspace_kind", "Workspace kind is invalid")
    try:
        priority = int(payload.get("priority", 0))
    except (TypeError, ValueError):
        raise KanbanApiError(400, "invalid_priority", "Priority is invalid") from None
    if priority < -1000 or priority > 1000:
        raise KanbanApiError(400, "invalid_priority", "Priority is invalid")
    key = _idempotency_key(payload.get("idempotency_key"))
    material = {
        "title": title, "body": body, "assignee": assignee,
        "workspace_kind": workspace_kind, "priority": priority,
    }
    request_hash = _request_hash("create", board, None, material)
    with kanban_db.connect_closing(board=board) as conn, kanban_db.write_txn(conn):
        replay, existing_id = _idempotency_replay(conn, key=key, request_hash=request_hash)
        if replay:
            return {"replayed": True, **task_snapshot(conn, existing_id or "")}
        task_id = kanban_db.create_task(
            conn, title=title, body=body, assignee=assignee, priority=priority,
            workspace_kind=workspace_kind, created_by="api_server",
            idempotency_key="api:" + _key_hash(key), board=board,
        )
        _record_idempotency(conn, key=key, request_hash=request_hash, task_id=task_id)
        return {"replayed": False, **task_snapshot(conn, task_id)}


def mutate_task(board_value: Any, task_value: Any, payload: dict[str, Any]) -> dict[str, Any]:
    board, task_id = _board(board_value), _task_id(task_value)
    _only_keys(
        payload,
        {"action", "expected_revision", "idempotency_key", "assignee", "author", "body", "reason", "kind"},
    )
    action = _text(payload.get("action"), 32).strip().lower()
    if action not in {"assign", "comment", "reply", "promote", "block", "retry", "terminate"}:
        raise KanbanApiError(400, "invalid_action", "Action is not supported")
    expected_revision = _revision(payload.get("expected_revision"))
    key = _idempotency_key(payload.get("idempotency_key"))
    material = {key: value for key, value in payload.items() if key != "idempotency_key"}
    request_hash = _request_hash(action, board, task_id, material)
    with kanban_db.connect_closing(board=board) as conn, kanban_db.write_txn(conn):
        replay, existing_id = _idempotency_replay(conn, key=key, request_hash=request_hash)
        if replay:
            if existing_id != task_id:
                raise KanbanApiError(409, "idempotency_conflict", "Idempotency key belongs to another task")
            return {"replayed": True, **task_snapshot(conn, task_id)}
        current = task_snapshot(conn, task_id)
        if not hmac.compare_digest(current["revision"], expected_revision):
            raise KanbanApiError(409, "stale_revision", "Task changed; refresh it before acting")
        if action == "assign":
            changed = kanban_db.assign_task(conn, task_id, _profile(payload.get("assignee")))
        elif action in {"comment", "reply"}:
            body = _required_text(payload.get("body"), "Comment", 4_000)
            author = _profile(payload.get("author") or "api_client", allow_none=False)
            kanban_db.add_comment(conn, task_id, author or "api_client", body)
            changed = True
        elif action == "promote":
            changed, _reason = kanban_db.promote_task(
                conn, task_id, actor="api_server", reason=_optional_text(payload.get("reason"), 2_000)
            )
        elif action == "block":
            kind = _text(payload.get("kind") or "needs_input", 32).strip()
            if kind not in kanban_db.VALID_BLOCK_KINDS:
                raise KanbanApiError(400, "invalid_block_kind", "Block kind is invalid")
            changed = kanban_db.block_task(
                conn, task_id, reason=_required_text(payload.get("reason"), "Reason", 2_000), kind=kind
            )
        elif action == "retry":
            changed = kanban_db.unblock_task(conn, task_id)
        else:
            changed = kanban_db.reclaim_task(
                conn, task_id, reason=_optional_text(payload.get("reason"), 2_000)
            )
        if not changed:
            raise KanbanApiError(409, "mutation_refused", "Task cannot accept that action in its current state")
        _record_idempotency(conn, key=key, request_hash=request_hash, task_id=task_id)
        return {"replayed": False, **task_snapshot(conn, task_id)}

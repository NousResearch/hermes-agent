"""Markdown idea store — SQLite index + per-board ``.md`` files under ``~/.hermes/ideas``.

Each idea belongs to a Kanban board slug (project boundary). The dashboard
plugin and ``hermes ideas`` / ``ideas_*`` tools share this module.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home
from hermes_cli import kanban_db

IDEA_STATUSES = frozenset({"draft", "active", "parked", "converted", "archived"})
DEFAULT_STATUS = "draft"
DB_NAME = "ideas.db"


class IdeasError(ValueError):
    """Raised for user-facing validation failures."""


class IdeaNotFoundError(IdeasError):
    pass


def _root() -> Path:
    root = get_hermes_home() / "ideas"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _db_path() -> Path:
    return _root() / DB_NAME


def _now() -> int:
    return int(time.time())


def _slugify(value: str, fallback: str = "idea") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", (value or "").strip().lower()).strip("-_")
    return slug[:80] or fallback


def resolve_board(board: Optional[str] = None) -> str:
    raw = (board or "").strip() or getattr(kanban_db, "DEFAULT_BOARD", "default")
    try:
        normed = kanban_db._normalize_board_slug(raw)
    except Exception as exc:
        raise IdeasError(str(exc)) from exc
    return normed or getattr(kanban_db, "DEFAULT_BOARD", "default")


def _board_dir(board: str) -> Path:
    path = _root() / "boards" / _slugify(board, "default")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _idea_path(board: str, idea_id: str, title: str) -> Path:
    return _board_dir(board) / f"{_slugify(title)}-{idea_id[:8]}.md"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ideas (
            id TEXT PRIMARY KEY,
            board TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT,
            status TEXT NOT NULL DEFAULT 'draft',
            tags TEXT NOT NULL DEFAULT '[]',
            task_id TEXT,
            file_path TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            archived_at INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ideas_board ON ideas(board, updated_at DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ideas_status ON ideas(status)")
    return conn


def _tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",")]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        tag = str(item).strip().lstrip("#")
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag[:40])
    return out[:24]


def _read_body(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def row_to_dict(row: sqlite3.Row, *, include_body: bool = False) -> dict[str, Any]:
    d = dict(row)
    try:
        d["tags"] = json.loads(d.get("tags") or "[]")
    except Exception:
        d["tags"] = []
    if include_body:
        d["body"] = _read_body(d["file_path"])
    return d


def _write_body(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body or "", encoding="utf-8")


def list_boards() -> dict[str, Any]:
    try:
        kanban_db.init_db(board=getattr(kanban_db, "DEFAULT_BOARD", "default"))
    except Exception:
        pass
    try:
        boards = kanban_db.list_boards(include_archived=False)
    except Exception:
        boards = [{"slug": getattr(kanban_db, "DEFAULT_BOARD", "default"), "name": "Default"}]
    current = kanban_db.get_current_board()
    with _connect() as conn:
        counts = {
            r["board"]: int(r["n"])
            for r in conn.execute(
                "SELECT board, COUNT(*) AS n FROM ideas WHERE status != 'archived' GROUP BY board"
            )
        }
    for b in boards:
        b["idea_count"] = counts.get(b.get("slug"), 0)
        b["is_current"] = b.get("slug") == current
    default_slug = getattr(kanban_db, "DEFAULT_BOARD", "default")
    if not any(b.get("slug") == default_slug for b in boards):
        boards.insert(
            0,
            {
                "slug": default_slug,
                "name": "Default",
                "idea_count": counts.get(default_slug, 0),
            },
        )
    return {"boards": boards, "current": current}


def list_ideas(
    *,
    board: Optional[str] = None,
    status: Optional[str] = None,
    q: Optional[str] = None,
    tag: Optional[str] = None,
    include_archived: bool = False,
) -> dict[str, Any]:
    board = resolve_board(board)
    clauses = ["board = ?"]
    params: list[Any] = [board]
    if status:
        if status not in IDEA_STATUSES:
            raise IdeasError("invalid status")
        clauses.append("status = ?")
        params.append(status)
    elif not include_archived:
        clauses.append("status != 'archived'")
    if q:
        like = f"%{q.strip()}%"
        clauses.append("(title LIKE ? OR summary LIKE ?)")
        params.extend([like, like])
    sql = "SELECT * FROM ideas WHERE " + " AND ".join(clauses) + " ORDER BY updated_at DESC"
    with _connect() as conn:
        rows = [row_to_dict(r) for r in conn.execute(sql, params).fetchall()]
    if tag:
        wanted = tag.strip().lstrip("#").casefold()
        rows = [r for r in rows if any(str(t).casefold() == wanted for t in r["tags"])]
    return {"board": board, "ideas": rows, "statuses": sorted(IDEA_STATUSES)}


def list_ideas_all_boards(
    *,
    status: Optional[str] = None,
    q: Optional[str] = None,
    tag: Optional[str] = None,
    include_archived: bool = False,
) -> dict[str, Any]:
    """List ideas across every board (one query)."""
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        if status not in IDEA_STATUSES:
            raise IdeasError("invalid status")
        clauses.append("status = ?")
        params.append(status)
    elif not include_archived:
        clauses.append("status != 'archived'")
    if q:
        like = f"%{q.strip()}%"
        clauses.append("(title LIKE ? OR summary LIKE ?)")
        params.extend([like, like])
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = "SELECT * FROM ideas" + where + " ORDER BY board ASC, updated_at DESC"
    with _connect() as conn:
        rows = [row_to_dict(r) for r in conn.execute(sql, params).fetchall()]
    if tag:
        wanted = tag.strip().lstrip("#").casefold()
        rows = [r for r in rows if any(str(t).casefold() == wanted for t in r["tags"])]
    by_board: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_board.setdefault(row["board"], []).append(row)
    return {
        "all_boards": True,
        "ideas": rows,
        "count": len(rows),
        "boards": sorted(by_board.keys()),
        "by_board": by_board,
        "statuses": sorted(IDEA_STATUSES),
    }


def create_idea(
    *,
    title: str,
    body: str = "",
    summary: Optional[str] = None,
    status: str = DEFAULT_STATUS,
    tags: Optional[list[str]] = None,
    board: Optional[str] = None,
) -> dict[str, Any]:
    board = resolve_board(board)
    status = status if status in IDEA_STATUSES else DEFAULT_STATUS
    idea_id = "i_" + uuid.uuid4().hex[:16]
    now = _now()
    path = _idea_path(board, idea_id, title)
    _write_body(path, body)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO ideas (id, board, title, summary, status, tags, file_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                idea_id,
                board,
                title.strip(),
                summary,
                status,
                json.dumps(_tags(tags)),
                str(path),
                now,
                now,
            ),
        )
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
    return row_to_dict(row, include_body=True)


def get_idea(idea_id: str, *, include_body: bool = True) -> dict[str, Any]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
    if row is None:
        raise IdeaNotFoundError("idea not found")
    return row_to_dict(row, include_body=include_body)


def update_idea(
    idea_id: str,
    *,
    title: Optional[str] = None,
    body: Optional[str] = None,
    summary: Optional[str] = None,
    status: Optional[str] = None,
    tags: Optional[list[str]] = None,
    task_id: Optional[str] = None,
) -> dict[str, Any]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        if row is None:
            raise IdeaNotFoundError("idea not found")
        data = row_to_dict(row)
        new_title = (title if title is not None else data["title"]).strip()
        new_status = status if status is not None else data["status"]
        if new_status not in IDEA_STATUSES:
            raise IdeasError("invalid status")
        new_summary = summary if summary is not None else data.get("summary")
        new_tags = _tags(tags if tags is not None else data.get("tags"))
        new_task_id = task_id if task_id is not None else data.get("task_id")
        file_path = Path(data["file_path"])
        if body is not None:
            _write_body(file_path, body)
        archived_at = (
            _now()
            if new_status == "archived" and data.get("status") != "archived"
            else data.get("archived_at")
        )
        conn.execute(
            """
            UPDATE ideas
            SET title = ?, summary = ?, status = ?, tags = ?, task_id = ?, updated_at = ?, archived_at = ?
            WHERE id = ?
            """,
            (
                new_title,
                new_summary,
                new_status,
                json.dumps(new_tags),
                new_task_id,
                _now(),
                archived_at,
                idea_id,
            ),
        )
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
    return row_to_dict(row, include_body=True)


def delete_idea(idea_id: str, *, delete_file: bool = True) -> None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        if row is None:
            raise IdeaNotFoundError("idea not found")
        conn.execute("DELETE FROM ideas WHERE id = ?", (idea_id,))
    if delete_file:
        try:
            Path(row["file_path"]).unlink(missing_ok=True)
        except Exception:
            pass


def duplicate_idea(idea_id: str) -> dict[str, Any]:
    source = get_idea(idea_id)
    return create_idea(
        title=f"{source['title']} copy",
        body=source.get("body") or "",
        summary=source.get("summary"),
        status="draft",
        tags=source.get("tags") or [],
        board=source["board"],
    )


def convert_to_task(
    idea_id: str,
    *,
    assignee: Optional[str] = None,
    priority: int = 0,
    triage: bool = True,
    tenant: Optional[str] = None,
) -> dict[str, Any]:
    idea = get_idea(idea_id)
    board = resolve_board(idea["board"])
    try:
        kanban_db.init_db(board=board)
        conn = kanban_db.connect(board=board)
        try:
            body = (
                f"# Idea draft: {idea['title']}\n\n"
                f"Source idea: {idea['id']}\n"
                f"Markdown file: {idea['file_path']}\n\n"
                f"## Summary\n{idea.get('summary') or ''}\n\n"
                f"## Draft\n{idea.get('body') or ''}\n"
            )
            task_id = kanban_db.create_task(
                conn,
                title=idea["title"],
                body=body,
                assignee=assignee,
                created_by="ideas",
                workspace_kind="scratch",
                tenant=tenant,
                priority=priority,
                triage=triage,
                idempotency_key=f"idea:{idea_id}",
            )
        finally:
            conn.close()
    except Exception as exc:
        raise IdeasError(str(exc)) from exc
    updated = update_idea(idea_id, status="converted", task_id=task_id)
    return {"task_id": task_id, "idea": updated, "board": board}

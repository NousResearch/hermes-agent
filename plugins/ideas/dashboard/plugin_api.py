"""Ideas dashboard plugin API.

Stores project-scoped markdown ideas under ``$HERMES_HOME/ideas``.  The
project boundary is the Kanban board slug: each board gets its own idea list,
and omitted/invalid board selections fall back to the default/scratch board.

The API intentionally keeps markdown bodies as ordinary ``.md`` files next to a
small SQLite index.  That makes the drafts easy for future agents to consume via
filesystem tools while the dashboard gets fast filtering and metadata updates.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from hermes_constants import get_hermes_home
from hermes_cli import kanban_db

router = APIRouter()

IDEA_STATUSES = {"draft", "active", "parked", "converted", "archived"}
DEFAULT_STATUS = "draft"
DB_NAME = "ideas.db"


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


def _resolve_board(board: Optional[str]) -> str:
    raw = (board or "").strip() or getattr(kanban_db, "DEFAULT_BOARD", "default")
    try:
        normed = kanban_db._normalize_board_slug(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
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


def _row_to_dict(row: sqlite3.Row, *, include_body: bool = False) -> dict[str, Any]:
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


class IdeaBody(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    body: str = ""
    summary: Optional[str] = None
    status: str = DEFAULT_STATUS
    tags: list[str] = Field(default_factory=list)


class IdeaUpdateBody(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    body: Optional[str] = None
    summary: Optional[str] = None
    status: Optional[str] = None
    tags: Optional[list[str]] = None
    task_id: Optional[str] = None


class ConvertBody(BaseModel):
    assignee: Optional[str] = None
    priority: int = 0
    triage: bool = True
    tenant: Optional[str] = None


@router.get("/boards")
def list_boards():
    """Return Kanban boards with idea counts for the selector."""
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
    if not any(b.get("slug") == getattr(kanban_db, "DEFAULT_BOARD", "default") for b in boards):
        boards.insert(0, {"slug": getattr(kanban_db, "DEFAULT_BOARD", "default"), "name": "Default", "idea_count": counts.get("default", 0)})
    return {"boards": boards, "current": current}


@router.get("/ideas")
def list_ideas(
    board: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    q: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
    include_archived: bool = Query(False),
):
    board = _resolve_board(board)
    clauses = ["board = ?"]
    params: list[Any] = [board]
    if status:
        if status not in IDEA_STATUSES:
            raise HTTPException(status_code=400, detail="invalid status")
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
        rows = [_row_to_dict(r) for r in conn.execute(sql, params).fetchall()]
    if tag:
        wanted = tag.strip().lstrip("#").casefold()
        rows = [r for r in rows if any(str(t).casefold() == wanted for t in r["tags"])]
    return {"board": board, "ideas": rows, "statuses": sorted(IDEA_STATUSES)}


@router.post("/ideas")
def create_idea(payload: IdeaBody, board: Optional[str] = Query(None)):
    board = _resolve_board(board)
    status = payload.status if payload.status in IDEA_STATUSES else DEFAULT_STATUS
    idea_id = "i_" + uuid.uuid4().hex[:16]
    now = _now()
    path = _idea_path(board, idea_id, payload.title)
    _write_body(path, payload.body)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO ideas (id, board, title, summary, status, tags, file_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (idea_id, board, payload.title.strip(), payload.summary, status, json.dumps(_tags(payload.tags)), str(path), now, now),
        )
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
    return {"idea": _row_to_dict(row, include_body=True)}


@router.get("/ideas/{idea_id}")
def get_idea(idea_id: str):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="idea not found")
    return {"idea": _row_to_dict(row, include_body=True)}


@router.put("/ideas/{idea_id}")
def update_idea(idea_id: str, payload: IdeaUpdateBody):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="idea not found")
        data = _row_to_dict(row)
        title = (payload.title if payload.title is not None else data["title"]).strip()
        status = payload.status if payload.status is not None else data["status"]
        if status not in IDEA_STATUSES:
            raise HTTPException(status_code=400, detail="invalid status")
        summary = payload.summary if payload.summary is not None else data.get("summary")
        tags = _tags(payload.tags if payload.tags is not None else data.get("tags"))
        task_id = payload.task_id if payload.task_id is not None else data.get("task_id")
        file_path = Path(data["file_path"])
        if payload.body is not None:
            _write_body(file_path, payload.body)
        archived_at = _now() if status == "archived" and data.get("status") != "archived" else data.get("archived_at")
        conn.execute(
            """
            UPDATE ideas
            SET title = ?, summary = ?, status = ?, tags = ?, task_id = ?, updated_at = ?, archived_at = ?
            WHERE id = ?
            """,
            (title, summary, status, json.dumps(tags), task_id, _now(), archived_at, idea_id),
        )
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
    return {"idea": _row_to_dict(row, include_body=True)}


@router.delete("/ideas/{idea_id}")
def delete_idea(idea_id: str, delete_file: bool = Query(True)):
    with _connect() as conn:
        row = conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="idea not found")
        conn.execute("DELETE FROM ideas WHERE id = ?", (idea_id,))
    if delete_file:
        try:
            Path(row["file_path"]).unlink(missing_ok=True)
        except Exception:
            pass
    return {"ok": True}


@router.post("/ideas/{idea_id}/duplicate")
def duplicate_idea(idea_id: str):
    source = get_idea(idea_id)["idea"]
    return create_idea(
        IdeaBody(
            title=f"{source['title']} copy",
            body=source.get("body") or "",
            summary=source.get("summary"),
            status="draft",
            tags=source.get("tags") or [],
        ),
        board=source["board"],
    )


@router.post("/ideas/{idea_id}/task")
def convert_to_task(idea_id: str, payload: ConvertBody):
    idea = get_idea(idea_id)["idea"]
    board = _resolve_board(idea["board"])
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
                assignee=payload.assignee,
                created_by="ideas-dashboard",
                workspace_kind="scratch",
                tenant=payload.tenant,
                priority=payload.priority,
                triage=payload.triage,
                idempotency_key=f"idea:{idea_id}",
            )
        finally:
            conn.close()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    updated = update_idea(idea_id, IdeaUpdateBody(status="converted", task_id=task_id))["idea"]
    return {"task_id": task_id, "idea": updated, "board": board}

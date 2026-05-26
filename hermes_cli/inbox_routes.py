"""Hermes Inbox HTTP API routes.

Exposes the Hermes inbox backend as a JSON HTTP API under /api/inbox/.

Endpoints:
    GET  /inbox/list          — list messages (cursor pag, newest-first, filter)
    GET  /inbox/get/{id}      — get one message by id
    POST /inbox/mark-read     — mark one or batch messages as read
    GET  /inbox/unread-count  — count unread messages

Schema: hermes.inbox.message.v1
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from hermes_constants import get_hermes_home

# ---------------------------------------------------------------------------
# Inbox backend (copied from hermes-mcp-server app/tools/hermes_inbox.py)
# ---------------------------------------------------------------------------

INBOX_SUBDIR = "_control/agents/hermes/queue/inbox"
VALID_LEVELS = frozenset({"info", "warning", "error"})


class HermesInboxError(RuntimeError):
    """Raised when an inbox message is invalid or cannot be written/read."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_message_id(source: str, run_id: str) -> str:
    prefix = f"{source}-{run_id}".replace("/", "-").replace(" ", "-")
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _message_path(inbox_dir: Path, message_id: str) -> Path:
    if not message_id or "/" in message_id or ".." in message_id:
        raise HermesInboxError("invalid_message_id")
    return inbox_dir / f"{message_id}.json"


def _get_inbox_dir() -> Path:
    return Path(get_hermes_home()) / INBOX_SUBDIR


def _read_message_file(path: Path) -> dict[str, Any] | None:
    try:
        message = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if message.get("schema") != "hermes.inbox.message.v1":
        return None
    message["_path"] = str(path)
    return message


def list_inbox_messages(
    *,
    inbox_dir: Path,
    limit: int = 20,
    cursor: str | None = None,
    unread_only: bool = False,
    source: str | None = None,
    level: str | None = None,
) -> dict[str, Any]:
    """Read recent inbox messages, newest first, with optional filters + cursor pagination."""
    root = Path(inbox_dir)
    if not root.exists():
        return {"messages": [], "next_cursor": None, "total": 0}

    if limit <= 0:
        return {"messages": [], "next_cursor": None, "total": 0}

    # Collect all messages, newest first. Cursor is the 0-based index of the
    # last item on the previous page, encoded as a string. We skip all items
    # at or before that index.
    all_messages: list[tuple[float, str, dict[str, Any]]] = []
    for path in sorted(root.glob("*.json"), key=lambda p: (-p.stat().st_mtime, p.name)):
        message = _read_message_file(path)
        if message is None:
            continue
        if unread_only and bool(message.get("read")):
            continue
        if source is not None and message.get("source") != source:
            continue
        if level is not None and message.get("level") != level:
            continue
        all_messages.append((path.stat().st_mtime, path.name, message))

    total = len(all_messages)  # save before slicing
    # Apply cursor (cursor = 0-based index of last item on previous page)
    offset = 0
    if cursor:
        try:
            offset = int(cursor) + 1
        except ValueError:
            pass  # Invalid cursor, return from start
    all_messages = all_messages[offset:]

    # Slice page
    page_messages = all_messages[:limit]
    next_cursor: str | None = None
    if len(all_messages) > limit:
        next_cursor = str(offset + limit)  # skip all items on this page on next call

    messages = [msg for _, _, msg in page_messages]

    return {
        "messages": messages,
        "next_cursor": next_cursor,
        "total": total,
    }


def get_inbox_message(*, inbox_dir: Path, message_id: str) -> dict[str, Any] | None:
    """Return one inbox message by id, or None if absent/invalid."""
    path = _message_path(inbox_dir, message_id)
    if not path.exists():
        return None
    return _read_message_file(path)


def mark_inbox_message_read(*, inbox_dir: Path, message_id: str) -> dict[str, Any]:
    """Mark one inbox message as read."""
    path = _message_path(inbox_dir, message_id)
    if not path.exists():
        raise HermesInboxError("message_not_found")
    message = _read_message_file(path)
    if message is None:
        raise HermesInboxError("invalid_message_file")
    message.pop("_path", None)
    message["read"] = True
    message["read_at"] = _utc_now()
    path.write_text(json.dumps(message, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "message_path": str(path), "message": message}


def get_unread_count(*, inbox_dir: Path) -> int:
    """Count unread messages."""
    root = Path(inbox_dir)
    if not root.exists():
        return 0
    count = 0
    for path in root.glob("*.json"):
        message = _read_message_file(path)
        if message and not bool(message.get("read")):
            count += 1
    return count


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["inbox"])


@router.get("/list")
def inbox_list(
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = Query(default=None),
    unread_only: bool = Query(default=False),
    source: str | None = Query(default=None),
    level: str | None = Query(default=None),
) -> dict[str, Any]:
    """List inbox messages, newest-first, with cursor pagination and filters."""
    if level is not None and level not in VALID_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid level. Must be one of: {', '.join(sorted(VALID_LEVELS))}",
        )
    inbox_dir = _get_inbox_dir()
    return list_inbox_messages(
        inbox_dir=inbox_dir,
        limit=limit,
        cursor=cursor,
        unread_only=unread_only,
        source=source,
        level=level,
    )


@router.get("/get/{message_id}")
def inbox_get(message_id: str) -> dict[str, Any]:
    """Get a single inbox message by id."""
    inbox_dir = _get_inbox_dir()
    message = get_inbox_message(inbox_dir=inbox_dir, message_id=message_id)
    if message is None:
        raise HTTPException(status_code=404, detail="message_not_found")
    return message


@router.post("/mark-read")
def inbox_mark_read(body: dict[str, Any]) -> dict[str, Any]:
    """Mark one or many messages as read.

    Body: {"message_ids": ["id1", "id2"]}  — batch
    Body: {"message_id": "id"}             — single
    """
    inbox_dir = _get_inbox_dir()
    message_ids: list[str] = []

    if "message_ids" in body:
        if not isinstance(body["message_ids"], list):
            raise HTTPException(status_code=400, detail="message_ids must be a list")
        message_ids = body["message_ids"]
    elif "message_id" in body:
        message_ids = [body["message_id"]]
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'message_id' or 'message_ids' in body",
        )

    if not message_ids:
        raise HTTPException(status_code=400, detail="empty message_ids list")

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for mid in message_ids:
        try:
            results.append(mark_inbox_message_read(inbox_dir=inbox_dir, message_id=mid))
        except HermesInboxError as exc:
            errors.append({"message_id": mid, "error": str(exc)})

    return {
        "ok": len(errors) == 0,
        "marked_count": len(results),
        "results": results,
        "errors": errors,
    }


@router.get("/unread-count")
def inbox_unread_count() -> dict[str, Any]:
    """Return the count of unread messages."""
    inbox_dir = _get_inbox_dir()
    return {"unread_count": get_unread_count(inbox_dir=inbox_dir)}

"""Kanban task attachments: size-capped blob writes, metadata rows, and
on-disk blob lifecycle.

Extracted verbatim from :mod:`hermes_cli.kanban_db` (godfile shard s2,
cluster c10). Public names are re-exported from :mod:`hermes_cli.kanban_db`
so existing ``import hermes_cli.kanban_db as kb`` callers are unaffected.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Verbatim section moved from hermes_cli/kanban_db.py (lines 3704-3927).
# The kanban_db.py -> sibling re-export cycle is import-safe because the
# parent-module imports at the bottom of this file are deferred to the end
# of the module (see kanban_db.py bottom re-export block).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------

# The attachment size cap is the module-level ``KANBAN_ATTACHMENT_MAX_BYTES``
# (defined near the top of this file) — one constant shared by the dashboard
# HTTP endpoint, the agent toolset, and the CLI so the limit cannot drift
# between surfaces.


class AttachmentTooLarge(ValueError):
    """Raised when an attachment exceeds the configured size cap.

    Subclasses :class:`ValueError` so generic ``except ValueError`` handlers
    (e.g. the dashboard's 400 fallback) still catch it, while callers that
    want a distinct user-facing message (the tool/CLI 413-equivalent) can
    catch it specifically.
    """


def _safe_attachment_name(raw: str) -> str:
    """Reduce a client-supplied filename to a safe basename.

    Strips any directory components (both separators) so a malicious
    ``../../etc/passwd`` or ``C:\\x`` collapses to its leaf. Drops control
    chars and leading dots so we never write a dotfile or a name with
    embedded NULs/newlines. Rejects empty / dotfile-only names. The result
    is only ever joined under the per-task attachments dir, never used
    verbatim as a path from the client.

    Raises :class:`ValueError` on an unusable name; HTTP callers map that
    to a 400.
    """
    name = (raw or "").replace("\\", "/").split("/")[-1].strip()
    name = "".join(ch for ch in name if ch.isprintable() and ch not in "\x00").strip()
    name = name.lstrip(".").strip()
    if not name:
        raise ValueError("invalid attachment filename")
    return name[:200]


def _collision_free_path(dest_dir: Path, safe_name: str) -> Path:
    """Return a path under ``dest_dir`` that doesn't clobber an existing file.

    ``foo.pdf`` → ``foo.pdf``, then ``foo (1).pdf``, ``foo (2).pdf``, …
    ``safe_name`` must already be sanitised via :func:`_safe_attachment_name`.
    """
    stem, dot, ext = safe_name.partition(".")
    candidate = safe_name
    n = 1
    while (dest_dir / candidate).exists():
        candidate = f"{stem} ({n}){dot}{ext}"
        n += 1
    return dest_dir / candidate


def store_attachment_bytes(
    conn: sqlite3.Connection,
    task_id: str,
    filename: str,
    data: bytes,
    *,
    content_type: Optional[str] = None,
    uploaded_by: Optional[str] = None,
    board: Optional[str] = None,
    max_bytes: Optional[int] = None,
) -> int:
    """Validate, size-check, persist a blob, and record its metadata row.

    This is the single write path shared by the dashboard endpoint, the
    agent toolset (``kanban_attach`` / ``kanban_attach_url``), and the CLI
    (``hermes kanban attach``) so name-sanitisation, the size cap, and the
    collision-resolution all behave identically everywhere.

    Steps: enforce ``max_bytes``, sanitise ``filename`` to a safe basename,
    write the bytes under :func:`task_attachments_dir` with a
    collision-free name, then insert the ``task_attachments`` row via
    :func:`add_attachment`. Returns the new attachment id.

    Raises :class:`AttachmentTooLarge` when ``data`` exceeds ``max_bytes``,
    or :class:`ValueError` for a bad filename / unknown task. On any failure
    after the blob is written (e.g. the task disappeared) the orphaned blob
    is removed before re-raising.
    """
    if max_bytes is None:
        max_bytes = KANBAN_ATTACHMENT_MAX_BYTES
    if len(data) > max_bytes:
        raise AttachmentTooLarge(
            f"attachment exceeds {max_bytes // (1024 * 1024)} MB limit"
        )
    safe_name = _safe_attachment_name(filename)
    dest_dir = task_attachments_dir(task_id, board=board)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = _collision_free_path(dest_dir, safe_name)
    dest_path.write_bytes(data)
    try:
        return add_attachment(
            conn,
            task_id,
            filename=dest_path.name,
            stored_path=str(dest_path.resolve()),
            content_type=content_type,
            size=len(data),
            uploaded_by=uploaded_by,
        )
    except Exception:
        # Don't leave an orphan blob if the metadata insert fails (most
        # commonly: the task id doesn't exist).
        try:
            dest_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def add_attachment(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    filename: str,
    stored_path: str,
    content_type: Optional[str] = None,
    size: int = 0,
    uploaded_by: Optional[str] = None,
) -> int:
    """Record a file attachment for a task. Returns the new attachment id.

    The caller is responsible for writing the blob to ``stored_path``
    first (under :func:`task_attachments_dir`); this only persists the
    metadata row and appends an ``attached`` event.
    """
    if not filename or not filename.strip():
        raise ValueError("attachment filename is required")
    if not stored_path or not stored_path.strip():
        raise ValueError("attachment stored_path is required")
    now = int(time.time())
    with write_txn(conn):
        if not conn.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
        ).fetchone():
            raise ValueError(f"unknown task {task_id}")
        cur = conn.execute(
            "INSERT INTO task_attachments "
            "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                filename.strip(),
                stored_path,
                content_type,
                int(size),
                uploaded_by,
                now,
            ),
        )
        _append_event(
            conn,
            task_id,
            "attached",
            {"filename": filename.strip(), "size": int(size), "by": uploaded_by},
        )
        return int(cur.lastrowid or 0)


def list_attachments(conn: sqlite3.Connection, task_id: str) -> list[Attachment]:
    rows = conn.execute(
        "SELECT * FROM task_attachments WHERE task_id = ? ORDER BY created_at ASC, id ASC",
        (task_id,),
    ).fetchall()
    return [
        Attachment(
            id=r["id"],
            task_id=r["task_id"],
            filename=r["filename"],
            stored_path=r["stored_path"],
            content_type=r["content_type"],
            size=r["size"] or 0,
            uploaded_by=r["uploaded_by"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


def get_attachment(conn: sqlite3.Connection, attachment_id: int) -> Optional[Attachment]:
    r = conn.execute(
        "SELECT * FROM task_attachments WHERE id = ?", (attachment_id,)
    ).fetchone()
    if r is None:
        return None
    return Attachment(
        id=r["id"],
        task_id=r["task_id"],
        filename=r["filename"],
        stored_path=r["stored_path"],
        content_type=r["content_type"],
        size=r["size"] or 0,
        uploaded_by=r["uploaded_by"],
        created_at=r["created_at"],
    )


def delete_attachment(conn: sqlite3.Connection, attachment_id: int) -> Optional[Attachment]:
    """Delete an attachment row and its on-disk blob. Returns the removed row.

    Returns ``None`` when no row matched. The blob is removed best-effort
    (a missing file is not an error); the metadata row is the source of
    truth for whether an attachment "exists".
    """
    with write_txn(conn):
        att = get_attachment(conn, attachment_id)
        if att is None:
            return None
        conn.execute("DELETE FROM task_attachments WHERE id = ?", (attachment_id,))
        _append_event(
            conn, att.task_id, "attachment_removed", {"filename": att.filename}
        )
    try:
        p = Path(att.stored_path)
        if p.is_file():
            p.unlink()
    except OSError:
        pass
    return att

# Names from the parent module (module-level constants / helpers that stay in
# kanban_db.py because tests and other clusters use them). Imported at the
# bottom of this module so the kanban_db <-> sibling re-export cycle resolves
# in any import order.
from hermes_cli.kanban_db import (  # noqa: E402
    Attachment,
    KANBAN_ATTACHMENT_MAX_BYTES,
    task_attachments_dir,
    write_txn,
    _append_event,
)

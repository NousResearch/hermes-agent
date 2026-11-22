"""Kanban attachment store lane: the single pre-store policy seam.

Every durable attachment ingress funnels through :func:`store_attachment_bytes`
(in-memory bytes: agent tools ``kanban_attach`` / ``kanban_attach_url`` and the CLI
``attach``) or :func:`store_attachment_file` (an already-staged file: the dashboard's
streamed upload). The completion / review artifact staging inside ``kanban_db`` runs
each copied file through :func:`run_attachment_policy` before its row exists.

The policy itself is the ``pre_attachment_store`` plugin hook: core ships the seam and
the fail-closed directive shape; the oracle (what counts as a secret, what to redact)
lives in operator-installed plugins. With no subscriber the store path is byte-identical
to its pre-hook form — no scan, no chmod, no hook machinery.

Moved out of ``kanban_db.py`` because that file is over the line-count ratchet and may
only shrink (root AGENTS.md); the policy seam's wiring had to make room by moving the
whole attachment write path, not by growing the facade.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sqlite3
from pathlib import Path
from typing import NoReturn, Optional

from hermes_cli import kanban_db as kb

logger = logging.getLogger(__name__)

ATTACHMENT_STORE_HOOK = "pre_attachment_store"
_BLOCK_MESSAGE_MAX = 500
_EVENT_MESSAGE_MAX = 200


class AttachmentTooLarge(ValueError):
    """Attachment over the size cap. A ``ValueError`` so generic 400 handlers
    still catch it while the tool/CLI can give a 413-style message."""


class AttachmentPolicyBlocked(ValueError):
    """Attachment refused by the ``pre_attachment_store`` policy hook before the
    row existed. A ``ValueError`` (mirrors :class:`AttachmentTooLarge`): the
    dashboard maps it to 400 and the tool wrapper reports it without a traceback.
    The message is the subscriber's own text — never content-derived by core."""


def _safe_attachment_name(raw: str) -> str:
    """Client filename -> safe basename: strip directories (both separators),
    control chars and leading dots (no dotfiles, no traversal); ValueError when
    nothing usable remains. Only ever joined under the per-task attachments dir."""
    name = (raw or "").replace("\\", "/").split("/")[-1].strip()
    name = "".join(ch for ch in name if ch.isprintable() and ch not in "\x00").strip()
    name = name.lstrip(".").strip()
    if not name:
        raise ValueError("invalid attachment filename")
    return name[:200]


def _collision_free_path(dest_dir: Path, safe_name: str) -> Path:
    """``foo.pdf`` -> ``foo.pdf``, ``foo (1).pdf``, ... first one that doesn't exist."""
    stem, dot, ext = safe_name.partition(".")
    candidate = safe_name
    n = 1
    while (dest_dir / candidate).exists():
        candidate = f"{stem} ({n}){dot}{ext}"
        n += 1
    return dest_dir / candidate


def attachment_policy_active() -> bool:
    """Hot-path short-circuit: False when nothing subscribes to the store hook, so
    an un-scanned deployment pays no chmod and no dispatch. Inspection failure
    counts as unconsumed — the hook itself still fail-closes on a raised callback."""
    try:
        from hermes_cli.lifecycle import has_hook

        return has_hook(ATTACHMENT_STORE_HOOK)
    except Exception:
        logger.debug("pre_attachment_store subscriber probe failed", exc_info=True)
        return False


def run_attachment_policy(
    path: Path, *, task_id: str, filename: str, size: int,
    content_type: Optional[str], source: str, uploaded_by: Optional[str],
    board: Optional[str],
) -> Optional[str]:
    """Fire ``pre_attachment_store`` against a staged blob before it is adopted.

    Returns ``None`` to proceed, or the subscriber's block message (truncated to
    :data:`_BLOCK_MESSAGE_MAX`). While a callback runs the file is mode 0600 — a
    not-yet-accepted blob never sits at the default mode during policy time;
    on return the original mode is restored. Timeout and raised callbacks yield
    a block directive from dispatch (the hook is fail-closed), so a broken
    scanner refuses stores instead of waving them through.
    """
    if not attachment_policy_active():
        return None
    path = Path(path)
    try:
        original_mode = path.stat().st_mode & 0o777
        os.chmod(path, 0o600)
    except OSError as exc:
        return f"attachment policy could not secure the staged file for scanning ({type(exc).__name__}); refusing to store"
    try:
        from hermes_cli.lifecycle import invoke_hook

        try:
            verdicts = invoke_hook(
                ATTACHMENT_STORE_HOOK, task_id=task_id, board=board, filename=filename,
                content_type=content_type, size=size, source=source,
                uploaded_by=uploaded_by, path=str(path),
            )
        except Exception as exc:
            logger.warning("pre_attachment_store dispatch failed", exc_info=True)
            return f"attachment policy could not run ({type(exc).__name__}); refusing to store"
        for verdict in verdicts:
            if isinstance(verdict, dict) and verdict.get("action") == "block":
                return str(verdict.get("message") or "blocked by attachment policy")[:_BLOCK_MESSAGE_MAX]
    finally:
        with contextlib.suppress(OSError):
            if path.exists():
                os.chmod(path, original_mode)
    return None


def _refuse(
    conn: sqlite3.Connection, task_id: str, path: Path, *,
    filename: str, source: str, message: str,
) -> NoReturn:
    """Refuse a staged blob: auditable event first (own write txn, the
    empty-completion-gate pattern — the refusal must survive the raise), then
    unlink, then surface the subscriber's message verbatim."""
    with kb.write_txn(conn):
        kb._append_event(
            conn, task_id, "attachment_policy_blocked",
            {"filename": filename, "source": source, "message": message[:_EVENT_MESSAGE_MAX]},
        )
    path.unlink(missing_ok=True)
    raise AttachmentPolicyBlocked(message)


def store_attachment_bytes(
    conn: sqlite3.Connection, task_id: str, filename: str, data: bytes, *,
    content_type: Optional[str] = None, uploaded_by: Optional[str] = None,
    board: Optional[str] = None, max_bytes: Optional[int] = None,
    source: str = "tool",
) -> int:
    """Single in-memory attachment write path (agent tools, CLI): size cap, safe
    basename, :func:`run_attachment_policy` gate, collision-free blob under
    :func:`hermes_cli.kanban_db.task_attachments_dir`, then the metadata row.
    Raises :class:`AttachmentTooLarge` / :class:`AttachmentPolicyBlocked` /
    ``ValueError``; a blob whose row insert fails is removed before re-raising.
    The row records the POST-hook on-disk size, so a subscriber that redacts the
    file in place cannot leave a lying ``size`` column. Returns the new id."""
    if max_bytes is None:
        max_bytes = kb.KANBAN_ATTACHMENT_MAX_BYTES
    if len(data) > max_bytes:
        raise AttachmentTooLarge(f"attachment exceeds {max_bytes // (1024 * 1024)} MB limit")
    kb._require_task(conn, task_id)
    safe_name = _safe_attachment_name(filename)
    dest_dir = kb.task_attachments_dir(task_id, board=board)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = _collision_free_path(dest_dir, safe_name)
    if attachment_policy_active():
        # Exclusive create: gate a real file, not the in-memory copy, so the
        # subscriber scans exactly the bytes that would become durable.
        with open(dest_path, "xb") as staged:
            staged.write(data)
        blocked = run_attachment_policy(
            dest_path, task_id=task_id, filename=dest_path.name, size=len(data),
            content_type=content_type, source=source, uploaded_by=uploaded_by, board=board,
        )
        if blocked is not None:
            _refuse(conn, task_id, dest_path, filename=dest_path.name, source=source, message=blocked)
    else:
        dest_path.write_bytes(data)
    try:
        return kb.add_attachment(
            conn, task_id, filename=dest_path.name, stored_path=str(dest_path.resolve()),
            content_type=content_type, size=dest_path.stat().st_size, uploaded_by=uploaded_by,
        )
    except Exception:
        # Don't leave an orphan blob if the metadata insert fails.
        with contextlib.suppress(OSError):
            dest_path.unlink(missing_ok=True)
        raise


def store_attachment_file(
    conn: sqlite3.Connection, task_id: str, filename: str, staged_path: Path, *,
    content_type: Optional[str] = None, uploaded_by: Optional[str] = None,
    board: Optional[str] = None, source: str = "dashboard",
) -> int:
    """Adopt a fully-staged upload file (the dashboard streams to a ``.part`` file
    in the task's attachments dir, then calls this): policy gate on the staged
    bytes, rename to the collision-free name, metadata row. Takes ownership of
    ``staged_path`` — it is removed on every failure path. Same exceptions as
    :func:`store_attachment_bytes`."""
    staged_path = Path(staged_path)
    try:
        kb._require_task(conn, task_id)
        safe_name = _safe_attachment_name(filename)  # validated before adopting the caller's file
        blocked = run_attachment_policy(
            staged_path, task_id=task_id, filename=safe_name, size=staged_path.stat().st_size,
            content_type=content_type, source=source, uploaded_by=uploaded_by, board=board,
        )
        if blocked is not None:
            _refuse(conn, task_id, staged_path, filename=safe_name, source=source, message=blocked)
        dest_dir = kb.task_attachments_dir(task_id, board=board)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = _collision_free_path(dest_dir, safe_name)
        os.replace(staged_path, dest_path)
    except Exception:
        with contextlib.suppress(OSError):
            staged_path.unlink(missing_ok=True)
        raise
    try:
        return kb.add_attachment(
            conn, task_id, filename=dest_path.name, stored_path=str(dest_path.resolve()),
            content_type=content_type, size=dest_path.stat().st_size, uploaded_by=uploaded_by,
        )
    except Exception:
        with contextlib.suppress(OSError):
            dest_path.unlink(missing_ok=True)
        raise

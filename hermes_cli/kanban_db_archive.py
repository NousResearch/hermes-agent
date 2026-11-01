"""Archive/unarchive: reversible board-level hiding, not deletion.

See ``company/kanban-archive-design-2026-10-13.md``. Split out of the
``kanban_db`` facade to offset that file's line-count growth (root
AGENTS.md: new behaviour never grows a facade past its cap); the facade
re-exports every public name at its tail so ``kanban_db.archive_task`` etc.
keep working unchanged for every caller.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional

# Archive is only ever allowed from a terminal human-facing status: a card
# someone could still be actively working, waiting for review on, or waiting
# to be dispatched must never vanish from the default board.
_ARCHIVABLE_SOURCE_STATUSES = frozenset({"done", "blocked"})

# Cap on how many active-descendant (id, status) pairs a refusal reports —
# the caller needs enough to act, not every row on a pathologically wide subtree.
_MAX_REPORTED_ACTIVE_DESCENDANTS = 20


@dataclass
class ArchiveResult:
    """Outcome of :func:`archive_task` / :func:`unarchive_task`.

    ``ok`` True on a successful transition; ``status`` is the task's status
    after the call (on refusal, its status at refusal time when known).
    ``error_code`` is set on refusal: ``not_found``, ``invalid_source_status``,
    ``active_claim``, ``active_descendants`` (``active_descendants`` then carries
    up to :data:`_MAX_REPORTED_ACTIVE_DESCENDANTS` ``(id, status)`` pairs),
    ``status_changed`` (lost the CAS to a concurrent writer), ``not_archived``,
    or ``legacy_archive_metadata_missing`` (archived before these columns
    existed; nothing to safely restore to).
    """

    ok: bool
    task_id: str
    status: Optional[str] = None
    error_code: Optional[str] = None
    active_descendants: list[tuple[str, str]] = field(default_factory=list)
    archived_from_status: Optional[str] = None
    archived_at: Optional[int] = None


def _active_descendants(conn: sqlite3.Connection, task_id: str) -> list[tuple[str, str]]:
    """``(id, status)`` for every recursive descendant of ``task_id`` whose status
    is not yet ``done``/``archived``. Walks the whole subtree (not just direct
    children) via a recursive CTE over ``task_links``, same shape as
    :func:`invalidate_descendants_for_parent_reopen`; ``task_links`` is a DAG
    guarded against cycles at link time, so the plain ``UNION`` (not ``UNION
    ALL``) is enough to dedupe a diamond without looping."""
    rows = conn.execute(
        """
        WITH RECURSIVE descendants(id) AS (
            SELECT child_id FROM task_links WHERE parent_id = ?
            UNION
            SELECT l.child_id
            FROM task_links l
            JOIN descendants d ON d.id = l.parent_id
        )
        SELECT t.id, t.status
        FROM descendants d
        JOIN tasks t ON t.id = d.id
        ORDER BY t.id
        """,
        (task_id,),
    ).fetchall()
    return [(row["id"], row["status"]) for row in rows if row["status"] not in ("done", "archived")]


def archive_task(conn: sqlite3.Connection, task_id: str, *, reason: str, actor: str) -> ArchiveResult:
    """Archive a ``done``/``blocked`` task with no active descendants.

    Reversible board-level hiding, not deletion: default list/stats/dispatcher
    keep excluding ``archived`` (unchanged contract), while ``include_archived``
    and ``get_task`` by id still see it. Unlike the legacy ``archive_task`` this
    can no longer be a channel that terminates a running worker (#76196's
    original fix): ``running``/``ready``/``review``/``triage``/``todo``/
    ``scheduled`` are refused outright with ``invalid_source_status`` before any
    write, and an active claim/run on a nominally done/blocked row (which
    should never happen, but TOCTOU is cheap insurance) refuses with
    ``active_claim`` rather than clearing it. Any recursive descendant that is
    not itself ``done``/``archived`` refuses with ``active_descendants`` — an
    archived parent is terminal for ``recompute_ready``/dependency gating, so
    hiding it while a child is still live would silently remove that child's
    upstream context from the default board. ``reason``/``actor`` are required
    and persisted on the row (``archived_from_status``/``archived_at``/
    ``archive_reason``/``archived_by``) alongside the ``archived`` event, so
    :func:`unarchive_task` restores from durable state, never from
    GC-prunable ``task_events``. Validate-then-write in one ``write_txn``
    closes the TOCTOU window between the read and the status flip.
    """
    from hermes_cli.kanban_db import _append_event, recompute_ready, write_txn
    from hermes_cli.kanban_db_workspace import _cleanup_workspace

    reason = (reason or "").strip()
    if not reason:
        raise ValueError("archive_task: reason is required")
    actor = (actor or "").strip()
    if not actor:
        raise ValueError("archive_task: actor is required")
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, claim_lock, current_run_id, worker_pid FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return ArchiveResult(ok=False, task_id=task_id, error_code="not_found")
        status = row["status"]
        if status not in _ARCHIVABLE_SOURCE_STATUSES:
            return ArchiveResult(ok=False, task_id=task_id, status=status, error_code="invalid_source_status")
        if row["claim_lock"] is not None or row["current_run_id"] is not None or row["worker_pid"] is not None:
            return ArchiveResult(ok=False, task_id=task_id, status=status, error_code="active_claim")
        active = _active_descendants(conn, task_id)
        if active:
            return ArchiveResult(
                ok=False, task_id=task_id, status=status, error_code="active_descendants",
                active_descendants=active[:_MAX_REPORTED_ACTIVE_DESCENDANTS],
            )
        now = int(time.time())
        cur = conn.execute(
            "UPDATE tasks SET status = 'archived', archived_from_status = ?, "
            "    archived_at = ?, archive_reason = ?, archived_by = ? "
            "WHERE id = ? AND status = ?",
            (status, now, reason, actor, task_id, status),
        )
        if cur.rowcount != 1:
            return ArchiveResult(ok=False, task_id=task_id, error_code="status_changed")
        _append_event(
            conn, task_id, "archived",
            {"from_status": status, "reason": reason, "actor": actor, "archived_at": now},
        )
    # ``archived`` parents no longer block children; promote any newly-satisfied ones now.
    recompute_ready(conn)
    # Reap the workspace on archive too (never-completed tasks kept it forever).
    _cleanup_workspace(conn, task_id)
    return ArchiveResult(
        ok=True, task_id=task_id, status="archived", archived_from_status=status, archived_at=now,
    )


def unarchive_task(conn: sqlite3.Connection, task_id: str, *, reason: str, actor: str) -> ArchiveResult:
    """Restore an archived task to its persisted pre-archive status.

    Restores EXACTLY from ``archived_from_status`` (``done`` stays ``done``,
    ``blocked`` stays ``blocked``) — never from ``task_events``, which ``kanban
    gc`` can prune, and never guessed. A legacy archived row seeded before
    these columns existed carries ``archived_from_status IS NULL`` and is
    refused as ``legacy_archive_metadata_missing`` rather than default to a
    restore target nothing here can verify. Deliberately skips
    :func:`recompute_ready`: archive only ever accepted a task with zero active
    descendants, so restoring it must not make any child runnable as a side
    effect — that happens only through the child's own normal path.
    """
    from hermes_cli.kanban_db import _append_event, write_txn

    reason = (reason or "").strip()
    if not reason:
        raise ValueError("unarchive_task: reason is required")
    actor = (actor or "").strip()
    if not actor:
        raise ValueError("unarchive_task: actor is required")
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, archived_from_status FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            return ArchiveResult(ok=False, task_id=task_id, error_code="not_found")
        if row["status"] != "archived":
            return ArchiveResult(ok=False, task_id=task_id, status=row["status"], error_code="not_archived")
        restore_status = row["archived_from_status"]
        if not restore_status:
            return ArchiveResult(
                ok=False, task_id=task_id, status="archived",
                error_code="legacy_archive_metadata_missing",
            )
        now = int(time.time())
        cur = conn.execute(
            "UPDATE tasks SET status = ? WHERE id = ? AND status = 'archived'",
            (restore_status, task_id),
        )
        if cur.rowcount != 1:
            return ArchiveResult(ok=False, task_id=task_id, error_code="status_changed")
        _append_event(
            conn, task_id, "unarchived",
            {"restored_status": restore_status, "reason": reason, "actor": actor, "unarchived_at": now},
        )
    return ArchiveResult(ok=True, task_id=task_id, status=restore_status)

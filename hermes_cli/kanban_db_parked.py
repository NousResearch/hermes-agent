"""The ``parked`` column — cards that are queued but cannot run yet.

A task lands in ``ready`` when its parents are done, whether or not anyone
can work on it. With no assignee the dispatcher skips it every tick
(``skipped_unassigned``), so the card sits in ``ready`` forever: the column
count says the board has work, and the card is invisible to anyone reading
the board as "ready = runnable".

``parked`` is that missing column. A card with no assignee belongs there, not
in ``ready``:

  * the board shows ``parked=20, ready=0 runnable`` instead of ``ready=20``
  * the dispatcher's ready lane holds only cards it can actually spawn
  * assigning a profile moves the card back to ``ready`` — routing is a
    first-class move, not a side effect of a dispatcher fallback

The dispatcher still honours ``kanban.default_assignee``: a parked card with
that fallback configured is routed on the next tick rather than waiting for a
human. What changes is that the *default* for a board with no fallback is
parking, not silently skipping.

A parked card remembers the lane it came from, so it returns there when it is
routed again: a card parked out of ``review`` goes back to ``review`` (it
still needs a reviewer), not to ``ready``.

Pure DB helpers + the column's transition rules. This module is imported
LATE by ``kanban_db``/``kanban_db_dispatch`` (they import each other), so it
must not import the facade at module level: it receives the connection and
does its own writes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Callable, Optional

# The column key. Kept in one place so the kernel, the workflow and the
# dashboard never drift on the spelling.
PARKED = "parked"

# Why a card was parked. The card is not broken — it is waiting for a
# decision about WHO runs it, which is a routing fact, not a failure.
PARK_REASON_NO_ASSIGNEE = "no_assignee"


def is_parked(status: Optional[str]) -> bool:
    """True when ``status`` is the parked column."""
    return status == PARKED


def unassigned_ready_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """``ready`` rows with no assignee — cards the dispatcher cannot spawn.

    Ordered like the dispatcher's ready lane (priority DESC, created_at ASC)
    so the parking pass promotes the same card the dispatcher would have
    reached first.
    """
    return conn.execute(
        "SELECT id, assignee, priority, created_at FROM tasks "
        "WHERE status = 'ready' AND claim_lock IS NULL "
        "  AND (assignee IS NULL OR assignee = '') "
        "ORDER BY priority DESC, created_at ASC",
    ).fetchall()


def park_task(
    conn: sqlite3.Connection, task_id: str, *, source_status: str = "ready",
    reason: str = PARK_REASON_NO_ASSIGNEE,
) -> bool:
    """Move an unassigned card to ``parked``; False when it did not move.

    A card that gained an assignee between the scan and this write is left
    alone: it is runnable now, and parking it would hide live work.
    ``source_status`` is the lane the card returns to when it is routed.
    """
    now = int(time.time())
    with conn:
        cur = conn.execute(
            "UPDATE tasks SET status = 'parked' "
            "WHERE id = ? AND status = ? "
            "  AND (assignee IS NULL OR assignee = '') AND claim_lock IS NULL",
            (task_id, source_status),
        )
        if cur.rowcount != 1:
            return False
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, 'parked', ?, ?)",
            (task_id, json.dumps({"reason": reason, "source_status": source_status}), now),
        )
    return True


def unpark_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """``parked`` -> the lane it was parked from; False when it did not move.

    The source lane comes from the ``parked`` event. A card parked out of
    ``ready`` returns to ``ready``; one parked out of ``review`` returns to
    ``review``, because it still needs a reviewer rather than a worker.
    """
    now = int(time.time())
    with conn:
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ? AND status = 'parked'", (task_id,),
        ).fetchone()
        if row is None:
            return False
        source = source_status(conn, task_id)
        cur = conn.execute(
            "UPDATE tasks SET status = ? WHERE id = ? AND status = 'parked'",
            (source, task_id),
        )
        if cur.rowcount != 1:
            return False
        conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, 'unparked', ?, ?)",
            (task_id, json.dumps({"status": source}), now),
        )
    return True


def source_status(conn: sqlite3.Connection, task_id: str) -> str:
    """The lane a parked card returns to (``ready`` when unknown).

    Read from the most recent ``parked`` event, so a card parked out of
    ``review`` goes back to ``review`` rather than to the worker lane.
    """
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'parked' "
        "ORDER BY created_at DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None or not row["payload"]:
        return "ready"
    try:
        payload = json.loads(row["payload"])
    except (TypeError, ValueError):
        return "ready"
    source = payload.get("source_status")
    return source if source in {"ready", "review"} else "ready"


def route_lane_cards(
    conn: sqlite3.Connection, rows: list[sqlite3.Row], *, source_status: str,
    default_assignee: Optional[str], dry_run: bool,
    apply_assignee: Callable[[str], bool],
) -> tuple[list[str], list[str]]:
    """Split one lane's cards into ``(routed, parked)``.

    ``routed`` = cards given the fallback assignee (they can now spawn).
    ``parked`` = cards nobody can run, moved to the ``parked`` column.

    The whole lane is swept REGARDLESS of the spawn budget: a card that
    cannot run must leave ``ready`` even when the tick has no room to
    spawn. The lane is ordered by priority, so an unassigned card sitting
    behind seven assigned ones was never reached at all, and the board kept
    advertising unrunnable work as ``ready``.
    """
    routed: list[str] = []
    parked: list[str] = []
    for row in rows:
        task_id = row["id"]
        if row["assignee"]:
            continue
        # Honour kanban.default_assignee so an unassigned task doesn't
        # park in 'ready' forever.
        if default_assignee and apply_assignee(task_id):
            routed.append(task_id)
            continue
        # No route and no fallback: park the card in the ``parked`` column
        # instead of leaving it in ``ready`` where it reads as runnable work
        # the board cannot deliver. A parked card is visible, countable, and
        # moves back to ``ready`` the moment it is assigned.
        park_unassigned(conn, task_id, source_status=source_status, dry_run=dry_run)
        parked.append(task_id)
    return routed, parked


def park_unassigned(
    conn: sqlite3.Connection, task_id: str, *, source_status: str, dry_run: bool,
) -> bool:
    """Park an unassigned card from ``source_status``; False when it did not move.

    The write is conditional (still in ``source_status``, still unassigned,
    still unclaimed), so a card that gained an assignee between the lane
    scan and this call is left alone. ``dry_run`` reports without writing.
    """
    if dry_run:
        return True
    return park_task(conn, task_id, source_status=source_status)


def route_parked_cards(
    conn: sqlite3.Connection, default_assignee: str, *, dry_run: bool,
    apply_assignee: Callable[[str], bool],
) -> list[str]:
    """Give every parked card the fallback assignee; return the routed ids.

    A parked card is a card the board could not run. With
    ``kanban.default_assignee`` configured it is routed again, so the
    fallback applies to cards parked before it was set.

    Routed cards land in the lane they were parked from (``ready`` or
    ``review``), so a review card still needs a reviewer. ``apply_assignee``
    persists the assignee (the caller owns the write); it returns False when
    the write failed, and the card stays parked.
    """
    rows = conn.execute(
        "SELECT id FROM tasks WHERE status = ? AND claim_lock IS NULL "
        "  AND (assignee IS NULL OR assignee = '')",
        (PARKED,),
    ).fetchall()
    routed: list[str] = []
    for row in rows:
        task_id = row["id"]
        if dry_run:
            routed.append(task_id)
            continue
        if not apply_assignee(task_id):
            continue
        if unpark_task(conn, task_id):
            routed.append(task_id)
    return routed

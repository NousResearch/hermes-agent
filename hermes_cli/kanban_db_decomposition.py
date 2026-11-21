"""Decomposition-only completion -> dependency wait (``kanban.evidence_gate`` wider than default).

An orchestrator card that only split its work into new cards and then handed off is not done;
refusing the handoff would deadlock because those cards usually wait on it. This module inverts
the links instead, in one write transaction.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Iterable, Optional

_log = logging.getLogger(__name__)


def _created_by(conn: sqlite3.Connection, task_id: str) -> set[str]:
    """Ids of tasks whose ``created`` event names ``task_id`` as ``creator_task_id``."""
    from hermes_cli.kanban_db import _json_dict

    rows = conn.execute(
        "SELECT task_id, payload FROM task_events WHERE kind = 'created' AND instr(payload, ?) > 0",
        (task_id,),
    ).fetchall()
    return {
        r["task_id"] for r in rows
        if r["task_id"] != task_id and _json_dict(r["payload"]).get("creator_task_id") == task_id
    }


def open_created_children(
    conn: sqlite3.Connection, task_id: str, created_cards: Optional[Iterable[str]] = None,
) -> list[str]:
    """Open (not done/archived) cards ``task_id`` created — by its ``created`` events, plus any
    existing id in the worker's ``created_cards`` claim — sorted. Read-only."""
    ids = _created_by(conn, task_id) | {c for c in (created_cards or ()) if c and c != task_id}
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id FROM tasks WHERE id IN ({placeholders}) "
        "AND status NOT IN ('done', 'archived')", tuple(ids),
    ).fetchall()
    return sorted(r["id"] for r in rows)


class _DecompositionAbort(Exception):
    """Internal: roll the inversion back and leave the task untouched."""


def convert_decomposition_to_dependency_wait(
    conn: sqlite3.Connection, task_id: str, *, created_cards: Optional[Iterable[str]] = None,
    reason: Optional[str] = None, metadata: Optional[dict] = None,
    expected_run_id: Optional[int] = None, force: bool = False,
) -> list[str]:
    """Turn a decomposition-only handoff of ``task_id`` into a dependency wait on the open cards
    it created. Returns the ids it now waits on; ``[]`` means nothing changed.

    An orchestrator that only split its card usually created the children with itself as their
    *parent*; refusing the handoff would then deadlock (child waits on T, T waits on child). So,
    in ONE transaction: drop ``T -> child`` for every child T created, link ``child -> T`` for
    every open created/claimed child, and park T in ``todo`` with ``block_kind='dependency'`` —
    the same landing as ``block_task(kind='dependency')`` — so ``recompute_ready`` re-runs T
    once they are all done. Claimed-but-not-created ids are waited on but never unlinked, so a
    claim cannot detach a pre-existing review/QA child. Refused (``[]``, untouched) when a link
    would cycle, the task is not ``running``/``ready``, or a running task's run is not the
    caller's (``expected_run_id``) and ``force`` is not set.
    """
    from hermes_cli.kanban_db import (
        _append_event, _end_or_synthesize_run, _fire_task_hook, _link, _retry_status_for_run,
        _would_cycle, get_task, recompute_ready, write_txn,
    )

    # Read-only pre-check: a plain rejection (no open created cards) never takes the write lock.
    if not open_created_children(conn, task_id, created_cards):
        return []
    try:
        with write_txn(conn):
            row = conn.execute(
                "SELECT status, current_run_id FROM tasks WHERE id = ?", (task_id,),
            ).fetchone()
            if row is None or row["status"] not in ("running", "ready"):
                return []
            if expected_run_id is not None:
                if row["current_run_id"] != int(expected_run_id):
                    return []
            elif row["status"] == "running" and not force:
                return []
            # Computed inside the write txn: a child finishing concurrently is either seen as
            # done here (and ignored) or finishes after commit (and promotes T normally).
            waits = open_created_children(conn, task_id, created_cards)
            if not waits:
                return []
            created = _created_by(conn, task_id)
            for child in waits:
                if child in created and conn.execute(
                    "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?", (task_id, child),
                ).rowcount:
                    _append_event(conn, child, "unlinked", {
                        "parent": task_id, "child": child, "reason": "decomposition_inverted"})
            for child in waits:
                if _would_cycle(conn, child, task_id):
                    raise _DecompositionAbort(child)
                _link(conn, child, task_id)
                _append_event(conn, task_id, "linked", {
                    "parent": child, "child": task_id, "reason": "decomposition_inverted"})
            source_status = (_retry_status_for_run(conn, task_id)
                             if row["status"] == "running" else "ready")
            sql = ("UPDATE tasks SET status = 'todo', claim_lock = NULL, claim_expires = NULL, "
                   "worker_pid = NULL, block_kind = 'dependency' "
                   "WHERE id = ? AND status IN ('running', 'ready')")
            params: tuple = (task_id,)
            if expected_run_id is not None:
                sql += " AND current_run_id = ?"
                params = (task_id, int(expected_run_id))
            if conn.execute(sql, params).rowcount != 1:
                raise _DecompositionAbort(task_id)
            run_id = _end_or_synthesize_run(
                conn, task_id, outcome="blocked", status="blocked", summary=reason,
                metadata=metadata, synthesize=bool(reason))
            _append_event(conn, task_id, "dependency_wait", {
                "reason": reason, "kind": "dependency", "source_status": source_status,
                "decomposition": True, "waiting_on": waits,
            }, run_id=run_id)
    except _DecompositionAbort as abort:
        _log.info("decomposition wait for %s refused (cycle or lost run via %s); left untouched",
                  task_id, abort)
        return []
    # Former children that only waited on T can start now.
    recompute_ready(conn)
    _fire_task_hook("kanban_task_blocked", get_task(conn, task_id), task_id, run_id, reason=reason)
    return waits

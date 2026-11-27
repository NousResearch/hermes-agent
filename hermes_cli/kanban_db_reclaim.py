"""Reclaim bookkeeping: closing a run as ``reclaimed`` and the liveness evidence a
closed run leaves behind for the next claim.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` / ``_kbd`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import sqlite3
from typing import Optional


def _prior_run_worker_still_alive(conn: sqlite3.Connection, task_id: str) -> bool:
    """True when the most recent ENDED run still has a live host-local worker.

    A status flip (dashboard drag/bulk patch) releases the claim and closes the run
    in one txn, then kills the worker AFTER the commit — a dispatcher tick in that
    window re-claims the card and spawns a duplicate beside the still-running worker
    (#140342). The closed run row keeps ``worker_pid``/``worker_started_at``/
    ``claim_lock`` (the same evidence ``reap_terminal_workers`` keys on), so the
    claim path can hold off until the flip's termination — or the reaper — lands.
    Non-host-local and fingerprint-less legacy rows are never held: their liveness
    is not observable here and the TTL path owns them."""
    row = conn.execute(
        "SELECT worker_pid, worker_started_at, claim_lock FROM task_runs WHERE task_id = ? AND ended_at IS NOT NULL ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None or row["worker_pid"] is None:
        return False
    if not (row["claim_lock"] or "").startswith(_kb._host_prefix()):
        return False
    return _kbd._worker_alive(row["worker_pid"], row["worker_started_at"])


def _record_reclaim(
    conn: sqlite3.Connection, task_id: str, termination: dict, *, error: str, payload: dict,
) -> Optional[int]:
    """Close the active run as ``reclaimed`` and emit the ``reclaimed`` event
    (payload merged with the termination report). Caller holds the txn."""
    run_id = _kb._end_run(
        conn, task_id, outcome="reclaimed", status="reclaimed", error=error, metadata=termination,
    )
    payload.update(termination)
    _kb._append_event(conn, task_id, "reclaimed", payload, run_id=run_id)
    return run_id


# Late-bound origin namespaces (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb
from hermes_cli import kanban_db_dispatch as _kbd

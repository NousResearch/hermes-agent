"""Fail-closed process/run reconciliation for host-wide lane reservations."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Mapping


def process_identity(pid: int, fingerprint: str) -> str:
    if (type(pid) is not int or pid <= 0 or not isinstance(fingerprint, str)
            or not fingerprint or fingerprint == "unverified"):
        raise ValueError("verified process identity required")
    return json.dumps({"pid": pid, "fingerprint": fingerprint}, sort_keys=True)


def identity_alive(identity: str | None) -> bool | None:
    """Unreadable fingerprints are unknown, not evidence of process death."""
    from hermes_cli.kanban_db_dispatch import _pid_alive, _process_fingerprint

    try:
        value = json.loads(identity or "null")
        pid, fingerprint = value["pid"], value["fingerprint"]
        process_identity(pid, fingerprint)
    except (ValueError, TypeError, KeyError):
        return None
    if not _pid_alive(pid):
        return False
    current = _process_fingerprint(pid)
    return None if current is None else current == fingerprint


def reservation_alive(row: Mapping) -> bool | None:
    """Recover a pending reservation using its persisted board/run witness.

    A dead dispatcher cannot prove that a spawned but not-yet-bound worker is
    dead. Without a worker witness pending reservations stay held until the
    spawn protocol explicitly records failure. Terminal task state alone never
    releases capacity while the retained physical worker is alive.
    """
    if row.get("worker"):
        return identity_alive(row["worker"])
    path = Path(row["board"])
    if not path.is_absolute():
        return None
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5) as conn:
            conn.row_factory = sqlite3.Row
            run = conn.execute(
                "SELECT task_id, worker_pid, worker_started_at FROM task_runs WHERE id = ?",
                (row["run"],),
            ).fetchone()
    except sqlite3.Error:
        return None
    if run is None or run["task_id"] != row["task"] or not run["worker_pid"]:
        return None
    try:
        identity = process_identity(run["worker_pid"], run["worker_started_at"])
    except ValueError:
        return None
    return identity_alive(identity)

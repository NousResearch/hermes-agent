"""Dispatcher maintenance before spawn caps: wake, reclaim, then promote."""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.kanban_db_dispatch import DispatchResult


def run_reclaim_phase(
    conn: sqlite3.Connection,
    result: DispatchResult,
    *,
    stale_timeout_seconds: int,
    failure_limit: int,
    reconcile_orphans: bool,
    board: str | None = None,
    dry_run: bool = False,
) -> None:
    """Wake due follow-ups, reclaim running tasks, then promote dependents."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli.kanban_db_schedule import wake_scheduled

    wake_scheduled(conn, dry_run=dry_run)
    kbd.reap_worker_zombies()
    result.reaped_terminal_workers = kbd.reap_terminal_workers(conn)
    result.reclaimed = kb.release_stale_claims(conn, failure_limit=failure_limit)
    if reconcile_orphans:
        result.reconciled_orphans = kbd.reconcile_orphaned_running(conn)
    result.stale = kbd.detect_stale_running(conn, stale_timeout_seconds=stale_timeout_seconds)
    result.crashed = kbd.detect_crashed_workers(conn, board=board)
    # Side-channel attributes (see detect_crashed_workers); rate-limited tasks
    # went back to ``ready`` and the respawn guard defers them until quota clears.
    result.auto_blocked.extend(getattr(kbd.detect_crashed_workers, "_last_auto_blocked", []))
    result.rate_limited.extend(getattr(kbd.detect_crashed_workers, "_last_rate_limited", []))
    result.timed_out = kbd.enforce_max_runtime(conn)
    result.promoted = kb.recompute_ready(conn, failure_limit=failure_limit)

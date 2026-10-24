"""The dispatcher's spawn budget and lane config, read from ``config.yaml``.

Every reader here answers "what did the operator configure" and fails open to the historic default,
so a corrupt or unreadable config degrades a board's throughput instead of stopping its dispatcher.
Split out of ``hermes_cli.kanban_db_dispatch`` (a 2,000-line facade may not grow).
"""

from __future__ import annotations

import sqlite3
from typing import Optional


def review_dispatch_enabled() -> bool:
    """Whether review tasks dispatch automatically. Default true (Hermes ships
    ``sdlc-review``); operators disable it for human-only review boards.
    """
    try:
        from hermes_cli.config import load_config
        return bool((load_config() or {}).get("kanban", {}).get("review_dispatch", True))
    except Exception:  # health: allow BLE001 -- a config read that raises must not stop the dispatcher; true is the shipped behaviour
        return True


def configured_review_reserved_slots() -> int:
    """Read ``kanban.review_reserved_slots`` from config, defaulting to 1.

    How many shared-budget slots the ready loop must leave for the review lane
    when spawnable review work exists. 1 preserves the historic single-slot
    hold; 0 disables the reservation (reviews then compete for leftover
    budget only). Never negative.
    """
    try:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get(
            "review_reserved_slots"
        )
    except Exception:  # health: allow BLE001 -- a config read that raises must not stop the dispatcher; 1 is the historic hold
        return 1
    if raw is None:
        return 1
    try:
        ival = int(raw)
    except (TypeError, ValueError):
        return 1
    return ival if ival >= 0 else 1


def configured_max_in_progress() -> Optional[int]:
    """Read ``kanban.max_in_progress`` from config, or None when unset/invalid.

    Shared so every dispatch entry point agrees on "explicitly configured": a
    positive integer wins, anything else falls through to the derived default.
    """
    try:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get("max_in_progress")
    except Exception:  # health: allow BLE001 -- a config read that raises must not stop the dispatcher; None derives the default
        return None
    if raw is None:
        return None
    try:
        ival = int(raw)
    except (TypeError, ValueError):
        return None
    return ival if ival >= 1 else None


def count_running_tasks(conn: sqlite3.Connection) -> int:
    """Number of tasks in ``status='running'``.

    Used by the multi-board sweep to count OTHER boards' workers against the
    host-level budget: the memory-derived cap bounds the machine, not the
    board. Fails open to 0 so a broken board doesn't brick dispatch on healthy ones.
    """
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
            ).fetchone()[0]
        )
    except Exception:  # health: allow BLE001 -- a board whose query fails counts as zero so the healthy boards keep dispatching
        return 0

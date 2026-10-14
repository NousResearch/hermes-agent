"""Read the durable exit trailer from a Kanban worker's log."""

from __future__ import annotations

import re
from typing import Optional

from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


_EXIT_TRAILER_RE = re.compile(
    r"^" + re.escape(KANBAN_WORKER_EXIT_TRAILER) + r"(\d+)(?: reset_at=(\d+))?\s*$", re.MULTILINE,
)


def worker_log_exit(task_id: str, board: Optional[str] = None) -> tuple[Optional[int], Optional[int]]:
    """Return the last logged ``(exit code, reset_at)``, or ``(None, None)``.

    The worker writes this trailer even when a different process reaps it.
    Logs are append-only across re-runs, so the last trailer wins.
    """
    from hermes_cli import kanban_db as kb

    try:
        raw = kb.read_worker_log(task_id, tail_bytes=4000, board=board)
    except OSError:
        return None, None
    matches = _EXIT_TRAILER_RE.findall(raw or "")
    if not matches:
        return None, None
    code, reset_at = matches[-1]
    return int(code), int(reset_at) if reset_at else None

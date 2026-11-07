"""Live background work per session id, stamped on the session-list rows the sidebar reads.

The desktop can only see a session's background processes and async delegations for sessions it
has open (it polls them per session). A job started from another window or the phone, or before a
restart of the app, read as Idle in the sidebar while it was still running. The list endpoints run
in the same process as the process registry and the delegation registry, so they stamp a count
here and every surface agrees.
"""

from __future__ import annotations

import contextlib
import logging
from collections import Counter
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# Row keys that can name the session a job was started under: a compressed conversation's
# sidebar row is its tip, while the job keeps the id it was spawned under.
_ROW_ID_KEYS = ("id", "resolved_id", "_lineage_root_id")


def background_work_counts() -> dict[str, int]:
    """Running background processes + live async delegations, keyed by owning session id."""
    counts: Counter = Counter()
    with contextlib.suppress(Exception):
        from tools.process_registry import process_registry
        counts.update(p["owner_task_id"] for p in process_registry.list_sessions()
                      if p["status"] == "running" and p.get("owner_task_id"))
    with contextlib.suppress(Exception):
        from tools.async_delegation import live_counts_by_session
        counts.update(live_counts_by_session())
    return dict(counts)


def stamp_background_work(rows: Iterable[dict[str, Any]], counts: Optional[dict[str, int]] = None) -> None:
    """Set ``background_work`` (> 0) on rows with live work; absent means none."""
    counts = background_work_counts() if counts is None else counts
    if not counts:
        return
    for row in rows:
        ids = {str(row.get(key) or "") for key in _ROW_ID_KEYS} - {""}
        if total := sum(counts.get(i, 0) for i in ids):
            row["background_work"] = total

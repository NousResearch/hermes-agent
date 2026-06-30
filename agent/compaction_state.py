"""Best-effort durable metadata packets for compaction boundaries."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

_DURABLE_BUCKETS = (
    "project_facts",
    "decisions",
    "open_threads",
    "procedures",
    "artifacts",
    "do_not_carry",
)


def _empty_candidates() -> Dict[str, list]:
    return {bucket: [] for bucket in _DURABLE_BUCKETS}


def _safe_get_session(session_db: Any, session_id: Optional[str]) -> Dict[str, Any]:
    if not session_db or not session_id:
        return {}
    try:
        row = session_db.get_session(session_id)
    except Exception:
        return {}
    return dict(row or {})


def _lineage_root_id(session_db: Any, session_id: str, old_session_id: str = "") -> str:
    """Return the best-known logical root id for a compaction boundary."""
    if not session_db:
        return old_session_id or session_id

    current_id = old_session_id or session_id
    seen: set[str] = set()
    while current_id and current_id not in seen:
        seen.add(current_id)
        row = _safe_get_session(session_db, current_id)
        parent_id = row.get("parent_session_id")
        if not parent_id:
            return current_id
        current_id = parent_id
    return old_session_id or session_id


def build_compaction_boundary_packet(
    *,
    session_db: Any,
    session_id: str,
    in_place: bool,
    compression_count: int,
    old_session_id: str = "",
    platform: str = "",
    boundary_at: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a small metadata packet describing a compaction boundary.

    The packet is deliberately metadata-first and best-effort: it does not
    include raw transcripts or tool output, and it must never make compression
    fail if session metadata cannot be read.
    """

    session_id = session_id or ""
    old_session_id = old_session_id or ""
    root_id = _lineage_root_id(session_db, session_id, old_session_id)
    current = _safe_get_session(session_db, session_id)
    old = _safe_get_session(session_db, old_session_id)
    root = _safe_get_session(session_db, root_id)

    def _first_non_empty(key: str) -> str:
        for row in (current, old, root):
            value = row.get(key)
            if value:
                return value
        return ""

    return {
        "session_id": session_id,
        "old_session_id": old_session_id,
        "lineage_root_id": root_id,
        "in_place": bool(in_place),
        "compression_count": int(compression_count or 0),
        "platform": platform or "",
        "boundary_at": float(boundary_at if boundary_at is not None else time.time()),
        "cwd": _first_non_empty("cwd"),
        "git_repo_root": _first_non_empty("git_repo_root"),
        "git_branch": _first_non_empty("git_branch"),
        "title": _first_non_empty("title"),
        "durable_candidates": _empty_candidates(),
    }

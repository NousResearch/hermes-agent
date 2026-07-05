"""Score adapters for SkillOpt validation gates."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home


def _db_path() -> Path:
    return get_hermes_home() / "verification_evidence.db"


def score_verification_evidence(
    *,
    root: str | Path,
    session_id: str | None = None,
    min_events: int = 2,
) -> dict[str, Any]:
    """Return a pass-rate score from the verification evidence ledger.

    ``heldout_ready`` is false until at least ``min_events`` are available. This
    lets SkillOpt gates fail closed instead of promoting candidates from a single
    ad-hoc success.
    """

    root_s = str(Path(root).resolve())
    sid = str(session_id) if session_id else None
    path = _db_path()
    if not path.exists():
        return {"score": 0.0, "total": 0, "passed": 0, "failed": 0, "heldout_ready": False, "events": []}

    where = ["root = ?"]
    params: list[Any] = [root_s]
    if sid is not None:
        where.append("session_id = ?")
        params.append(sid)
    sql = (
        "SELECT id, created_at, session_id, canonical_command, kind, scope, status "
        "FROM verification_events WHERE " + " AND ".join(where) + " ORDER BY id ASC"
    )
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        except sqlite3.Error:
            rows = []

    total = len(rows)
    passed = sum(1 for row in rows if row.get("status") == "passed")
    failed = sum(1 for row in rows if row.get("status") == "failed")
    score = (passed / total) if total else 0.0
    return {
        "score": score,
        "total": total,
        "passed": passed,
        "failed": failed,
        "heldout_ready": total >= max(1, int(min_events)),
        "events": rows,
    }

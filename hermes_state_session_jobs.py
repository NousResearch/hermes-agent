"""``session_jobs`` / ``session_job_runs``: agent-scheduled work on a session's logical clock.

A turn/step job is session-bound: it lives beside the origin session, never in ``cron/jobs.json``
(wall-clock invariants — timezones, catch-up windows, quota holds — do not apply to a logical
schedule). Firing is a compare-and-set on ``next_due_seq`` so exactly one owner advances it.

Design: website/docs/developer-guide/scheduled-tasks.md § Agent-scheduled turn/step jobs.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional

_JOB_FIELDS = frozenset((
    "name", "prompt", "skills", "deliver", "failure_deliver", "every", "remaining",
    "next_due_seq", "enabled", "paused_reason", "last_status", "last_fired_at", "updated_at",
))
_MAX_CHAIN_DEPTH = 3


def _encode(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _decode_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    for key in ("skills", "deliver", "failure_deliver"):
        raw = data.get(key)
        if isinstance(raw, str) and raw:
            try:
                data[key] = json.loads(raw)
            except ValueError:
                pass
    return data


class SessionJobsMixin:
    def create_session_job(
        self, *, origin_session_id: str, kind: str, every: int, prompt: str, next_due_seq: int,
        name: Optional[str] = None, skills: Any = None, deliver: Any = None,
        failure_deliver: Any = None, remaining: Optional[int] = None, profile_name: str = "",
        origin_depth: int = 0,
    ) -> str:
        if kind not in ("turn", "step"):
            raise ValueError("kind must be turn or step")
        if int(every) < 1:
            raise ValueError("every must be >= 1")
        if int(origin_depth) >= _MAX_CHAIN_DEPTH:
            raise ValueError("scheduled job chain depth exceeded")
        job_id = "tj_" + uuid.uuid4().hex[:8]
        now = time.time()

        def _do(conn):
            conn.execute(
                "INSERT INTO session_jobs (id, origin_session_id, profile_name, name, kind, every,"
                " remaining, next_due_seq, prompt, skills, deliver, failure_deliver, enabled,"
                " origin_depth, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
                (job_id, origin_session_id, profile_name or "", name, kind, int(every), remaining,
                 int(next_due_seq), prompt, _encode(skills), _encode(deliver),
                 _encode(failure_deliver), int(origin_depth), now, now),
            )

        self._execute_write(_do)
        return job_id

    def get_session_job(self, job_id: str) -> Optional[dict[str, Any]]:
        row = self._read_one("SELECT * FROM session_jobs WHERE id = ?", (job_id,))
        return _decode_row(row) if row is not None else None

    def list_session_jobs(
        self, origin_session_id: Optional[str] = None, *, enabled_only: bool = False,
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        if origin_session_id:
            clauses.append("origin_session_id = ?")
            params.append(origin_session_id)
        if enabled_only:
            clauses.append("enabled = 1")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._read_all(
            f"SELECT * FROM session_jobs{where} ORDER BY created_at ASC, id ASC", params)
        return [_decode_row(row) for row in rows]

    def update_session_job(self, job_id: str, **fields: Any) -> bool:
        updates = {key: value for key, value in fields.items() if key in _JOB_FIELDS}
        if not updates:
            return False
        updates.setdefault("updated_at", time.time())
        assignments = ", ".join(f"{key} = ?" for key in updates)
        params = [*(_encode(v) if key in ("skills", "deliver", "failure_deliver") else v
                    for key, v in updates.items()), job_id]

        def _do(conn):
            return conn.execute(
                f"UPDATE session_jobs SET {assignments} WHERE id = ?", params).rowcount

        return bool(self._execute_write(_do))

    def remove_session_job(self, job_id: str) -> bool:
        def _do(conn):
            return conn.execute("DELETE FROM session_jobs WHERE id = ?", (job_id,)).rowcount

        return bool(self._execute_write(_do))

    def claim_session_job(self, job_id: str, *, expected_due: int, patience_s: float = 0.5) -> bool:
        """Compare-and-set: advance ``next_due_seq`` only if this owner saw ``expected_due``."""
        now = time.time()

        def _do(conn):
            return conn.execute(
                "UPDATE session_jobs SET next_due_seq = next_due_seq + every,"
                " remaining = CASE WHEN remaining IS NULL THEN NULL ELSE remaining - 1 END,"
                " enabled = CASE WHEN remaining IS NOT NULL AND remaining <= 1 THEN 0 ELSE enabled END,"
                " last_fired_at = ?, updated_at = ?"
                " WHERE id = ? AND next_due_seq = ? AND enabled = 1",
                (now, now, job_id, int(expected_due)),
            ).rowcount

        return bool(self._execute_write(_do, patience_s=patience_s))

    def record_session_job_run(
        self, job_id: str, *, fired_at_seq: int, status: str, origin_session_id: Optional[str] = None,
        run_session_id: Optional[str] = None, parent_job_id: Optional[str] = None,
        started_at: Optional[float] = None, ended_at: Optional[float] = None,
        error: Optional[str] = None,
    ) -> int:
        start = time.time() if started_at is None else float(started_at)
        end = time.time() if ended_at is None else float(ended_at)

        def _do(conn):
            cursor = conn.execute(
                "INSERT INTO session_job_runs (job_id, origin_session_id, fired_at_seq,"
                " run_session_id, parent_job_id, started_at, ended_at, status, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (job_id, origin_session_id, int(fired_at_seq), run_session_id, parent_job_id,
                 start, end, status, error),
            )
            return int(cursor.lastrowid or 0)

        return int(self._execute_write(_do) or 0)

    def finish_session_job_run(
        self, run_id: int, *, status: str, error: Optional[str] = None,
        run_session_id: Optional[str] = None,
    ) -> bool:
        def _do(conn):
            return conn.execute(
                "UPDATE session_job_runs SET ended_at = ?, status = ?, error = ?,"
                " run_session_id = COALESCE(?, run_session_id) WHERE id = ?",
                (time.time(), status, error, run_session_id, int(run_id)),
            ).rowcount

        return bool(self._execute_write(_do))

    def session_job_runs(self, job_id: Optional[str] = None, *, limit: int = 50) -> list[dict[str, Any]]:
        if job_id:
            rows = self._read_all(
                "SELECT * FROM session_job_runs WHERE job_id = ? ORDER BY id DESC LIMIT ?",
                (job_id, max(0, int(limit))),
            )
        else:
            rows = self._read_all(
                "SELECT * FROM session_job_runs ORDER BY id DESC LIMIT ?", (max(0, int(limit)),))
        return [dict(row) for row in rows]

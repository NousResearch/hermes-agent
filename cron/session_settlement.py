"""Settle cron session rows abandoned after their scheduler owner exits (#138304)."""

from __future__ import annotations

import re

_QUIET_CRON_SESSION_SECONDS = 300.0
_CRON_SESSION_ID = re.compile(r"^cron_(?P<job_id>.+)_\d{8}_\d{6}$")


def _job_id_from_session_id(session_id: object) -> str | None:
    match = _CRON_SESSION_ID.match(str(session_id))
    return match.group("job_id") if match else None


def close_interrupted_run_sessions(session_db=None) -> list[str]:
    """Close quiet cron rows only after the ledger proves their latest run is unknown.

    The execution ledger is authoritative for scheduler ownership.  A final liveness
    check closes the read-to-write race where a new worker claims the same job after
    the ledger snapshot but before its old session row is settled.
    """
    owns_db = session_db is None
    if owns_db:
        from hermes_state_registry import acquire

        session_db = acquire()
    try:
        rows = session_db.list_quiet_open_cron_sessions(
            max_idle_seconds=_QUIET_CRON_SESSION_SECONDS
        )
        candidates = {
            str(row["id"]): _job_id_from_session_id(row["id"])
            for row in rows
        }
        job_ids = [job_id for job_id in candidates.values() if job_id]
        if not job_ids:
            return []

        from cron.executions import latest_executions, live_inflight_execution

        latest = latest_executions(job_ids)
        settled = []
        for session_id, job_id in candidates.items():
            if not job_id or latest.get(job_id, {}).get("status") != "unknown":
                continue
            if live_inflight_execution(job_id) is not None:
                continue
            session_db.end_session(session_id, "cron_incomplete_no_output")
            settled.append(session_id)
        return settled
    finally:
        if owns_db:
            from hermes_state_registry import release_or_close

            release_or_close(session_db)

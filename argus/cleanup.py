"""
ARGUS dead session cleanup — find and mark orphaned sessions.

Scans hermes state.db for sessions that appear to be dead:
- Started > 2h ago, no messages in 30min
- Delegate sessions with no parent
- Cron sessions referencing deleted jobs
- Sessions with ended_at but metadata suggests they're still "active" in argus

Results are recorded in argus.db for audit. Sessions are marked 'orphaned'
in argus.db (not deleted — audit trail preserved).
"""

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("argus.cleanup")


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except ImportError:
        return Path.home() / ".hermes"


def _state_db_path() -> Path:
    try:
        from hermes_state import DEFAULT_DB_PATH

        return Path(DEFAULT_DB_PATH)
    except ImportError:
        return _hermes_home() / "state.db"


def _connect_state_db() -> Optional[sqlite3.Connection]:
    path = _state_db_path()
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error("Failed to connect to state.db: %s", e)
        return None


def _load_cron_job_ids() -> set:
    """Load all cron job IDs from jobs.json."""
    jobs_file = _hermes_home() / "cron" / "jobs.json"
    if not jobs_file.exists():
        return set()
    try:
        with open(jobs_file) as f:
            jobs = json.load(f)
        return {j["id"] for j in jobs if isinstance(j, dict) and "id" in j}
    except Exception:
        return set()


def find_stale_sessions(
    state_conn: sqlite3.Connection,
    max_age_hours: float = 2.0,
    stale_minutes: float = 30.0,
) -> List[Dict[str, Any]]:
    """Find sessions that started long ago and have no recent messages.

    Returns list of session dicts with id, source, started_at, last_message_at,
    message_count, and staleness reason.
    """
    cutoff_stale = time.time() - (stale_minutes * 60)
    cutoff_old = time.time() - (max_age_hours * 3600)

    try:
        rows = state_conn.execute(
            """
            SELECT
                s.id,
                s.source,
                s.started_at,
                s.ended_at,
                s.message_count,
                s.parent_session_id,
                s.title,
                (SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = s.id) as last_message_at
            FROM sessions s
            WHERE s.ended_at IS NULL
              AND s.started_at < ?
        """,
            (cutoff_old,),
        ).fetchall()
    except Exception as e:
        logger.error("Error querying stale sessions: %s", e)
        return []

    stale = []
    for row in rows:
        last_msg = row["last_message_at"]
        # Session is stale if it has no recent messages
        if last_msg is None or last_msg < cutoff_stale:
            age_hours = (time.time() - row["started_at"]) / 3600
            idle_minutes = (time.time() - (last_msg or row["started_at"])) / 60
            stale.append(
                {
                    "session_id": row["id"],
                    "source": row["source"],
                    "started_at": row["started_at"],
                    "last_message_at": last_msg,
                    "message_count": row["message_count"] or 0,
                    "parent_session_id": row["parent_session_id"],
                    "title": row["title"],
                    "age_hours": round(age_hours, 1),
                    "idle_minutes": round(idle_minutes, 1),
                    "reason": "stale" if last_msg else "never_active",
                }
            )

    return stale


def find_orphaned_delegates(
    state_conn: sqlite3.Connection,
) -> List[Dict[str, Any]]:
    """Find delegate sessions whose parent no longer exists or has ended."""
    try:
        delegates = state_conn.execute("""
            SELECT s.id, s.source, s.parent_session_id, s.started_at, s.title
            FROM sessions s
            WHERE s.source IN ('delegate', 'delegate_task')
              AND s.parent_session_id IS NOT NULL
              AND s.ended_at IS NULL
        """).fetchall()
    except Exception:
        return []

    # Get all active parent session IDs
    try:
        parent_ids = state_conn.execute("""
            SELECT id FROM sessions WHERE ended_at IS NULL
        """).fetchall()
        active_parents = {r["id"] for r in parent_ids}
    except Exception:
        return []

    orphaned = []
    for d in delegates:
        if d["parent_session_id"] not in active_parents:
            orphaned.append(
                {
                    "session_id": d["id"],
                    "source": d["source"],
                    "parent_session_id": d["parent_session_id"],
                    "started_at": d["started_at"],
                    "title": d["title"],
                    "reason": "parent_ended",
                }
            )

    return orphaned


def find_zombie_cron_sessions(
    state_conn: sqlite3.Connection,
    argus_cursor: sqlite3.Cursor,
) -> List[Dict[str, Any]]:
    """Find cron sessions in argus.db referencing jobs that no longer exist."""
    active_job_ids = _load_cron_job_ids()

    try:
        argus_cursor.execute("""
            SELECT session_id, job_id, task_description, started_at, status
            FROM sessions
            WHERE session_type = 'cron' AND status = 'active'
        """)
        argus_cron_sessions = [dict(r) for r in argus_cursor.fetchall()]
    except Exception:
        return []

    zombies = []
    for s in argus_cron_sessions:
        job_id = s.get("job_id")
        if job_id and job_id not in active_job_ids:
            zombies.append(
                {
                    "session_id": s["session_id"],
                    "job_id": job_id,
                    "task_description": s.get("task_description"),
                    "started_at": s.get("started_at"),
                    "reason": "job_deleted",
                }
            )

    return zombies


def find_ended_but_registered(
    state_conn: sqlite3.Connection,
    argus_cursor: sqlite3.Cursor,
) -> List[Dict[str, Any]]:
    """Find sessions marked active in argus but already ended in state.db."""
    try:
        argus_cursor.execute("""
            SELECT session_id FROM sessions WHERE status = 'active'
        """)
        active_argus_ids = {r["session_id"] for r in argus_cursor.fetchall()}
    except Exception:
        return []

    # Strip type prefixes to get real session IDs
    real_ids = set()
    for sid in active_argus_ids:
        parts = sid.split("_", 1)
        real_ids.add(parts[1] if len(parts) == 2 else sid)

    if not real_ids:
        return []

    try:
        placeholders = ",".join("?" for _ in real_ids)
        ended = state_conn.execute(
            f"""  # nosec
            SELECT id, source, ended_at, end_reason, title
            FROM sessions
            WHERE id IN ({placeholders}) AND ended_at IS NOT NULL
        """,  # nosec B608
            list(real_ids),
        ).fetchall()
    except Exception:
        return []

    results = []
    for row in ended:
        # Find the argus session_id (with prefix)
        for sid in active_argus_ids:
            real = sid.split("_", 1)[1] if "_" in sid else sid
            if real == row["id"]:
                results.append(
                    {
                        "session_id": sid,
                        "real_session_id": row["id"],
                        "source": row["source"],
                        "ended_at": row["ended_at"],
                        "end_reason": row["end_reason"],
                        "title": row["title"],
                        "reason": "ended_in_state_db",
                    }
                )
                break

    return results


def mark_as_orphaned(
    argus_cursor: sqlite3.Cursor,
    argus_conn: sqlite3.Connection,
    session_id: str,
    reason: str,
) -> None:
    """Mark a session as orphaned in argus.db."""
    try:
        argus_cursor.execute(
            """
            UPDATE sessions
            SET status = 'orphaned', metadata = json_set(
                COALESCE(metadata, '{}'),
                '$.orphan_reason', ?,
                '$.orphaned_at', datetime('now')
            )
            WHERE session_id = ? AND status = 'active'
        """,
            (reason, session_id),
        )
        argus_conn.commit()
        logger.info("Marked session %s as orphaned: %s", session_id, reason)
    except Exception as e:
        logger.error("Failed to mark %s as orphaned: %s", session_id, e)


def run_cleanup(
    argus_cursor: sqlite3.Cursor,
    argus_conn: sqlite3.Connection,
) -> Dict[str, List[Dict]]:
    """Run all dead session checks and mark orphans.

    Returns dict with keys: stale, orphaned_delegates, zombie_cron, ended_registered.
    """
    results = {
        "stale": [],
        "orphaned_delegates": [],
        "zombie_cron": [],
        "ended_registered": [],
    }

    state_conn = _connect_state_db()
    if not state_conn:
        logger.warning("Cannot connect to state.db — skipping cleanup")
        return results

    try:
        # 1. Stale sessions
        results["stale"] = find_stale_sessions(state_conn)

        # 2. Orphaned delegates
        results["orphaned_delegates"] = find_orphaned_delegates(state_conn)

        # 3. Zombie cron sessions (job deleted)
        results["zombie_cron"] = find_zombie_cron_sessions(state_conn, argus_cursor)

        # 4. Sessions ended in state.db but still active in argus
        results["ended_registered"] = find_ended_but_registered(
            state_conn, argus_cursor
        )

        # Mark all findings as orphaned
        all_findings = []
        for category, items in results.items():
            for item in items:
                sid = item["session_id"]
                reason = "%s: %s" % (category, item.get("reason", "unknown"))
                all_findings.append((sid, reason))

        for sid, reason in all_findings:
            mark_as_orphaned(argus_cursor, argus_conn, sid, reason)

        if all_findings:
            logger.info("Cleanup: found %d orphaned sessions", len(all_findings))

    finally:
        state_conn.close()

    return results

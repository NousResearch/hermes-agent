"""Report-only diagnostic for Discord thread session mapping loss.

This module intentionally accepts explicit paths and has no repair/write mode.
It reads ``sessions.json`` and ``state.db`` metadata only; transcript content is
never queried.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote


_NO_THREAD_KEY_MATCH_LIMITATION = (
    "state.db stores sessions by session_id/source but does not store the "
    "gateway session_key or Discord thread_id, so orphan transcripts cannot "
    "be reliably matched back to this Discord thread key from state.db alone."
)


def inspect_discord_thread_mapping(
    *,
    sessions_json: str | Path,
    state_db: str | Path,
    session_key: str,
    candidate_limit: int = 10,
) -> dict[str, Any]:
    """Return a metadata-only report for a Discord thread session key.

    The report is read-only. It never creates, migrates, repairs, prunes, or
    rewrites either input path, and it never reads message content.
    """
    sessions_path = Path(sessions_json)
    state_db_path = Path(state_db)
    sessions_report, sessions_index, sessions_errors = _read_sessions_index(
        sessions_path
    )
    mapped_entry = sessions_index.get(session_key)
    active_session_id = _entry_session_id(mapped_entry)

    state_report, active_session, exact_orphans, candidates, state_errors = _inspect_state_db(
        state_db_path,
        session_key=session_key,
        active_session_id=active_session_id,
        mapped_session_ids=_mapped_session_ids(sessions_index),
        candidate_limit=candidate_limit,
    )

    mapping_exists = active_session_id is not None
    return {
        "session_key": session_key,
        "sessions_json": sessions_report,
        "state_db": state_report,
        "mapping": {
            "exists": mapping_exists,
            "status": "mapped" if mapping_exists else "missing",
            "active_session_id": active_session_id,
        },
        "active_session": active_session,
        "exact_orphan_sessions": exact_orphans,
        "candidate_orphan_sessions": candidates,
        "errors": sessions_errors + state_errors,
    }


def _read_sessions_index(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    report = {
        "path": str(path),
        "exists": path.exists(),
        "loaded": False,
        "entry_count": 0,
    }
    if not path.exists():
        return report, {}, []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return report, {}, [f"sessions_json_unreadable: {exc}"]
    if not isinstance(data, dict):
        return report, {}, ["sessions_json_invalid: expected object at top level"]
    report["loaded"] = True
    report["entry_count"] = len(data)
    return report, data, []


def _inspect_state_db(
    path: Path,
    *,
    session_key: str,
    active_session_id: str | None,
    mapped_session_ids: set[str],
    candidate_limit: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    report = {
        "path": str(path),
        "exists": path.exists(),
        "available": False,
        "can_reliably_match_orphans_to_thread_key": False,
        "limitations": [_NO_THREAD_KEY_MATCH_LIMITATION],
    }
    if not path.exists():
        return report, None, [], [], []

    try:
        conn = _connect_readonly(path)
    except sqlite3.Error as exc:
        return report, None, [], [], [f"state_db_unreadable: {exc}"]

    try:
        if not _table_exists(conn, "sessions"):
            return report, None, [], [], ["state_db_missing_sessions_table"]
        report["available"] = True
        has_routing_metadata = _table_exists(conn, "session_routing_metadata")
        if has_routing_metadata:
            report["can_reliably_match_orphans_to_thread_key"] = True
            report["limitations"] = []
        active = (
            _fetch_session_metadata(conn, active_session_id)
            if active_session_id
            else None
        )
        if active is not None:
            active["match_type"] = "active_sessions_json_mapping"
        exact_orphans = (
            _fetch_exact_orphan_sessions(
                conn,
                session_key=session_key,
                mapped_session_ids=mapped_session_ids,
                limit=candidate_limit,
            )
            if has_routing_metadata
            else []
        )
        exact_ids = {row["session_id"] for row in exact_orphans}
        candidates = _fetch_candidate_orphan_sessions(
            conn,
            mapped_session_ids=mapped_session_ids,
            excluded_session_ids=exact_ids,
            limit=candidate_limit,
        )
        return report, active, exact_orphans, candidates, []
    finally:
        conn.close()


def _connect_readonly(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(str(path), safe='/')}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _fetch_session_metadata(
    conn: sqlite3.Connection,
    session_id: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT id, source, user_id, parent_session_id, started_at, ended_at,
               end_reason, message_count
        FROM sessions
        WHERE id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    metadata = _row_to_session_metadata(row, candidate=False)
    metadata.update(_fetch_transcript_stats(conn, session_id))
    routing = _fetch_routing_metadata(conn, session_id)
    if routing:
        metadata["routing_metadata"] = routing
    return metadata


def _fetch_exact_orphan_sessions(
    conn: sqlite3.Connection,
    *,
    session_key: str,
    mapped_session_ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    params: list[Any] = [session_key]
    match_where = ["rm.session_key = ?"]
    filter_where = ["COALESCE(s.message_count, 0) > 0"]
    thread_id = _thread_id_from_session_key(session_key)
    if thread_id:
        match_where.append("(rm.platform = 'discord' AND rm.thread_id = ?)")
        params.append(thread_id)
    excluded = set(mapped_session_ids)
    if excluded:
        placeholders = ",".join("?" for _ in excluded)
        filter_where.append(f"s.id NOT IN ({placeholders})")
        params.extend(sorted(excluded))
    params.append(max(1, int(limit)))
    rows = conn.execute(
        f"""
        SELECT s.id, s.source, s.user_id, s.parent_session_id, s.started_at,
               s.ended_at, s.end_reason, s.message_count,
               rm.session_key, rm.platform, rm.chat_type, rm.chat_id,
               rm.thread_id, rm.parent_chat_id, rm.guild_id,
               rm.user_id AS routing_user_id, rm.message_id,
               rm.created_at AS routing_created_at,
               rm.updated_at AS routing_updated_at
        FROM sessions s
        JOIN session_routing_metadata rm ON rm.session_id = s.id
        WHERE ({' OR '.join(match_where)}) AND {' AND '.join(filter_where)}
        ORDER BY s.started_at DESC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    results = []
    for row in rows:
        item = _row_to_session_metadata(row, candidate=False)
        item.update(_fetch_transcript_stats(conn, row["id"]))
        item["match_type"] = "exact_metadata_match"
        item["routing_metadata"] = _row_to_routing_metadata(row)
        results.append(item)
    return results


def _fetch_candidate_orphan_sessions(
    conn: sqlite3.Connection,
    *,
    mapped_session_ids: set[str],
    excluded_session_ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    params: list[Any] = ["discord"]
    where = ["source = ?", "COALESCE(message_count, 0) > 0"]
    excluded = set(mapped_session_ids) | set(excluded_session_ids)
    if excluded:
        placeholders = ",".join("?" for _ in excluded)
        where.append(f"id NOT IN ({placeholders})")
        params.extend(sorted(excluded))
    params.append(max(1, int(limit)))
    rows = conn.execute(
        f"""
        SELECT id, source, user_id, parent_session_id, started_at, ended_at,
               end_reason, message_count
        FROM sessions
        WHERE {' AND '.join(where)}
        ORDER BY started_at DESC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    results = []
    for row in rows:
        item = _row_to_session_metadata(row, candidate=True)
        item.update(_fetch_transcript_stats(conn, row["id"]))
        results.append(item)
    return results


def _row_to_session_metadata(row: sqlite3.Row, *, candidate: bool) -> dict[str, Any]:
    result = {
        "session_id": row["id"],
        "parent_session_id": row["parent_session_id"],
        "source": row["source"],
        "user_id": row["user_id"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "end_reason": row["end_reason"],
        "message_count": row["message_count"] or 0,
    }
    if candidate:
        result["candidate_only"] = True
        result["match_type"] = "candidate_only"
        result["candidate_reason"] = (
            "Discord session with transcript rows not currently mapped by "
            "the supplied sessions.json index."
        )
    return result


def _fetch_routing_metadata(
    conn: sqlite3.Connection,
    session_id: str,
) -> dict[str, Any] | None:
    if not _table_exists(conn, "session_routing_metadata"):
        return None
    row = conn.execute(
        """
        SELECT session_key, platform, chat_type, chat_id, thread_id,
               parent_chat_id, guild_id, user_id AS routing_user_id,
               message_id, created_at AS routing_created_at,
               updated_at AS routing_updated_at
        FROM session_routing_metadata
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_routing_metadata(row)


def _fetch_transcript_stats(conn: sqlite3.Connection, session_id: str) -> dict[str, Any]:
    """Return aggregate transcript metadata without reading message content."""
    if not _table_exists(conn, "messages"):
        return {
            "transcript_message_count": None,
            "last_transcript_timestamp": None,
        }
    row = conn.execute(
        """
        SELECT COUNT(*) AS transcript_message_count,
               MAX(timestamp) AS last_transcript_timestamp
        FROM messages
        WHERE session_id = ?
        """,
        (session_id,),
    ).fetchone()
    if row is None:
        return {
            "transcript_message_count": 0,
            "last_transcript_timestamp": None,
        }
    return {
        "transcript_message_count": int(row["transcript_message_count"] or 0),
        "last_transcript_timestamp": row["last_transcript_timestamp"],
    }


def _row_to_routing_metadata(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "session_key": row["session_key"],
        "platform": row["platform"],
        "chat_type": row["chat_type"],
        "chat_id": row["chat_id"],
        "thread_id": row["thread_id"],
        "parent_chat_id": row["parent_chat_id"],
        "guild_id": row["guild_id"],
        "user_id": row["routing_user_id"],
        "message_id": row["message_id"],
        "created_at": row["routing_created_at"],
        "updated_at": row["routing_updated_at"],
    }


def _thread_id_from_session_key(session_key: str) -> str | None:
    parts = session_key.split(":")
    if len(parts) >= 6 and parts[2] == "discord" and parts[3] == "thread":
        return parts[5]
    return None


def _entry_session_id(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("session_id")
    if value is None:
        return None
    return str(value)


def _mapped_session_ids(sessions_index: dict[str, Any]) -> set[str]:
    return {
        session_id
        for session_id in (_entry_session_id(entry) for entry in sessions_index.values())
        if session_id
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report Discord thread session mapping status without repairs.",
    )
    parser.add_argument("--sessions-json", required=True, type=Path)
    parser.add_argument("--state-db", required=True, type=Path)
    parser.add_argument("--session-key", required=True)
    parser.add_argument("--candidate-limit", type=int, default=10)
    args = parser.parse_args(argv)

    report = inspect_discord_thread_mapping(
        sessions_json=args.sessions_json,
        state_db=args.state_db,
        session_key=args.session_key,
        candidate_limit=args.candidate_limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Replayable, user-visible activity events shared by Hermes surfaces.

The TUI and API server can run in different processes, so the activity feed is
kept in a small SQLite journal under the active Hermes home.  Events are
deliberately limited to the same safe, user-visible progress that the desktop
already renders; raw tool arguments and results are never stored here.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


MAX_EVENTS_PER_TURN = 512
MAX_TURNS_PER_SESSION = 50
RETENTION_SECONDS = 30 * 24 * 60 * 60
_COALESCED_TYPES = frozenset({"message.delta", "reasoning.delta"})


class ActivityCursorExpired(ValueError):
    """Raised when a client asks to resume before the retained journal window."""


class SessionActivityStore:
    """SQLite-backed per-session activity journal.

    A fresh connection is opened for each operation.  This keeps the store
    safe for the API server, TUI backend and other Hermes surfaces sharing the
    same home directory without relying on an in-process event bus.
    """

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            from hermes_constants import get_hermes_home

            path = get_hermes_home() / "session_activity.db"
        self.path = Path(path)
        self._lock = threading.RLock()
        self._initialised = False

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        if not self._initialised:
            with self._lock:
                if not self._initialised:
                    conn.executescript(
                        """
                        CREATE TABLE IF NOT EXISTS session_activity_events (
                            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            turn_id TEXT NOT NULL,
                            submission_id TEXT,
                            type TEXT NOT NULL,
                            timestamp REAL NOT NULL,
                            updated_at REAL NOT NULL,
                            surface TEXT NOT NULL,
                            payload_json TEXT NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS idx_session_activity_cursor
                            ON session_activity_events(session_id, event_id);
                        CREATE INDEX IF NOT EXISTS idx_session_activity_turn
                            ON session_activity_events(session_id, turn_id, event_id);
                        CREATE INDEX IF NOT EXISTS idx_session_activity_retention
                            ON session_activity_events(timestamp);
                        """
                    )
                    self._initialised = True
        return conn

    @staticmethod
    def _payload(payload: dict[str, Any] | None) -> str:
        """Serialize a small JSON-safe payload, falling back rather than failing work."""
        try:
            return json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            return "{}"

    @staticmethod
    def _event(row: sqlite3.Row) -> dict[str, Any]:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        return {
            "event_id": str(row["event_id"]),
            "session_id": row["session_id"],
            "turn_id": row["turn_id"],
            "submission_id": row["submission_id"],
            "type": row["type"],
            "timestamp": row["timestamp"],
            "surface": row["surface"],
            "payload": payload if isinstance(payload, dict) else {},
        }

    def append(
        self,
        *,
        session_id: str,
        turn_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        surface: str,
        submission_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Append one event and enforce bounded per-session retention.

        Adjacent streaming deltas update their existing row.  That preserves a
        replayable current thought/message while avoiding an unbounded record
        for token-by-token output.
        """
        session_id = str(session_id or "").strip()
        turn_id = str(turn_id or "").strip()
        event_type = str(event_type or "").strip()
        if not session_id or not turn_id or not event_type:
            return None
        now = time.time()
        payload_json = self._payload(payload)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                existing = None
                if event_type in _COALESCED_TYPES:
                    existing = conn.execute(
                        """
                        SELECT * FROM session_activity_events
                        WHERE session_id = ? AND turn_id = ? AND type = ?
                        ORDER BY event_id DESC LIMIT 1
                        """,
                        (session_id, turn_id, event_type),
                    ).fetchone()
                if existing is not None:
                    previous = self._event(existing)["payload"]
                    if event_type in {"message.delta", "reasoning.delta"}:
                        previous["text"] = (
                            str(previous.get("text") or previous.get("delta") or "")
                            + str((payload or {}).get("text") or (payload or {}).get("delta") or "")
                        )[-20_000:]
                        previous.pop("delta", None)
                    else:
                        previous.update(payload or {})
                    conn.execute(
                        "UPDATE session_activity_events SET payload_json = ?, updated_at = ? WHERE event_id = ?",
                        (self._payload(previous), now, existing["event_id"]),
                    )
                    row = conn.execute(
                        "SELECT * FROM session_activity_events WHERE event_id = ?", (existing["event_id"],)
                    ).fetchone()
                else:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM session_activity_events WHERE session_id = ? AND turn_id = ?",
                        (session_id, turn_id),
                    ).fetchone()[0]
                    if count >= MAX_EVENTS_PER_TURN:
                        conn.execute("COMMIT")
                        return None
                    cursor = conn.execute(
                        """
                        INSERT INTO session_activity_events
                            (session_id, turn_id, submission_id, type, timestamp, updated_at, surface, payload_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (session_id, turn_id, submission_id, event_type, now, now, surface, payload_json),
                    )
                    row = conn.execute(
                        "SELECT * FROM session_activity_events WHERE event_id = ?", (cursor.lastrowid,)
                    ).fetchone()
                self._prune(conn, session_id, now)
                conn.execute("COMMIT")
                return self._event(row) if row is not None else None
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.close()

    @staticmethod
    def _prune(conn: sqlite3.Connection, session_id: str, now: float) -> None:
        conn.execute(
            "DELETE FROM session_activity_events WHERE timestamp < ?", (now - RETENTION_SECONDS,)
        )
        keep_turns = conn.execute(
            """
            SELECT turn_id FROM session_activity_events
            WHERE session_id = ?
            GROUP BY turn_id
            ORDER BY MAX(event_id) DESC
            LIMIT ?
            """,
            (session_id, MAX_TURNS_PER_SESSION),
        ).fetchall()
        if not keep_turns:
            return
        placeholders = ",".join("?" for _ in keep_turns)
        conn.execute(
            f"DELETE FROM session_activity_events WHERE session_id = ? AND turn_id NOT IN ({placeholders})",
            (session_id, *(row["turn_id"] for row in keep_turns)),
        )

    def events(
        self,
        session_id: str,
        *,
        after_event_id: int | None = None,
        before_event_id: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return chronologically ordered retained events for a session."""
        limit = max(1, min(int(limit), 512))
        with self._lock:
            conn = self._connect()
            try:
                clauses = ["session_id = ?"]
                params: list[Any] = [session_id]
                if after_event_id is not None:
                    clauses.append("event_id > ?")
                    params.append(after_event_id)
                if before_event_id is not None:
                    clauses.append("event_id < ?")
                    params.append(before_event_id)
                query = (
                    "SELECT * FROM session_activity_events WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY event_id DESC LIMIT ?"
                )
                rows = conn.execute(query, (*params, limit)).fetchall()
                return [self._event(row) for row in reversed(rows)]
            finally:
                conn.close()

    def assert_cursor_available(self, session_id: str, event_id: int | None) -> None:
        if event_id is None or event_id <= 0:
            return
        with self._lock:
            conn = self._connect()
            try:
                bounds = conn.execute(
                    "SELECT MIN(event_id) AS oldest, MAX(event_id) AS newest FROM session_activity_events WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                oldest = bounds["oldest"] if bounds is not None else None
                newest = bounds["newest"] if bounds is not None else None
                if oldest is not None and newest is not None and event_id < oldest - 1:
                    raise ActivityCursorExpired("Requested activity cursor is no longer retained")
            finally:
                conn.close()

    def latest_cursor(self, session_id: str) -> str | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT MAX(event_id) AS event_id FROM session_activity_events WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return str(row["event_id"]) if row is not None and row["event_id"] is not None else None
            finally:
                conn.close()

    def delete_session(self, session_id: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM session_activity_events WHERE session_id = ?", (session_id,))
            finally:
                conn.close()

    def rekey_session(self, old_session_id: str, new_session_id: str) -> None:
        if not old_session_id or not new_session_id or old_session_id == new_session_id:
            return
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    "UPDATE session_activity_events SET session_id = ? WHERE session_id = ?",
                    (new_session_id, old_session_id),
                )
            finally:
                conn.close()

"""SQLite-backed session memory — survives gateway restarts.

Replaces the in-memory ``router._session_history`` dict with a persistent
store. Schema is keyed by ``(profile_name, user_key, ts)`` so each user's
history is isolated per profile.

Persistence semantics:
  - ``append`` adds a single (role, content) row with current timestamp.
  - ``load_recent`` returns the last N messages for a (profile, user)
    ordered by timestamp ascending — same shape the LLM expects.
  - ``clear`` removes every row for a (profile, user). The table is small
    enough that hard delete is fine.

Default DB path is ``~/.hermes/multitenancy.db`` so it co-resides with
``RoutingTable`` — they're conceptually one "multitenancy state" store.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = Path.home() / ".hermes" / "multitenancy.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS multitenancy_sessions (
    profile_name TEXT NOT NULL,
    user_key     TEXT NOT NULL,
    ts           INTEGER NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    PRIMARY KEY (profile_name, user_key, ts, role)
);
CREATE INDEX IF NOT EXISTS idx_sessions_profile_user
    ON multitenancy_sessions(profile_name, user_key, ts);
"""


class SessionStore:
    """Persistent (profile, user) → conversation history store."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = str(db_path) if db_path is not None else str(DEFAULT_DB_PATH)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def append(self, profile_name: str, user_key: str, role: str, content: str) -> None:
        """Append one message to a user's history. Microsecond ts dedups bursts."""
        ts = time.monotonic_ns()
        self._conn.execute(
            "INSERT OR IGNORE INTO multitenancy_sessions"
            " (profile_name, user_key, ts, role, content) VALUES (?, ?, ?, ?, ?)",
            (profile_name, user_key, ts, role, content),
        )
        self._conn.commit()

    def load_recent(self, profile_name: str, user_key: str, limit: int) -> list[dict]:
        """Return the last ``limit`` messages oldest-first (LLM order)."""
        cur = self._conn.execute(
            "SELECT role, content FROM multitenancy_sessions"
            " WHERE profile_name = ? AND user_key = ?"
            " ORDER BY ts DESC LIMIT ?",
            (profile_name, user_key, limit),
        )
        rows = list(cur.fetchall())
        rows.reverse()  # back to oldest-first
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def clear(self, profile_name: str, user_key: str) -> int:
        """Hard-delete a user's history. Returns rows removed."""
        cur = self._conn.execute(
            "DELETE FROM multitenancy_sessions WHERE profile_name = ? AND user_key = ?",
            (profile_name, user_key),
        )
        self._conn.commit()
        return cur.rowcount

    def count(self, profile_name: str, user_key: str) -> int:
        """Number of messages stored for a (profile, user). Diagnostic only."""
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM multitenancy_sessions"
            " WHERE profile_name = ? AND user_key = ?",
            (profile_name, user_key),
        )
        return int(cur.fetchone()[0])

    def close(self) -> None:
        self._conn.close()

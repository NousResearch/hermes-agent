"""Shared thread ownership state for Mattermost profiles.

Each Hermes Mattermost profile runs as an independent process. To support
"sticky" thread behavior across profiles, active thread ownership must live in
shared storage rather than in-memory state.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from hermes_constants import get_default_hermes_root


class MattermostThreadStateStore:
    """Persistent shared store for Mattermost thread ownership."""

    def __init__(self, ttl_seconds: int = 7 * 24 * 3600):
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._path = self._db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @staticmethod
    def _db_path() -> Path:
        return get_default_hermes_root() / "shared" / "mattermost-thread-state.sqlite3"

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mattermost_thread_state (
                    platform TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    active_agent_profile TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY (platform, channel_id, thread_id)
                )
                """
            )

    def _purge_expired(self, conn: sqlite3.Connection) -> None:
        if self._ttl_seconds <= 0:
            return
        cutoff = int(time.time()) - self._ttl_seconds
        conn.execute(
            """
            DELETE FROM mattermost_thread_state
            WHERE platform = 'mattermost' AND updated_at < ?
            """,
            (cutoff,),
        )

    def claim_thread(self, channel_id: str, thread_id: str, active_agent_profile: str) -> None:
        now = int(time.time())
        with self._lock:
            with self._connect() as conn:
                self._purge_expired(conn)
                conn.execute(
                    """
                    INSERT INTO mattermost_thread_state
                        (platform, channel_id, thread_id, active_agent_profile, updated_at)
                    VALUES ('mattermost', ?, ?, ?, ?)
                    ON CONFLICT(platform, channel_id, thread_id)
                    DO UPDATE SET
                        active_agent_profile = excluded.active_agent_profile,
                        updated_at = excluded.updated_at
                    """,
                    (channel_id, thread_id, active_agent_profile, now),
                )

    def get_active_agent(self, channel_id: str, thread_id: str) -> Optional[str]:
        with self._lock:
            with self._connect() as conn:
                self._purge_expired(conn)
                row = conn.execute(
                    """
                    SELECT active_agent_profile
                    FROM mattermost_thread_state
                    WHERE platform = 'mattermost' AND channel_id = ? AND thread_id = ?
                    """,
                    (channel_id, thread_id),
                ).fetchone()
        return row[0] if row else None


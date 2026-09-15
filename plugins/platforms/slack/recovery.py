"""Durable state for Slack reconnect message recovery."""

from __future__ import annotations

import datetime as dt
import logging
import os
import sqlite3
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DB_FILENAME = "slack_message_recovery.db"
_RETENTION_DAYS = 30


class SlackRecoveryStore:
    """Small profile-scoped SQLite ledger for completed Slack messages."""

    def __init__(self, hermes_home: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self._hermes_home = Path(hermes_home or get_hermes_home())

    def path(self) -> Path:
        directory = self._hermes_home / "gateway"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / _DB_FILENAME

    def call(self, fn: Callable[[sqlite3.Connection], Any], default: Any = None) -> Any:
        try:
            with self._lock:
                path = self.path()
                conn = sqlite3.connect(path, timeout=0.1)
                try:
                    if not self._initialized:
                        self._initialize(conn)
                        self._initialized = True
                        with suppress(OSError):
                            os.chmod(path, 0o600)
                    result = fn(conn)
                    conn.commit()
                    return result
                finally:
                    conn.close()
        except Exception as exc:
            logger.warning("Slack recovery ledger unavailable: %s", exc)
            return default

    def _initialize(self, conn: sqlite3.Connection) -> None:
        from hermes_state_wal import apply_wal_with_fallback

        apply_wal_with_fallback(conn, db_label="slack_recovery.db")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS slack_messages (
                team_id TEXT NOT NULL, channel_id TEXT NOT NULL, message_ts TEXT NOT NULL,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                last_attempt_at TEXT, last_error TEXT, updated_at TEXT NOT NULL,
                PRIMARY KEY (team_id, channel_id, message_ts)
            );
            CREATE TABLE IF NOT EXISTS slack_recovery_cursors (
                team_id TEXT NOT NULL, channel_id TEXT NOT NULL,
                last_message_ts TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY (team_id, channel_id)
            );
        """)
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=_RETENTION_DAYS)).isoformat()
        conn.execute("DELETE FROM slack_messages WHERE updated_at < ?", (cutoff,))
        conn.execute("DELETE FROM slack_recovery_cursors WHERE updated_at < ?", (cutoff,))

"""Per-session router state: pins and loop-escalation counters (SQLite).

State survives gateway restarts so a session's pin (and its cache economics)
is durable. All operations are failure-safe: a broken database degrades to
"no state", never to a routing error. Owner-only file permissions.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from .types import SessionPin

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_pins (
    session_id TEXT PRIMARY KEY,
    pinned_model_id TEXT NOT NULL,
    pin_reason TEXT NOT NULL DEFAULT 'auto',
    turns_held INTEGER NOT NULL DEFAULT 0,
    consecutive_tool_failures INTEGER NOT NULL DEFAULT 0,
    last_tool_failure_signature TEXT,
    updated_at REAL NOT NULL
);
"""


class RouterStateStore:
    """SQLite-backed session pin store. Thread-safe, failure-safe."""

    def __init__(self, db_path, *, read_only: bool = False):
        self._db_path = str(db_path)
        self._read_only = bool(read_only)
        self._lock = threading.Lock()
        self._available = True
        if self._read_only:
            self._available = Path(self._db_path).is_file()
            return
        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
            os.chmod(self._db_path, 0o600)
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def _connect(self) -> sqlite3.Connection:
        if self._read_only:
            uri = Path(self._db_path).resolve().as_uri() + "?mode=ro"
            return sqlite3.connect(uri, uri=True, timeout=5)
        conn = sqlite3.connect(self._db_path, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def load_pin(self, session_id: str) -> Optional[SessionPin]:
        if not self._available or not session_id:
            return None
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT session_id, pinned_model_id, pin_reason, turns_held,"
                    " consecutive_tool_failures, last_tool_failure_signature, updated_at"
                    " FROM session_pins WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        return SessionPin(
            session_id=row[0],
            pinned_model_id=row[1],
            pin_reason=row[2],
            turns_held=int(row[3]),
            consecutive_tool_failures=int(row[4]),
            last_tool_failure_signature=row[5],
            updated_at=float(row[6]),
        )

    def save_pin(self, pin: SessionPin) -> None:
        if self._read_only or not self._available or not pin.session_id:
            return
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO session_pins (session_id, pinned_model_id, pin_reason,"
                    " turns_held, consecutive_tool_failures, last_tool_failure_signature, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(session_id) DO UPDATE SET"
                    " pinned_model_id=excluded.pinned_model_id,"
                    " pin_reason=excluded.pin_reason,"
                    " turns_held=excluded.turns_held,"
                    " consecutive_tool_failures=excluded.consecutive_tool_failures,"
                    " last_tool_failure_signature=excluded.last_tool_failure_signature,"
                    " updated_at=excluded.updated_at",
                    (
                        pin.session_id,
                        pin.pinned_model_id,
                        pin.pin_reason,
                        pin.turns_held,
                        pin.consecutive_tool_failures,
                        pin.last_tool_failure_signature,
                        pin.updated_at or time.time(),
                    ),
                )
        except Exception:
            pass

    def clear_pin(self, session_id: str) -> None:
        if self._read_only or not self._available or not session_id:
            return
        try:
            with self._lock, self._connect() as conn:
                conn.execute("DELETE FROM session_pins WHERE session_id = ?", (session_id,))
        except Exception:
            pass

    def list_pins(self) -> list:
        if not self._available:
            return []
        try:
            with self._lock, self._connect() as conn:
                rows = conn.execute(
                    "SELECT session_id, pinned_model_id, pin_reason, turns_held, updated_at"
                    " FROM session_pins ORDER BY updated_at DESC LIMIT 100"
                ).fetchall()
        except Exception:
            return []
        return [
            {
                "session_id": r[0],
                "pinned_model_id": r[1],
                "pin_reason": r[2],
                "turns_held": r[3],
                "updated_at": r[4],
            }
            for r in rows
        ]

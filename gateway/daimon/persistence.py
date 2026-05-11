"""SQLite persistence for Daimon state.

Stores thread ownership, turn counts, daily usage, and bans.
Write-through pattern: in-memory dicts for fast reads, SQLite for durability.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS thread_ownership (
    thread_id TEXT PRIMARY KEY,
    creator_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS daily_usage (
    user_date TEXT PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS bans (
    user_id TEXT PRIMARY KEY,
    banned_at REAL NOT NULL,
    reason TEXT DEFAULT ''
);
"""


class DaimonDB:
    """SQLite persistence for Daimon session state.

    Thread-safe. Uses WAL mode for concurrent read/write performance.
    """

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist and run migrations."""
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)
            # Check/set schema version
            cur = self._conn.execute("SELECT MAX(version) FROM schema_version")
            row = cur.fetchone()
            current = row[0] if row and row[0] else 0
            if current < _SCHEMA_VERSION:
                self._conn.execute(
                    "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                    (_SCHEMA_VERSION,),
                )
                self._conn.commit()

    # ── Thread Ownership ──────────────────────────────────────────────────

    def register_thread(self, thread_id: str, creator_id: str) -> None:
        """Record thread ownership."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO thread_ownership (thread_id, creator_id, created_at, turn_count) "
                "VALUES (?, ?, ?, 0)",
                (thread_id, creator_id, time.time()),
            )
            self._conn.commit()

    def get_thread_owner(self, thread_id: str) -> Optional[str]:
        """Get creator of a thread, or None if not tracked."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT creator_id FROM thread_ownership WHERE thread_id = ?",
                (thread_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def unregister_thread(self, thread_id: str) -> None:
        """Remove a thread from tracking."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM thread_ownership WHERE thread_id = ?", (thread_id,)
            )
            self._conn.commit()

    def get_all_threads(self) -> dict[str, str]:
        """Load all thread → creator mappings for startup recovery."""
        with self._lock:
            cur = self._conn.execute("SELECT thread_id, creator_id FROM thread_ownership")
            return {row[0]: row[1] for row in cur.fetchall()}

    # ── Turn Counting ─────────────────────────────────────────────────────

    def get_turn_count(self, thread_id: str) -> int:
        """Get current turn count for a thread."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT turn_count FROM thread_ownership WHERE thread_id = ?",
                (thread_id,),
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def increment_turn(self, thread_id: str) -> int:
        """Increment turn count, return new value."""
        with self._lock:
            self._conn.execute(
                "UPDATE thread_ownership SET turn_count = turn_count + 1 WHERE thread_id = ?",
                (thread_id,),
            )
            self._conn.commit()
            cur = self._conn.execute(
                "SELECT turn_count FROM thread_ownership WHERE thread_id = ?",
                (thread_id,),
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def clear_turns(self, thread_id: str) -> None:
        """Reset turn count (or just delete via unregister_thread)."""
        with self._lock:
            self._conn.execute(
                "UPDATE thread_ownership SET turn_count = 0 WHERE thread_id = ?",
                (thread_id,),
            )
            self._conn.commit()

    # ── Daily Usage ───────────────────────────────────────────────────────

    def get_daily_usage(self, user_id: str) -> int:
        """Get today's usage count for a user."""
        key = f"{user_id}:{date.today().isoformat()}"
        with self._lock:
            cur = self._conn.execute(
                "SELECT count FROM daily_usage WHERE user_date = ?", (key,)
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def increment_daily_usage(self, user_id: str) -> int:
        """Increment today's usage, return new count."""
        key = f"{user_id}:{date.today().isoformat()}"
        with self._lock:
            self._conn.execute(
                "INSERT INTO daily_usage (user_date, count) VALUES (?, 1) "
                "ON CONFLICT(user_date) DO UPDATE SET count = count + 1",
                (key,),
            )
            self._conn.commit()
            cur = self._conn.execute(
                "SELECT count FROM daily_usage WHERE user_date = ?", (key,)
            )
            row = cur.fetchone()
            return row[0] if row else 1

    def get_all_daily_usage(self) -> dict[str, int]:
        """Load all daily usage records (for startup, filtered to today)."""
        today_str = date.today().isoformat()
        with self._lock:
            cur = self._conn.execute(
                "SELECT user_date, count FROM daily_usage WHERE user_date LIKE ?",
                (f"%:{today_str}",),
            )
            return {row[0]: row[1] for row in cur.fetchall()}

    def cleanup_old_daily_usage(self, days_to_keep: int = 7) -> int:
        """Remove daily usage records older than N days. Returns rows deleted."""
        cutoff = date.today().isoformat()
        # Simple approach: delete all entries that don't end with recent dates
        # Since key format is "user_id:YYYY-MM-DD", we can compare lexicographically
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM daily_usage")
            before = cur.fetchone()[0]
            # Keep only entries from the last N days
            from datetime import timedelta
            keep_dates = {(date.today() - timedelta(days=i)).isoformat() for i in range(days_to_keep)}
            placeholders = ",".join("?" * len(keep_dates))
            # Delete entries where the date portion doesn't match any recent date
            self._conn.execute(
                f"DELETE FROM daily_usage WHERE substr(user_date, -10) NOT IN ({placeholders})",
                tuple(keep_dates),
            )
            self._conn.commit()
            cur = self._conn.execute("SELECT COUNT(*) FROM daily_usage")
            after = cur.fetchone()[0]
            return before - after

    # ── Bans ──────────────────────────────────────────────────────────────

    def ban_user(self, user_id: str, reason: str = "") -> None:
        """Ban a user."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO bans (user_id, banned_at, reason) VALUES (?, ?, ?)",
                (user_id, time.time(), reason),
            )
            self._conn.commit()

    def unban_user(self, user_id: str) -> None:
        """Remove a ban."""
        with self._lock:
            self._conn.execute("DELETE FROM bans WHERE user_id = ?", (user_id,))
            self._conn.commit()

    def is_banned(self, user_id: str) -> bool:
        """Check if user is banned."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM bans WHERE user_id = ?", (user_id,)
            )
            return cur.fetchone() is not None

    def get_all_bans(self) -> set[str]:
        """Load all banned user IDs for startup recovery."""
        with self._lock:
            cur = self._conn.execute("SELECT user_id FROM bans")
            return {row[0] for row in cur.fetchall()}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
        except Exception:
            pass

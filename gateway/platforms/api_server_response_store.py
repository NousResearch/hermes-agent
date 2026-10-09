"""SQLite-backed store for Responses API state (``previous_response_id`` chaining, named conversations)."""

import json
import logging
import sqlite3
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, Optional

# Logger parity with the origin module (moved log records keep their name).
logger = logging.getLogger("gateway.platforms.api_server")

MAX_STORED_RESPONSES = 100


class ResponseStore:
    """SQLite-backed LRU store for Responses API state (full conversation history per response
    for ``previous_response_id`` chaining). Persists across restarts; in-memory fallback."""

    def __init__(self, max_size: int = MAX_STORED_RESPONSES, db_path: str = None):
        self._max_size = max_size
        if db_path is None:
            db_path = ":memory:"
            with suppress(Exception):
                from hermes_cli.config import get_hermes_home
                db_path = str(get_hermes_home() / "response_store.db")
        self._db_path: Optional[str] = db_path if db_path != ":memory:" else None
        try:
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        except Exception:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._db_path = None
        # Shared WAL-fallback so response_store.db degrades gracefully on NFS/SMB/FUSE homes.
        from hermes_state_wal import apply_wal_with_fallback
        apply_wal_with_fallback(self._conn, db_label="response_store.db")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "response_id TEXT PRIMARY KEY, data TEXT NOT NULL, accessed_at REAL NOT NULL)")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS conversations (name TEXT PRIMARY KEY, response_id TEXT NOT NULL)")
        self._conn.commit()
        # Conversation history lives here: owner-only perms, once at init (not per commit).
        self._tighten_file_permissions()

    def _tighten_file_permissions(self) -> None:
        """Force owner-only permissions on the DB and SQLite sidecars."""
        if not self._db_path:
            return
        for candidate in (Path(self._db_path), Path(f"{self._db_path}-wal"), Path(f"{self._db_path}-shm")):
            try:
                if candidate.exists():
                    candidate.chmod(0o600)
            except OSError:
                logger.debug("Failed to restrict response store permissions for %s", candidate, exc_info=True)

    def get(self, response_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored response by ID (updates access time for LRU)."""
        row = self._conn.execute(
            "SELECT data FROM responses WHERE response_id = ?", (response_id,)).fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE responses SET accessed_at = ? WHERE response_id = ?",
            (time.time(), response_id))
        self._conn.commit()
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupted JSON in response store for id=%s, evicting entry", response_id)
            self._conn.execute("DELETE FROM responses WHERE response_id = ?", (response_id,))
            self._conn.commit()
            return None

    def put(self, response_id: str, data: Dict[str, Any]) -> None:
        """Store a response, evicting the oldest if at capacity."""
        self._conn.execute(
            "INSERT OR REPLACE INTO responses (response_id, data, accessed_at) VALUES (?, ?, ?)",
            (response_id, json.dumps(data, default=str), time.time()))
        count = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        if count > self._max_size:
            evict_ids = [row[0] for row in self._conn.execute(
                "SELECT response_id FROM responses ORDER BY accessed_at ASC LIMIT ?",
                (count - self._max_size,)).fetchall()]
            if evict_ids:
                placeholders = ",".join("?" for _ in evict_ids)
                # Conversation mappings pointing at evicted responses go too.
                self._conn.execute(f"DELETE FROM conversations WHERE response_id IN ({placeholders})", evict_ids)
                self._conn.execute(f"DELETE FROM responses WHERE response_id IN ({placeholders})", evict_ids)
        self._conn.commit()

    def delete(self, response_id: str) -> bool:
        """Remove a response (and conversation mappings to it). True if found and deleted."""
        self._conn.execute("DELETE FROM conversations WHERE response_id = ?", (response_id,))
        cursor = self._conn.execute("DELETE FROM responses WHERE response_id = ?", (response_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def get_conversation(self, name: str) -> Optional[str]:
        """Get the latest response_id for a conversation name."""
        row = self._conn.execute("SELECT response_id FROM conversations WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def set_conversation(self, name: str, response_id: str) -> None:
        """Map a conversation name to its latest response_id."""
        self._conn.execute("INSERT OR REPLACE INTO conversations (name, response_id) VALUES (?, ?)", (name, response_id))
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        with suppress(Exception):
            self._conn.close()

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()
        return row[0] if row else 0

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
        # Compact immutable request identity outlives the LRU response bodies.
        # Distinct canonical session targets must not turn a global retry key
        # into a last-writer-wins response cache entry.
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS response_keys (request_key TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, created_at INTEGER NOT NULL)")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS response_admissions (request_key TEXT PRIMARY KEY, admission_id TEXT NOT NULL)")
        self._conn.execute('CREATE TABLE IF NOT EXISTS response_output_order '
            '(request_key TEXT NOT NULL, item_id TEXT NOT NULL, output_index INTEGER NOT NULL, '
            'PRIMARY KEY(request_key,item_id), UNIQUE(request_key,output_index))')
        from storage.sqlite_util import add_column_if_missing
        add_column_if_missing(self._conn, 'response_keys', 'model_json', 'model_json TEXT')
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
        insert = 'INSERT OR IGNORE' if response_id.startswith('idem:') else 'INSERT OR REPLACE'
        self._conn.execute(
            insert + " INTO responses (response_id, data, accessed_at) VALUES (?, ?, ?)",
            (response_id, json.dumps(data, default=str), time.time()))
        self._evict_and_commit()

    def _evict_and_commit(self) -> None:
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

    def bind_request_key(self, key: str, fingerprint: str, *, model=None) -> bool:
        """Reserve one immutable digest before any canonical admission can run."""
        with self._conn:
            self._conn.execute('INSERT OR IGNORE INTO response_keys '
                '(request_key,fingerprint,created_at,model_json) VALUES (?,?,?,?)',
                (key, fingerprint, int(time.time()), json.dumps(model)))
            row = self._conn.execute('SELECT fingerprint FROM response_keys WHERE request_key=?', (key,)).fetchone()
        return row[0] == fingerprint

    def request_model(self, key: str, fallback):
        """The accepted wire label survives route changes and response-body LRU eviction."""
        with self._conn:
            # Upgrade older compact bindings only on a cache miss; their existing cached
            # envelopes keep replaying byte-for-byte through the ordinary fast path.
            self._conn.execute('UPDATE response_keys SET model_json=? WHERE request_key=? AND model_json IS NULL',
                               (json.dumps(fallback), key))
            row = self._conn.execute('SELECT model_json FROM response_keys WHERE request_key=?', (key,)).fetchone()
        return json.loads(row[0]) if row is not None else fallback

    def output_index(self, key: str, item_id: str) -> int:
        """Allocate before live publication; retries and parallel observers share this order.

        One compact id/integer pair per output item survives body LRU eviction. No tool output,
        reasoning text or cumulative conversation is copied into this identity table.
        """
        with self._conn:
            self._conn.execute('INSERT OR IGNORE INTO response_output_order '
                '(request_key,item_id,output_index) SELECT ?,?,COALESCE(MAX(output_index)+1,0) '
                'FROM response_output_order WHERE request_key=?', (key, item_id, key))
            row = self._conn.execute('SELECT output_index FROM response_output_order WHERE request_key=? AND item_id=?',
                                     (key, item_id)).fetchone()
        return row[0]

    def order_output(self, key: str, items):
        return sorted(items, key=lambda item: self.output_index(key, item['id']))

    def request_created_at(self, key: str) -> Optional[int]:
        row = self._conn.execute('SELECT created_at FROM response_keys WHERE request_key=?', (key,)).fetchone()
        return row[0] if row else None

    def request_key_matches(self, key: str, fingerprint: str) -> bool:
        row = self._conn.execute('SELECT fingerprint FROM response_keys WHERE request_key=?', (key,)).fetchone()
        return row is None or row[0] == fingerprint

    def request_admission(self, key: str) -> Optional[str]:
        row = self._conn.execute('SELECT admission_id FROM response_admissions WHERE request_key=?', (key,)).fetchone()
        return row[0] if row else None

    def bind_request_admission(self, key: str, admission_id: str) -> None:
        from hermes_state_runtime import RuntimeStoreError
        with self._conn:
            self._conn.execute('INSERT OR IGNORE INTO response_admissions VALUES (?,?)', (key, admission_id))
            if self.request_admission(key) != admission_id:
                raise RuntimeStoreError('admission_conflict')

    def forget_unadmitted_request(self, key: str) -> None:
        with self._conn:
            self._conn.execute('DELETE FROM response_keys WHERE request_key=? AND NOT EXISTS '
                '(SELECT 1 FROM response_admissions WHERE request_key=?)', (key, key))

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

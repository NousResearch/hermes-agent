"""
SQLite-backed vector store for cognitive memory.

Stores memories with their embeddings, metadata, and importance scores.
Uses a dedicated SQLite database at ~/.hermes/cognitive_memory.db
to avoid conflicts with the existing state.db.
"""

import json
import logging
import os
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cognitive_memory.embeddings import cosine_similarity

logger = logging.getLogger(__name__)


def _escape_like(value: str) -> str:
    """Escape special LIKE pattern characters (_ and %)."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

# Default database location
DEFAULT_DB_DIR = os.path.join(
    os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
    "cognitive_memory",
)


@dataclass
class Memory:
    """A single memory entry with metadata."""
    id: int
    content: str
    scope: str
    categories: List[str]
    importance: float
    created_at: float
    updated_at: float
    last_accessed: float
    access_count: int
    forgotten: bool
    embedding: Optional[List[float]] = None


@dataclass
class ScoredMemory:
    """A memory with its retrieval score and match metadata."""
    memory: Memory
    score: float
    similarity: float = 0.0
    match_reasons: List[str] = field(default_factory=list)


def _serialize_embedding(embedding: List[float]) -> bytes:
    """Pack a float list into a compact binary blob."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _deserialize_embedding(blob: bytes) -> List[float]:
    """Unpack a binary blob back into a float list."""
    count = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"{count}f", blob))


class CognitiveStore:
    """
    SQLite-backed store for memories with vector embeddings.

    Thread-safe with connection-per-thread pattern.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            os.makedirs(DEFAULT_DB_DIR, exist_ok=True)
            db_path = os.path.join(DEFAULT_DB_DIR, "cognitive_memory.db")

        self._db_path = db_path
        self._local = threading.local()
        self._ensure_tables()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cognitive_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB,
                scope TEXT NOT NULL DEFAULT '/',
                categories TEXT NOT NULL DEFAULT '[]',
                importance REAL NOT NULL DEFAULT 0.5,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                forgotten INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_cm_scope
                ON cognitive_memories(scope);
            CREATE INDEX IF NOT EXISTS idx_cm_importance
                ON cognitive_memories(importance);
            CREATE INDEX IF NOT EXISTS idx_cm_forgotten
                ON cognitive_memories(forgotten);
            CREATE INDEX IF NOT EXISTS idx_cm_created
                ON cognitive_memories(created_at);

            CREATE TABLE IF NOT EXISTS cognitive_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)

        # Store schema version
        conn.execute(
            "INSERT OR REPLACE INTO cognitive_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(self.SCHEMA_VERSION)),
        )
        conn.commit()

    def add_memory(
        self,
        content: str,
        embedding: Optional[List[float]] = None,
        scope: str = "/",
        importance: float = 0.5,
        categories: Optional[List[str]] = None,
    ) -> int:
        """
        Add a new memory to the store.

        Returns the memory ID.
        """
        now = time.time()
        conn = self._get_conn()

        emb_blob = _serialize_embedding(embedding) if embedding else None
        cats_json = json.dumps(categories or [])

        cursor = conn.execute(
            """
            INSERT INTO cognitive_memories
                (content, embedding, scope, categories, importance,
                 created_at, updated_at, last_accessed, access_count, forgotten)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0)
            """,
            (content, emb_blob, scope, cats_json, importance, now, now, now),
        )
        conn.commit()
        memory_id = cursor.lastrowid
        logger.debug("Added memory #%d: %.50s...", memory_id, content)
        return memory_id

    def get_memory(self, memory_id: int) -> Optional[Memory]:
        """Retrieve a single memory by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM cognitive_memories WHERE id = ?", (memory_id,)
        ).fetchone()

        if row is None:
            return None

        return self._row_to_memory(row)

    def get_all_active(self, scope: Optional[str] = None) -> List[Memory]:
        """
        Get all non-forgotten memories, optionally filtered by scope prefix.
        """
        conn = self._get_conn()
        if scope:
            rows = conn.execute(
                """
                SELECT * FROM cognitive_memories
                WHERE forgotten = 0 AND scope LIKE ? ESCAPE '\\'
                ORDER BY importance DESC, created_at DESC
                """,
                (_escape_like(scope) + "%",),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM cognitive_memories
                WHERE forgotten = 0
                ORDER BY importance DESC, created_at DESC
                """
            ).fetchall()

        return [self._row_to_memory(row) for row in rows]

    def search_similar(
        self,
        query_embedding: List[float],
        threshold: float = 0.3,
        limit: int = 10,
        include_forgotten: bool = False,
    ) -> List[ScoredMemory]:
        """
        Find memories similar to the query embedding.

        Returns memories sorted by cosine similarity (descending).
        Only returns memories with similarity >= threshold.
        """
        conn = self._get_conn()

        forgotten_filter = "" if include_forgotten else "AND forgotten = 0"
        rows = conn.execute(
            f"""
            SELECT * FROM cognitive_memories
            WHERE embedding IS NOT NULL {forgotten_filter}
            """
        ).fetchall()

        scored = []
        for row in rows:
            emb = _deserialize_embedding(row["embedding"])
            sim = cosine_similarity(query_embedding, emb)

            if sim >= threshold:
                memory = self._row_to_memory(row, embedding=emb)
                scored.append(ScoredMemory(
                    memory=memory,
                    score=sim,
                    similarity=sim,
                    match_reasons=["semantic"],
                ))

        # Sort by similarity descending
        scored.sort(key=lambda s: s.similarity, reverse=True)

        return scored[:limit]

    def update_memory(
        self,
        memory_id: int,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        scope: Optional[str] = None,
        importance: Optional[float] = None,
        categories: Optional[List[str]] = None,
    ) -> bool:
        """Update fields of an existing memory. Returns True if found."""
        conn = self._get_conn()
        now = time.time()

        updates = ["updated_at = ?"]
        params: list = [now]

        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if embedding is not None:
            updates.append("embedding = ?")
            params.append(_serialize_embedding(embedding))
        if scope is not None:
            updates.append("scope = ?")
            params.append(scope)
        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)
        if categories is not None:
            updates.append("categories = ?")
            params.append(json.dumps(categories))

        params.append(memory_id)
        cursor = conn.execute(
            f"UPDATE cognitive_memories SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()
        return cursor.rowcount > 0

    def record_access(self, memory_id: int):
        """Record that a memory was accessed (for importance tracking)."""
        conn = self._get_conn()
        conn.execute(
            """
            UPDATE cognitive_memories
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
            """,
            (time.time(), memory_id),
        )
        conn.commit()

    def soft_delete(self, memory_id: int) -> bool:
        """Mark a memory as forgotten (soft delete). Returns True if found."""
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE cognitive_memories SET forgotten = 1, updated_at = ? WHERE id = ?",
            (time.time(), memory_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def soft_delete_by_scope(
        self,
        scope: str,
        older_than_days: Optional[int] = None,
    ) -> int:
        """
        Soft-delete memories matching scope prefix and optional age filter.

        Returns the number of memories affected.
        """
        conn = self._get_conn()
        params: list = [time.time(), _escape_like(scope) + "%"]

        age_filter = ""
        if older_than_days is not None:
            cutoff = time.time() - (older_than_days * 86400)
            age_filter = "AND created_at < ?"
            params.append(cutoff)

        cursor = conn.execute(
            f"""
            UPDATE cognitive_memories
            SET forgotten = 1, updated_at = ?
            WHERE forgotten = 0 AND scope LIKE ? ESCAPE '\\' {age_filter}
            """,
            params,
        )
        conn.commit()
        return cursor.rowcount

    def decay_importance(
        self,
        half_life_days: float = 30.0,
        exempt_scopes: Optional[List[str]] = None,
    ) -> int:
        """
        Apply importance decay to all active memories.

        Formula: new_importance = importance * 0.5^(days_since_access / half_life)

        Returns the number of memories updated.
        """
        now = time.time()
        exempt = exempt_scopes or []
        conn = self._get_conn()

        rows = conn.execute(
            "SELECT id, importance, last_accessed, scope FROM cognitive_memories WHERE forgotten = 0"
        ).fetchall()

        updated = 0
        for row in rows:
            # Skip exempt scopes
            if any(row["scope"].startswith(s) for s in exempt):
                continue

            days_since = (now - row["last_accessed"]) / 86400
            if days_since <= 0:
                continue

            decay = 0.5 ** (days_since / half_life_days)
            new_importance = row["importance"] * decay

            if abs(new_importance - row["importance"]) > 0.001:
                conn.execute(
                    "UPDATE cognitive_memories SET importance = ? WHERE id = ?",
                    (new_importance, row["id"]),
                )
                updated += 1

        conn.commit()
        return updated

    def prune(self, threshold: float = 0.05) -> int:
        """
        Soft-delete memories with importance below threshold.

        Returns the number of memories pruned.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            UPDATE cognitive_memories
            SET forgotten = 1, updated_at = ?
            WHERE forgotten = 0 AND importance < ?
            """,
            (time.time(), threshold),
        )
        conn.commit()
        return cursor.rowcount

    def count(self, include_forgotten: bool = False) -> int:
        """Count memories in the store."""
        conn = self._get_conn()
        if include_forgotten:
            row = conn.execute("SELECT COUNT(*) FROM cognitive_memories").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM cognitive_memories WHERE forgotten = 0"
            ).fetchone()
        return row[0]

    def close(self):
        """Close the thread-local database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _row_to_memory(
        self, row: sqlite3.Row, embedding: Optional[List[float]] = None
    ) -> Memory:
        """Convert a database row to a Memory dataclass."""
        emb = embedding
        if emb is None and row["embedding"]:
            emb = _deserialize_embedding(row["embedding"])

        return Memory(
            id=row["id"],
            content=row["content"],
            scope=row["scope"],
            categories=json.loads(row["categories"]),
            importance=row["importance"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_accessed=row["last_accessed"],
            access_count=row["access_count"],
            forgotten=bool(row["forgotten"]),
            embedding=emb,
        )

"""FTS5 index over the builtin memory store.

SQLite FTS5 provides BM25-ranked full-text search over memory entries.
Index is rebuilt from scratch when the memory file changes (tracked by mtime).
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

FTS5_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    target,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS fts_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class FTS5Index:
    """FTS5 full-text search index for memory entries.

    Thread-safe. The index is stored alongside the memory files in the
    HERMES_HOME/memories/ directory.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._connected = False

    def connect(self) -> None:
        """Open the SQLite database and create tables if needed."""
        if self._connected:
            return
        try:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=10.0,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(FTS5_SCHEMA)
            self._connected = True
            logger.debug("FTS5 index opened: %s", self._db_path)
        except Exception as e:
            logger.warning("FTS5 index init failed: %s", e)
            self._connected = False

    def rebuild(self, entries: List[str]) -> int:
        """Rebuild the entire FTS5 index from a list of entry strings.

        Args:
            entries: Raw memory entry strings (tags intact).

        Returns:
            Number of entries indexed, or 0 on failure.
        """
        with self._lock:
            try:
                self.connect()
                if not self._connected or self._conn is None:
                    return 0

                self._conn.execute("DELETE FROM memory_fts")
                self._conn.execute("DELETE FROM fts_meta")

                count = 0
                for i, entry in enumerate(entries):
                    # FTS5 indexes on cleaned content + target
                    clean = entry.strip()
                    if not clean:
                        continue
                    target = "memory"  # could be "user" for user profile entries
                    # Use INSERT, not UPDATE — FTS5 doesn't support upsert on content tables
                    self._conn.execute(
                        "INSERT INTO memory_fts (content, target) VALUES (?, ?)",
                        (clean, target),
                    )
                    count += 1

                self._conn.execute(
                    "INSERT OR REPLACE INTO fts_meta (key, value) VALUES (?, ?)",
                    ("entry_count", str(count)),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO fts_meta (key, value) VALUES (?, ?)",
                    ("updated_at", str(time.time())),
                )
                self._conn.commit()
                logger.debug("FTS5 index rebuilt: %d entries", count)
                return count
            except Exception as e:
                logger.warning("FTS5 rebuild failed: %s", e)
                return 0

    def search(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Dict]:
        """Search the FTS5 index and return ranked results.

        Args:
            query: Free-text search query.
            limit: Max results to return.

        Returns:
            List of dicts with keys: content, target, score (BM25 rank).
        """
        with self._lock:
            try:
                self.connect()
                if not self._connected or self._conn is None:
                    return []

                # Sanitize query for FTS5 syntax
                safe_query = self._sanitize_query(query)
                if not safe_query:
                    return []

                rows = self._conn.execute(
                    """
                    SELECT content, target, rank
                    FROM memory_fts
                    WHERE memory_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (safe_query, limit),
                ).fetchall()

                # Normalize BM25 rank to [0, 1] score
                # FTS5 rank is negative (smaller = better match).
                # We invert and clamp: score = max(0, -rank)
                results = []
                max_rank = 0.0
                for content, target, rank in rows:
                    score = max(0.0, -rank)
                    if score > max_rank:
                        max_rank = score

                for content, target, rank in rows:
                    score = max(0.0, -rank)
                    norm_score = score / max_rank if max_rank > 0 else 0.0
                    results.append({
                        "content": content,
                        "target": target,
                        "score": min(norm_score, 1.0),
                        "bm25_raw": rank,
                    })

                return results

            except Exception as e:
                logger.debug("FTS5 search failed: %s", e)
                return []

    def entry_count(self) -> int:
        """Return the number of indexed entries."""
        with self._lock:
            try:
                self.connect()
                if not self._connected or self._conn is None:
                    return 0
                row = self._conn.execute(
                    "SELECT value FROM fts_meta WHERE key='entry_count'"
                ).fetchone()
                return int(row[0]) if row else 0
            except Exception:
                return 0

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
                self._connected = False

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Clean user query for FTS5 MATCH syntax.

        Removes special FTS5 operators and escapes quotes.
        """
        if not query or not query.strip():
            return ""

        # Remove special FTS5 operators
        import re
        cleaned = re.sub(r'[^\w\s-]', ' ', query)
        # Collapse whitespace
        cleaned = ' '.join(cleaned.split())
        if len(cleaned) < 2:
            return ""

        # Add OR between words for broad match (FTS5 default is AND)
        # Use phrase prefix matching for better recall
        terms = cleaned.split()
        if len(terms) == 1:
            return f'"{terms[0]}"*'  # prefix match for single term
        # Multi-term: OR each term with prefix
        return " OR ".join(f'"{t}"*' for t in terms)


def default_fts_path() -> Path:
    """Return the default FTS5 database path under HERMES_HOME."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "memories" / "memory_fts.db"

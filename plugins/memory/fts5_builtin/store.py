"""
SQLite-backed fact store with FTS5 full-text search.

Stripped from holographic/store.py — no HRR vectors, no entities,
no trust scoring, no memory_banks.  Pure keyword retrieval.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT NOT NULL,
    category   TEXT DEFAULT 'general',
    tags       TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;
"""


class FTS5Store:
    """Minimal FTS5-backed fact store — zero external deps."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.executescript(_SCHEMA)
                conn.commit()
            finally:
                conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_fact(self, content: str, *, category: str = "general",
                 tags: str = "") -> int:
        """Insert a fact. Returns the new fact_id."""
        content = content.strip()
        if not content:
            raise ValueError("fact content must not be empty")
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    "INSERT INTO facts (content, category, tags) VALUES (?, ?, ?)",
                    (content, category, tags),
                )
                conn.commit()
                return cur.lastrowid or 0
            finally:
                conn.close()

    def search_facts(self, query: str, *, category: Optional[str] = None,
                     limit: int = 10) -> list[dict]:
        """FTS5 full-text search.  Returns list of {fact_id, content, category, tags, created_at}."""
        query = query.strip()
        if not query:
            return []
        # Escape FTS5 special chars (simple — only quote the query)
        safe = query.replace('"', '""')
        fts_query = f'"{safe}"'
        with self._lock:
            conn = self._get_conn()
            try:
                sql = """
                    SELECT f.fact_id, f.content, f.category, f.tags, f.created_at
                    FROM facts f
                    JOIN facts_fts ft ON f.fact_id = ft.rowid
                    WHERE facts_fts MATCH ?
                """
                params: list = [fts_query]
                if category:
                    sql += " AND f.category = ?"
                    params.append(category)
                sql += " ORDER BY rank LIMIT ?"
                params.append(limit)
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def list_facts(self, *, category: Optional[str] = None,
                   limit: int = 50) -> list[dict]:
        """List recent facts, optionally filtered by category."""
        with self._lock:
            conn = self._get_conn()
            try:
                if category:
                    rows = conn.execute(
                        "SELECT fact_id, content, category, tags, created_at "
                        "FROM facts WHERE category = ? ORDER BY fact_id DESC LIMIT ?",
                        (category, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT fact_id, content, category, tags, created_at "
                        "FROM facts ORDER BY fact_id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def remove_fact(self, fact_id: int) -> bool:
        """Delete a fact by ID. Returns True if deleted."""
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute(
                    "DELETE FROM facts WHERE fact_id = ?", (fact_id,)
                )
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def clear(self) -> int:
        """Delete all facts. Returns count of deleted rows."""
        with self._lock:
            conn = self._get_conn()
            try:
                cur = conn.execute("DELETE FROM facts")
                conn.commit()
                return cur.rowcount
            finally:
                conn.close()

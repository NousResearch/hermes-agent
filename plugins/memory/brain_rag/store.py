"""SQLite-backed knowledge store for Brain RAG."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chunker import chunk_text
from .embeddings import build_idf, hash_embed, tfidf_embed


class BrainRAGStore:
    """Persistent chunk store with FTS5 and vector columns."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._idf: Dict[str, float] = {}
        self._init_schema()
        self._refresh_idf()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    vector TEXT NOT NULL DEFAULT '[]',
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content,
                    title,
                    source,
                    content='chunks',
                    content_rowid='id',
                    tokenize='porter unicode61'
                );

                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    vector TEXT NOT NULL DEFAULT '[]',
                    importance REAL NOT NULL DEFAULT 0.7,
                    created_at REAL NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    content,
                    category,
                    content='memories',
                    content_rowid='id',
                    tokenize='porter unicode61'
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                """
            )
            self._conn.commit()

    def _refresh_idf(self) -> None:
        with self._lock:
            rows = self._conn.execute("SELECT content FROM chunks").fetchall()
            mem_rows = self._conn.execute("SELECT content FROM memories").fetchall()
            corpus = [r["content"] for r in rows] + [r["content"] for r in mem_rows]
            self._idf = build_idf(corpus)

    def _embed(self, text: str) -> List[float]:
        if self._idf:
            return tfidf_embed(text, self._idf)
        return hash_embed(text)

    def _vector_json(self, text: str) -> str:
        return json.dumps(self._embed(text))

    def ingest_text(
        self,
        content: str,
        *,
        source: str = "manual",
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 800,
        overlap: int = 120,
    ) -> Dict[str, Any]:
        """Chunk and index a document."""
        chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            return {"success": False, "error": "No content to ingest"}

        now = time.time()
        meta_json = json.dumps(metadata or {})
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO documents (source, title, metadata, created_at) VALUES (?, ?, ?, ?)",
                (source, title or source, meta_json, now),
            )
            doc_id = cur.lastrowid
            for i, chunk in enumerate(chunks):
                vec = self._vector_json(chunk)
                cur = self._conn.execute(
                    "INSERT INTO chunks (doc_id, content, vector, chunk_index, created_at) VALUES (?, ?, ?, ?, ?)",
                    (doc_id, chunk, vec, i, now),
                )
                chunk_id = cur.lastrowid
                self._conn.execute(
                    "INSERT INTO chunks_fts (rowid, content, title, source) VALUES (?, ?, ?, ?)",
                    (chunk_id, chunk, title or source, source),
                )
            self._conn.commit()
        self._refresh_idf()
        return {"success": True, "doc_id": doc_id, "chunks": len(chunks), "source": source}

    def ingest_file(self, path: str, *, title: str = "") -> Dict[str, Any]:
        p = Path(path).expanduser()
        if not p.is_file():
            return {"success": False, "error": f"File not found: {path}"}
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return {"success": False, "error": str(e)}
        return self.ingest_text(
            content,
            source=str(p),
            title=title or p.name,
            metadata={"path": str(p)},
        )

    def remember(
        self,
        content: str,
        *,
        category: str = "general",
        importance: float = 0.7,
    ) -> Dict[str, Any]:
        """Store an explicit memory fact."""
        content = (content or "").strip()
        if not content:
            return {"success": False, "error": "Content is required"}
        now = time.time()
        vec = self._vector_json(content)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memories (content, category, vector, importance, created_at) VALUES (?, ?, ?, ?, ?)",
                (content, category, vec, importance, now),
            )
            mem_id = cur.lastrowid
            self._conn.execute(
                "INSERT INTO memories_fts (rowid, content, category) VALUES (?, ?, ?)",
                (mem_id, content, category),
            )
            self._conn.commit()
        self._refresh_idf()
        return {"success": True, "memory_id": mem_id}

    def fts_search_chunks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT c.id, c.content, c.vector, d.title, d.source,
                       bm25(chunks_fts) AS rank
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                JOIN documents d ON d.id = c.doc_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (self._fts_query(q), limit),
            ).fetchall()
        return [self._row_to_hit(r, rank_key="rank") for r in rows]

    def fts_search_memories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT m.id, m.content, m.vector, m.category, m.importance,
                       bm25(memories_fts) AS rank
                FROM memories_fts
                JOIN memories m ON m.id = memories_fts.rowid
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (self._fts_query(q), limit),
            ).fetchall()
        results = []
        for r in rows:
            hit = self._row_to_hit(r, rank_key="rank")
            hit["category"] = r["category"]
            hit["importance"] = r["importance"]
            hit["kind"] = "memory"
            results.append(hit)
        return results

    def all_chunks(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT c.id, c.content, c.vector, d.title, d.source
                FROM chunks c
                JOIN documents d ON d.id = c.doc_id
                ORDER BY c.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_hit(r) for r in rows]

    def all_memories(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, content, vector, category, importance FROM memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            hit = self._row_to_hit(r)
            hit["category"] = r["category"]
            hit["importance"] = r["importance"]
            hit["kind"] = "memory"
            out.append(hit)
        return out

    def stats(self) -> Dict[str, int]:
        with self._lock:
            docs = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            memories = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return {"documents": docs, "chunks": chunks, "memories": memories}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _fts_query(query: str) -> str:
        """Escape FTS5 query tokens."""
        tokens = [t for t in query.split() if t]
        if not tokens:
            return '""'
        return " OR ".join(f'"{t.replace(chr(34), "")}"' for t in tokens)

    @staticmethod
    def _row_to_hit(row: sqlite3.Row, rank_key: Optional[str] = None) -> Dict[str, Any]:
        hit: Dict[str, Any] = {
            "id": row["id"],
            "content": row["content"],
            "vector": json.loads(row["vector"] or "[]"),
            "kind": "chunk",
        }
        if "title" in row.keys():
            hit["title"] = row["title"]
        if "source" in row.keys():
            hit["source"] = row["source"]
        if rank_key and rank_key in row.keys():
            # bm25 returns negative values — lower is better
            hit["bm25"] = float(row[rank_key])
        return hit

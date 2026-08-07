"""LocalProvider — stdlib SQLite vector store implementing KnowledgeProvider.

This is the default backend so the knowledge subsystem works out of the box
with zero external services. It proves the abstraction: swapping it for
AnythingLLMProvider / QdrantProvider changes nothing above the interface.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from ..embeddings import chunk_text, cosine, embed, keyword_overlap
from ..provider import KnowledgeProvider
from ..types import Chunk, Citation, Document, HealthStatus, IndexResult, SearchResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    workspace TEXT NOT NULL DEFAULT 'default',
    title TEXT, path TEXT, source TEXT,
    checksum TEXT, mtime REAL, metadata TEXT, content TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    workspace TEXT NOT NULL DEFAULT 'default',
    ordinal INTEGER, text TEXT, vector TEXT,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_ws ON chunks(workspace);
CREATE INDEX IF NOT EXISTS idx_docs_ws ON documents(workspace);
"""


class LocalProvider(KnowledgeProvider):
    name = "local"

    def __init__(self, db_path: str, default_workspace: str = "default"):
        self.db_path = db_path
        self.default_workspace = default_workspace
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # -- write ---------------------------------------------------------
    def index(self, document: Document) -> IndexResult:
        ws = document.workspace or self.default_workspace
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    "SELECT checksum FROM documents WHERE id=?", (document.id,)
                ).fetchone()
                if row and row["checksum"] == document.checksum:
                    return IndexResult(True, document.id, "skipped", "checksum unchanged")
                action = "updated" if row else "indexed"
                conn.execute("DELETE FROM chunks WHERE document_id=?", (document.id,))
                conn.execute(
                    "INSERT OR REPLACE INTO documents"
                    "(id,workspace,title,path,source,checksum,mtime,metadata,content)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (document.id, ws, document.title, document.path, document.source,
                     document.checksum, document.mtime,
                     json.dumps(document.metadata or {}), document.content),
                )
                for i, piece in enumerate(chunk_text(document.content)):
                    conn.execute(
                        "INSERT INTO chunks(id,document_id,workspace,ordinal,text,vector)"
                        " VALUES (?,?,?,?,?,?)",
                        (f"{document.id}#{i}", document.id, ws, i, piece,
                         json.dumps(embed(f"{document.title}\n{piece}"))),
                    )
                conn.commit()
            return IndexResult(True, document.id, action)
        except Exception as exc:  # pragma: no cover - defensive
            return IndexResult(False, document.id, "error", str(exc))

    def update(self, document: Document) -> IndexResult:
        return self.index(document)

    def delete(self, document_id: str, workspace: Optional[str] = None) -> IndexResult:
        try:
            with self._lock, self._connect() as conn:
                cur = conn.execute("DELETE FROM documents WHERE id=?", (document_id,))
                conn.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
                conn.commit()
            if cur.rowcount == 0:
                return IndexResult(True, document_id, "skipped", "not found")
            return IndexResult(True, document_id, "deleted")
        except Exception as exc:  # pragma: no cover
            return IndexResult(False, document_id, "error", str(exc))

    # -- read ----------------------------------------------------------
    def search(self, query, limit=5, workspace=None, filters=None) -> SearchResult:
        t0 = time.perf_counter()
        ws = workspace or self.default_workspace
        filters = filters or {}
        qvec = embed(query)
        sql = (
            "SELECT c.id, c.text, c.vector, c.document_id, d.title, d.path,"
            " d.source, d.workspace FROM chunks c JOIN documents d"
            " ON d.id=c.document_id WHERE c.workspace=?"
        )
        params: List[Any] = [ws]
        if filters.get("source"):
            sql += " AND d.source=?"
            params.append(filters["source"])
        if filters.get("path_prefix"):
            sql += " AND d.path LIKE ?"
            params.append(f"{filters['path_prefix']}%")
        if filters.get("exclude_document_id"):
            sql += " AND d.id != ?"
            params.append(filters["exclude_document_id"])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        scored: List[Chunk] = []
        for r in rows:
            try:
                vec = json.loads(r["vector"])
            except Exception:
                continue
            score = 0.75 * cosine(qvec, vec) + 0.25 * keyword_overlap(query, r["text"])
            if score <= 0:
                continue
            scored.append(Chunk(
                id=r["id"], text=r["text"], score=score, document_id=r["document_id"],
                citation=Citation(
                    title=r["title"] or os.path.basename(r["path"] or r["document_id"]),
                    file=os.path.basename(r["path"] or ""),
                    path=r["path"] or "",
                    score=round(score, 4),
                    workspace=r["workspace"],
                    chunk_id=r["id"],
                    document_id=r["document_id"],
                    provider=self.name,
                ),
            ))
        scored.sort(key=lambda c: c.score, reverse=True)
        top = scored[: max(1, int(limit))]
        return SearchResult(
            query=query, chunks=top, provider=self.name, workspace=ws,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            confidence=top[0].score if top else 0.0,
        )

    def retrieve(self, query, limit=5, workspace=None, filters=None) -> SearchResult:
        res = self.search(query, limit=limit, workspace=workspace, filters=filters)
        if res.chunks:
            # extractive stitch — no LLM reasoning happens in this module
            res.answer = "\n\n".join(
                f"{c.citation.title if c.citation else c.document_id}: {c.text.strip()}"
                for c in res.chunks
            )
        return res

    def find_similar(self, document_id, limit=5, workspace=None) -> SearchResult:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT title, content, workspace FROM documents WHERE id=?",
                (document_id,),
            ).fetchone()
        if not row:
            return SearchResult(query=document_id, provider=self.name,
                                error=f"document not found: {document_id}")
        return self.search(
            f"{row['title']}\n{row['content'][:2000]}",
            limit=limit,
            workspace=workspace or row["workspace"],
            filters={"exclude_document_id": document_id},
        )

    def list_documents(self, workspace=None) -> List[Dict[str, Any]]:
        ws = workspace or self.default_workspace
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id,title,path,source,checksum,mtime FROM documents WHERE workspace=?",
                (ws,),
            ).fetchall()
        return [dict(r) for r in rows]

    def health(self) -> HealthStatus:
        t0 = time.perf_counter()
        try:
            with self._connect() as conn:
                n = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                m = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            return HealthStatus(True, self.name, f"{n} documents / {m} chunks",
                                (time.perf_counter() - t0) * 1000)
        except Exception as exc:
            return HealthStatus(False, self.name, str(exc),
                                (time.perf_counter() - t0) * 1000)

"""DuckDB-backed local research memory for Hermes search recipes."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urldefrag

from hermes_constants import get_hermes_home


class ResearchSearchUnavailableError(RuntimeError):
    """Raised when the local research-search backend is unavailable."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonicalize_url(url: str) -> str:
    """Return a stable URL key for local indexing/deduplication."""
    clean, _frag = urldefrag(str(url or "").strip())
    if clean.endswith("/") and clean.count("/") > 2:
        clean = clean.rstrip("/")
    return clean


def content_hash(content: str) -> str:
    return hashlib.sha256(str(content or "").encode("utf-8")).hexdigest()


def document_id_for_url(url: str) -> str:
    canonical = canonicalize_url(url)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def chunk_id_for_document(document_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(
        f"{document_id}:{chunk_index}:{content_hash(text)}".encode("utf-8")
    ).hexdigest()
    return digest


def chunk_text(
    text: str,
    chunk_chars: int = 1800,
    overlap_chars: int = 250,
) -> list[str]:
    """Split document text into overlapping chunks for retrieval."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if not clean:
        return []
    chunk_chars = max(int(chunk_chars or 1800), 300)
    overlap_chars = max(min(int(overlap_chars or 0), chunk_chars // 2), 0)
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_chars, len(clean))
        if end < len(clean):
            boundary = clean.rfind(" ", start + chunk_chars // 2, end)
            if boundary > start:
                end = boundary
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = max(end - overlap_chars, start + 1)
    return chunks


def duckdb_available() -> bool:
    try:
        importlib.import_module("duckdb")
        return True
    except Exception:
        return False


def default_db_path() -> Path:
    return get_hermes_home() / "research" / "search.duckdb"


def resolve_db_path(config: dict[str, Any] | None = None) -> Path:
    raw = ((config or {}).get("research_search") or {}).get("db_path")
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser()
    return default_db_path()


class ResearchSearchStore:
    """Small DuckDB wrapper for document, evidence, and run metadata."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path).expanduser() if db_path else default_db_path()
        self._conn = None
        self._fts_loaded: bool | None = None

    def _duckdb(self):
        try:
            return importlib.import_module("duckdb")
        except Exception as exc:
            raise ResearchSearchUnavailableError(
                "DuckDB is not installed. Install Hermes with the "
                "`research-search` extra to enable local research indexing."
            ) from exc

    def connect(self):
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = self._duckdb().connect(str(self.db_path))
            self.ensure_schema()
        return self._conn

    def ensure_schema(self) -> None:
        conn = self._conn if self._conn is not None else self.connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR PRIMARY KEY,
                url VARCHAR,
                canonical_url VARCHAR,
                title VARCHAR,
                content VARCHAR,
                content_hash VARCHAR,
                vertical VARCHAR,
                source_type VARCHAR,
                fetched_at VARCHAR,
                status VARCHAR,
                error VARCHAR,
                metadata_json VARCHAR
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence (
                id VARCHAR PRIMARY KEY,
                document_id VARCHAR,
                chunk_id VARCHAR,
                query VARCHAR,
                claim VARCHAR,
                excerpt VARCHAR,
                relevance_score DOUBLE,
                source_quality_score DOUBLE,
                confidence DOUBLE,
                created_at VARCHAR,
                metadata_json VARCHAR
            )
            """
        )
        self._ensure_column(conn, "evidence", "chunk_id", "VARCHAR")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id VARCHAR PRIMARY KEY,
                document_id VARCHAR,
                chunk_index INTEGER,
                text VARCHAR,
                token_count INTEGER,
                created_at VARCHAR,
                metadata_json VARCHAR
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_embeddings (
                chunk_id VARCHAR,
                provider VARCHAR,
                model VARCHAR,
                dim INTEGER,
                vector BLOB,
                created_at VARCHAR,
                PRIMARY KEY (chunk_id, provider, model)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS research_runs (
                id VARCHAR PRIMARY KEY,
                question VARCHAR,
                topic_type VARCHAR,
                freshness VARCHAR,
                depth VARCHAR,
                plan_json VARCHAR,
                gaps_json VARCHAR,
                created_at VARCHAR
            )
            """
        )
        self._load_fts(conn)

    def _ensure_column(self, conn, table: str, column: str, column_type: str) -> None:
        try:
            rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
            existing = {str(row[1]) for row in rows}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
        except Exception:
            pass

    def _load_fts(self, conn) -> bool:
        if self._fts_loaded is not None:
            return self._fts_loaded
        try:
            conn.execute("INSTALL fts")
            conn.execute("LOAD fts")
            self._fts_loaded = True
        except Exception:
            self._fts_loaded = False
        return bool(self._fts_loaded)

    @property
    def fts_available(self) -> bool:
        try:
            self.connect()
        except ResearchSearchUnavailableError:
            return False
        return bool(self._fts_loaded)

    def rebuild_fts_index(self) -> None:
        conn = self.connect()
        if not self._load_fts(conn):
            raise ResearchSearchUnavailableError(
                "DuckDB FTS extension is unavailable. Local research search "
                "cannot run until the fts extension can be installed/loaded."
            )
        doc_error: Exception | None = None
        chunk_error: Exception | None = None
        try:
            conn.execute(
                """
                PRAGMA create_fts_index(
                    'documents', 'id', 'title', 'content', overwrite=1
                )
                """
            )
        except Exception as exc:
            doc_error = exc
        try:
            chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            if chunk_count:
                conn.execute(
                    """
                    PRAGMA create_fts_index(
                        'chunks', 'id', 'text', overwrite=1
                    )
                    """
                )
        except Exception as exc:
            chunk_error = exc
        if doc_error is not None:
            raise ResearchSearchUnavailableError(
                f"DuckDB FTS index creation failed: {doc_error}"
            ) from doc_error
        if chunk_error is not None:
            # Chunk FTS is secondary; document FTS can still serve older stores.
            return

    def _delete_document_children(self, conn, document_id: str) -> None:
        try:
            chunk_ids = [
                str(row[0])
                for row in conn.execute(
                    "SELECT id FROM chunks WHERE document_id = ?", [document_id]
                ).fetchall()
            ]
        except Exception:
            chunk_ids = []
        for chunk_id in chunk_ids:
            conn.execute("DELETE FROM chunk_embeddings WHERE chunk_id = ?", [chunk_id])
        conn.execute("DELETE FROM chunks WHERE document_id = ?", [document_id])
        conn.execute("DELETE FROM evidence WHERE document_id = ?", [document_id])

    def upsert_document(self, document: dict[str, Any]) -> dict[str, Any]:
        conn = self.connect()
        url = str(document.get("url") or "")
        canonical = canonicalize_url(str(document.get("canonical_url") or url))
        doc_id = str(document.get("id") or document_id_for_url(canonical))
        content = str(document.get("content") or "")
        row = {
            "id": doc_id,
            "url": url,
            "canonical_url": canonical,
            "title": str(document.get("title") or ""),
            "content": content,
            "content_hash": str(document.get("content_hash") or content_hash(content)),
            "vertical": str(document.get("vertical") or "web"),
            "source_type": str(document.get("source_type") or "unknown"),
            "fetched_at": str(document.get("fetched_at") or utc_now_iso()),
            "status": str(document.get("status") or "extracted"),
            "error": str(document.get("error") or ""),
            "metadata_json": json.dumps(
                document.get("metadata") or {}, ensure_ascii=False
            ),
        }
        old_rows = conn.execute(
            "SELECT id FROM documents WHERE canonical_url = ?", [canonical]
        ).fetchall()
        for old_row in old_rows:
            self._delete_document_children(conn, str(old_row[0]))
        conn.execute("DELETE FROM documents WHERE canonical_url = ?", [canonical])
        conn.execute(
            """
            INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["id"],
                row["url"],
                row["canonical_url"],
                row["title"],
                row["content"],
                row["content_hash"],
                row["vertical"],
                row["source_type"],
                row["fetched_at"],
                row["status"],
                row["error"],
                row["metadata_json"],
            ],
        )
        if self._fts_loaded:
            try:
                self.rebuild_fts_index()
            except ResearchSearchUnavailableError:
                pass
        return row

    def upsert_chunks(
        self,
        document_id: str,
        content: str,
        chunk_chars: int = 1800,
        overlap_chars: int = 250,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        conn = self.connect()
        document_id = str(document_id or "")
        if not document_id:
            return []
        self._delete_document_children(conn, document_id)
        rows: list[dict[str, Any]] = []
        created_at = utc_now_iso()
        for idx, text in enumerate(chunk_text(content, chunk_chars, overlap_chars)):
            chunk_id = chunk_id_for_document(document_id, idx, text)
            row = {
                "id": chunk_id,
                "document_id": document_id,
                "chunk_index": idx,
                "text": text,
                "token_count": max(1, len(text) // 4),
                "created_at": created_at,
                "metadata_json": json.dumps(metadata or {}, ensure_ascii=False),
            }
            conn.execute(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    row["id"],
                    row["document_id"],
                    row["chunk_index"],
                    row["text"],
                    row["token_count"],
                    row["created_at"],
                    row["metadata_json"],
                ],
            )
            rows.append(row)
        if self._fts_loaded:
            try:
                self.rebuild_fts_index()
            except ResearchSearchUnavailableError:
                pass
        return rows

    def upsert_evidence(self, evidence: dict[str, Any]) -> dict[str, Any]:
        conn = self.connect()
        document_id = str(evidence.get("document_id") or "")
        query = str(evidence.get("query") or "")
        excerpt = str(evidence.get("excerpt") or "")
        evidence_id = str(
            evidence.get("id")
            or hashlib.sha256(
                f"{document_id}:{query}:{content_hash(excerpt)}".encode("utf-8")
            ).hexdigest()
        )
        row = {
            "id": evidence_id,
            "document_id": document_id,
            "chunk_id": str(evidence.get("chunk_id") or ""),
            "query": query,
            "claim": str(evidence.get("claim") or ""),
            "excerpt": excerpt,
            "relevance_score": float(evidence.get("relevance_score") or 0.0),
            "source_quality_score": float(evidence.get("source_quality_score") or 0.0),
            "confidence": float(evidence.get("confidence") or 0.0),
            "created_at": str(evidence.get("created_at") or utc_now_iso()),
            "metadata_json": json.dumps(evidence.get("metadata") or {}, ensure_ascii=False),
        }
        conn.execute("DELETE FROM evidence WHERE id = ?", [row["id"]])
        conn.execute(
            """
            INSERT INTO evidence (
                id, document_id, chunk_id, query, claim, excerpt,
                relevance_score, source_quality_score, confidence,
                created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["id"],
                row["document_id"],
                row["chunk_id"],
                row["query"],
                row["claim"],
                row["excerpt"],
                row["relevance_score"],
                row["source_quality_score"],
                row["confidence"],
                row["created_at"],
                row["metadata_json"],
            ],
        )
        return row

    def upsert_embedding(
        self,
        chunk_id: str,
        provider: str,
        model: str,
        vector_blob: bytes,
        dim: int,
    ) -> None:
        conn = self.connect()
        conn.execute(
            "DELETE FROM chunk_embeddings WHERE chunk_id = ? AND provider = ? AND model = ?",
            [chunk_id, provider, model],
        )
        conn.execute(
            "INSERT INTO chunk_embeddings VALUES (?, ?, ?, ?, ?, ?)",
            [chunk_id, provider, model, int(dim), vector_blob, utc_now_iso()],
        )

    def get_chunk_embeddings(
        self,
        chunk_ids: list[str],
        provider: str,
        model: str,
    ) -> dict[str, dict[str, Any]]:
        conn = self.connect()
        result: dict[str, dict[str, Any]] = {}
        for chunk_id in chunk_ids:
            row = conn.execute(
                """
                SELECT chunk_id, dim, vector
                FROM chunk_embeddings
                WHERE chunk_id = ? AND provider = ? AND model = ?
                """,
                [chunk_id, provider, model],
            ).fetchone()
            if row:
                result[str(row[0])] = {"dim": int(row[1]), "vector": bytes(row[2])}
        return result

    def record_run(self, run: dict[str, Any]) -> None:
        conn = self.connect()
        run_id = str(
            run.get("id")
            or hashlib.sha256(
                f"{run.get('question')}:{utc_now_iso()}".encode("utf-8")
            ).hexdigest()
        )
        conn.execute(
            "INSERT INTO research_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                run_id,
                str(run.get("question") or ""),
                str(run.get("topic_type") or "auto"),
                str(run.get("freshness") or "auto"),
                str(run.get("depth") or "auto"),
                json.dumps(run.get("plan") or {}, ensure_ascii=False),
                json.dumps(run.get("gaps") or [], ensure_ascii=False),
                str(run.get("created_at") or utc_now_iso()),
            ],
        )

    def search(self, query: str, limit: int = 10, vertical: str = "auto") -> list[dict]:
        conn = self.connect()
        if not self._load_fts(conn):
            raise ResearchSearchUnavailableError(
                "DuckDB FTS extension is unavailable. Use research_status() "
                "for details or install Hermes with the research-search extra."
            )
        try:
            self.rebuild_fts_index()
            rows = conn.execute(
                """
                SELECT
                    id, url, title, content, vertical, source_type, fetched_at,
                    score
                FROM (
                    SELECT
                        id, url, title, content, vertical, source_type,
                        fetched_at, fts_main_documents.match_bm25(id, ?) AS score
                    FROM documents
                    WHERE (? = 'auto' OR vertical = ?)
                ) sq
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT ?
                """,
                [query, vertical, vertical, int(limit)],
            ).fetchall()
        except ResearchSearchUnavailableError:
            raise
        except Exception as exc:
            raise ResearchSearchUnavailableError(
                f"DuckDB FTS query failed: {exc}"
            ) from exc

        results: list[dict] = []
        for row in rows:
            content = row[3] or ""
            results.append(
                {
                    "id": row[0],
                    "url": row[1],
                    "title": row[2],
                    "excerpt": content[:1200],
                    "vertical": row[4],
                    "source_type": row[5],
                    "fetched_at": row[6],
                    "score": float(row[7] or 0.0),
                }
            )
        return results

    def search_chunks(
        self,
        query: str,
        limit: int = 10,
        vertical: str = "auto",
    ) -> list[dict[str, Any]]:
        conn = self.connect()
        if not self._load_fts(conn):
            raise ResearchSearchUnavailableError(
                "DuckDB FTS extension is unavailable. Use research_status() "
                "for details or install Hermes with the research-search extra."
            )
        try:
            self.rebuild_fts_index()
            rows = conn.execute(
                """
                SELECT
                    c.id, c.document_id, d.url, d.title, c.text, d.vertical,
                    d.source_type, d.fetched_at, score
                FROM (
                    SELECT
                        id, document_id, text,
                        fts_main_chunks.match_bm25(id, ?) AS score
                    FROM chunks
                ) c
                JOIN documents d ON d.id = c.document_id
                WHERE score IS NOT NULL AND (? = 'auto' OR d.vertical = ?)
                ORDER BY score DESC
                LIMIT ?
                """,
                [query, vertical, vertical, int(limit)],
            ).fetchall()
        except ResearchSearchUnavailableError:
            raise
        except Exception as exc:
            raise ResearchSearchUnavailableError(
                f"DuckDB chunk FTS query failed: {exc}"
            ) from exc

        return [
            {
                "chunk_id": row[0],
                "id": row[1],
                "document_id": row[1],
                "url": row[2],
                "title": row[3],
                "excerpt": str(row[4] or "")[:1200],
                "content": row[4],
                "vertical": row[5],
                "source_type": row[6],
                "fetched_at": row[7],
                "score": float(row[8] or 0.0),
            }
            for row in rows
        ]

    def status(self) -> dict[str, Any]:
        available = duckdb_available()
        status: dict[str, Any] = {
            "success": True,
            "duckdb_available": available,
            "db_path": str(self.db_path),
            "fts_available": False,
            "documents": 0,
            "chunks": 0,
            "evidence": 0,
            "embeddings": 0,
        }
        if not available:
            status["error"] = (
                "DuckDB is not installed. Install the research-search extra "
                "to enable local indexing."
            )
            return status
        try:
            conn = self.connect()
            status["fts_available"] = bool(self._fts_loaded)
            status["documents"] = int(
                conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            )
            status["chunks"] = int(
                conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            )
            status["evidence"] = int(
                conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
            )
            status["embeddings"] = int(
                conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
            )
        except Exception as exc:
            status["success"] = False
            status["error"] = str(exc)
        return status

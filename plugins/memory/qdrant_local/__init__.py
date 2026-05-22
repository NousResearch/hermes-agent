"""Local hybrid memory provider for Hermes Agent.

Design invariants:
- SQLite registry/FTS is the canonical local ledger.
- Qdrant local mode is a rebuildable semantic projection, never the source of truth.
- No automatic prompt injection happens in shadow mode.
- Tool schemas stay empty until explicit operator-facing tools are enabled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_MODE = "shadow"
DEFAULT_COLLECTION = "hermes_qdrant_local"
DEFAULT_EMBEDDING_DIMENSIONS = 384
CHUNKER_VERSION = "qdrant-local-v1"
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b[A-Za-z0-9_-]*(?:token|secret|password|credential)[A-Za-z0-9_-]*\s*[:=]\s*\S+", re.I),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)

REGISTRY_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    target TEXT,
    session_id TEXT,
    parent_session_id TEXT,
    lineage_root TEXT,
    platform TEXT,
    profile TEXT,
    user_scope TEXT NOT NULL,
    trust_level TEXT NOT NULL,
    title TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    deleted_at REAL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    vector_point_id TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    role TEXT,
    ordinal INTEGER,
    token_count INTEGER,
    importance REAL DEFAULT 0.5,
    embedding_spec_hash TEXT,
    chunker_version TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    deleted_at REAL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    chunk_id UNINDEXED,
    doc_id UNINDEXED
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks
WHEN new.deleted_at IS NULL
BEGIN
    INSERT INTO chunks_fts(rowid, content, chunk_id, doc_id)
    VALUES (new.rowid, new.content, new.chunk_id, new.doc_id);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, chunk_id, doc_id)
    VALUES ('delete', old.rowid, old.content, old.chunk_id, old.doc_id);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, chunk_id, doc_id)
    VALUES ('delete', old.rowid, old.content, old.chunk_id, old.doc_id);
    INSERT INTO chunks_fts(rowid, content, chunk_id, doc_id)
    SELECT new.rowid, new.content, new.chunk_id, new.doc_id
    WHERE new.deleted_at IS NULL;
END;

CREATE TABLE IF NOT EXISTS ingest_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    last_error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    actor_scope TEXT,
    source_scope TEXT,
    result TEXT NOT NULL,
    metadata_json TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id);
CREATE INDEX IF NOT EXISTS idx_documents_scope ON documents(user_scope, trust_level);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
"""


class QdrantLocalMemoryProvider(MemoryProvider):
    """Profile-scoped local hybrid memory provider."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = dict(config or {})
        self._mode = str(self._config.get("mode") or DEFAULT_MODE)
        self._session_id = ""
        self._platform = ""
        self._agent_context = "primary"
        self._agent_identity = ""
        self._provider_dir: Optional[Path] = None
        self._registry_path: Optional[Path] = None
        self._qdrant_path: Optional[Path] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._qdrant_client = None
        self._collection = str(self._config.get("collection") or DEFAULT_COLLECTION)
        self._embedding_dimensions = int(self._config.get("embedding_dimensions") or DEFAULT_EMBEDDING_DIMENSIONS)
        self._lock = threading.RLock()
        self._last_prefetch: Dict[str, str] = {}

    @property
    def name(self) -> str:
        return "qdrant_local"

    def is_available(self) -> bool:
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "mode", "description": "Local vector memory mode", "default": DEFAULT_MODE, "choices": ["off", "shadow", "fts_only", "vector"]},
            {"key": "storage_path", "description": "Provider storage path relative to HERMES_HOME", "default": "memory/qdrant_local"},
            {"key": "embedding_dimensions", "description": "Deterministic local hash embedding dimensions", "default": str(DEFAULT_EMBEDDING_DIMENSIONS)},
            {"key": "collection", "description": "Qdrant collection name", "default": DEFAULT_COLLECTION},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        provider_dir = self._resolve_provider_dir(hermes_home, values)
        provider_dir.mkdir(parents=True, exist_ok=True)
        self._write_json_config(provider_dir, values)

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home")
        if not hermes_home:
            from hermes_constants import get_hermes_home
            hermes_home = str(get_hermes_home())

        self._session_id = session_id or ""
        self._platform = str(kwargs.get("platform") or "")
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        self._agent_identity = str(kwargs.get("agent_identity") or "")

        self._provider_dir = self._resolve_provider_dir(str(hermes_home), self._config)
        self._provider_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._provider_dir / "registry.sqlite"
        self._qdrant_path = self._provider_dir / "qdrant"
        self._write_json_config(self._provider_dir, self._effective_config())
        self._conn = sqlite3.connect(str(self._registry_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize_sqlite(self._conn)
        self._audit("initialize", result="ok", metadata={"mode": self._mode, "platform": self._platform, "agent_context": self._agent_context, "session_id_present": bool(self._session_id)})

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        cache_key = session_id or self._session_id or "default"
        return self._last_prefetch.get(cache_key, "")

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        cache_key = session_id or self._session_id or "default"
        if self._mode not in {"fts_only", "vector"}:
            self._last_prefetch[cache_key] = ""
            return
        if self._mode == "vector":
            self._last_prefetch[cache_key] = self._search_vector(query) or self._search_fts(query)
            return
        self._last_prefetch[cache_key] = self._search_fts(query)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if self._mode == "off" or self._agent_context != "primary":
            return
        active_session = session_id or self._session_id
        self._index_turn(user_content, assistant_content, session_id=active_session)
        self._enqueue_event("turn", {"session_id": active_session, "has_user_content": bool(user_content), "has_assistant_content": bool(assistant_content)})

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if not self._config.get("enable_tools"):
            return []
        return [
            {
                "name": "vector_memory_status",
                "description": "Return local vector memory provider status without exposing stored memory content.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "vector_memory_rebuild",
                "description": "Rebuild the disposable Qdrant local vector projection from the SQLite ledger.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "vector_memory_search",
                "description": "Search local private memory with strict hydration from SQLite and bounded results.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        try:
            if tool_name == "vector_memory_status":
                return json.dumps({"success": True, **self.vector_status()}, ensure_ascii=False, sort_keys=True)
            if tool_name == "vector_memory_rebuild":
                return json.dumps(self.rebuild_vector_index(), ensure_ascii=False, sort_keys=True)
            if tool_name == "vector_memory_search":
                query = str((args or {}).get("query") or "")
                limit = min(10, max(1, int((args or {}).get("limit") or 5)))
                return json.dumps(self.search_memory(query, limit=limit), ensure_ascii=False, sort_keys=True)
            return json.dumps({"success": False, "error": f"unknown qdrant_local tool: {tool_name}"})
        except Exception as exc:
            logger.exception("qdrant_local tool call failed: %s", tool_name)
            return json.dumps({"success": False, "error": str(exc)})

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, **kwargs) -> None:
        old_session_id = self._session_id
        self._session_id = new_session_id or ""
        if reset:
            self._last_prefetch.clear()
        self._audit("session_switch", result="ok", metadata={"old_session_id_present": bool(old_session_id), "new_session_id_present": bool(self._session_id), "parent_session_id_present": bool(parent_session_id), "reset": bool(reset)})

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        if self._mode == "off" or self._agent_context != "primary":
            return ""
        self._enqueue_event("pre_compress", {"message_count": len(messages or [])})
        return ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._mode == "off" or self._agent_context != "primary":
            return
        self._enqueue_event("session_end", {"message_count": len(messages or [])})

    def on_memory_write(self, action: str, target: str, content: str, metadata=None) -> None:
        if self._mode == "off" or self._agent_context != "primary":
            return
        self._enqueue_event("memory_write", {"action": action, "target": target, "content_present": bool(content), "metadata_keys": sorted((metadata or {}).keys())})

    def shutdown(self) -> None:
        with self._lock:
            if self._qdrant_client is not None:
                try:
                    close = getattr(self._qdrant_client, "close", None)
                    if close:
                        close()
                finally:
                    self._qdrant_client = None
            if self._conn is not None:
                try:
                    self._conn.commit()
                    self._conn.close()
                finally:
                    self._conn = None

    def embed_text(self, text: str) -> List[float]:
        """Return a deterministic, local, normalized hash embedding."""
        dims = max(8, self._embedding_dimensions)
        vector = [0.0] * dims
        tokens = re.findall(r"[\w-]+", text.lower(), flags=re.UNICODE)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            vector[idx] += -1.0 if digest[4] & 1 else 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def rebuild_vector_index(self) -> Dict[str, Any]:
        if self._conn is None:
            return {"success": False, "error": "provider_not_initialized", "indexed_chunks": 0}
        client = self._ensure_qdrant_collection(recreate=True)
        if client is None:
            return {"success": False, "error": "qdrant_client_unavailable", "indexed_chunks": 0}
        try:
            from qdrant_client.models import PointStruct  # type: ignore[import-not-found]
        except Exception as exc:
            return {"success": False, "error": f"qdrant_models_unavailable: {exc}", "indexed_chunks": 0}

        rows = self._all_active_chunks()
        points = [
            PointStruct(
                id=str(row["vector_point_id"]),
                vector=self.embed_text(str(row["content"])),
                payload={
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "source": row["source"],
                    "user_scope": row["user_scope"],
                    "trust_level": row["trust_level"],
                    "role": row["role"],
                    "session_id": row["session_id"],
                    "content_hash": row["content_hash"],
                },
            )
            for row in rows
        ]
        if points:
            client.upsert(collection_name=self._collection, points=points, wait=True)
        self._audit("vector_rebuild", result="ok", metadata={"indexed_chunks": len(points), "collection": self._collection})
        return {"success": True, "collection": self._collection, "indexed_chunks": len(points), "vector_size": self._embedding_dimensions}

    def vector_status(self) -> Dict[str, Any]:
        available = self._qdrant_import_available()
        count = 0
        if available:
            client = self._ensure_qdrant_collection(recreate=False)
            if client is not None:
                try:
                    count = int(client.count(collection_name=self._collection, exact=True).count)
                except Exception:
                    count = 0
        return {"qdrant_available": available, "collection": self._collection, "vector_size": self._embedding_dimensions, "points_count": count, "path": str(self._qdrant_path) if self._qdrant_path else ""}

    def search_memory(self, query: str, *, limit: int = 5) -> Dict[str, Any]:
        if not query.strip():
            return {"success": False, "error": "query_required", "backend": "none", "results": []}
        rows: List[sqlite3.Row] = []
        backend = "fts"
        if self._mode == "vector":
            rows = self._vector_rows(query, limit=limit)
            if rows:
                backend = "qdrant"
        if not rows:
            rows = self._fts_rows(query, limit=limit) or self._like_rows(query, limit=limit)
            backend = "fts"
        return {
            "success": True,
            "backend": backend,
            "scope": "operator_private",
            "results": [
                {"chunk_id": row["chunk_id"], "doc_id": row["doc_id"], "content": self._snippet(str(row["content"]))}
                for row in rows[:limit]
            ],
        }

    def _effective_config(self) -> Dict[str, Any]:
        cfg = {"schema_version": SCHEMA_VERSION, "mode": self._mode, "storage_path": self._config.get("storage_path", "memory/qdrant_local"), "collection": self._collection, "embedding_dimensions": self._embedding_dimensions}
        for key, value in self._config.items():
            if any(marker in key.lower() for marker in ("key", "token", "secret", "password", "credential")):
                continue
            cfg.setdefault(key, value)
        return cfg

    @staticmethod
    def _resolve_provider_dir(hermes_home: str, config: Dict[str, Any]) -> Path:
        raw = str(config.get("storage_path") or "memory/qdrant_local")
        home = Path(hermes_home).expanduser().resolve()
        raw = raw.replace("$HERMES_HOME", str(home)).replace("${HERMES_HOME}", str(home))
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = home / path
        return path

    @staticmethod
    def _write_json_config(provider_dir: Path, config: Dict[str, Any]) -> None:
        payload = dict(config)
        payload.setdefault("schema_version", SCHEMA_VERSION)
        payload.setdefault("mode", DEFAULT_MODE)
        tmp = provider_dir / "qdrant_local.json.tmp"
        final = provider_dir / "qdrant_local.json"
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(final)

    def _initialize_sqlite(self, conn: sqlite3.Connection) -> None:
        with self._lock:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.OperationalError as exc:
                logger.warning("qdrant_local registry WAL unavailable: %s", exc)
                conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(REGISTRY_SQL)
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", ("schema_version", str(SCHEMA_VERSION)))
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", ("mode", self._mode))
            conn.commit()

    def _enqueue_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self._conn is None:
            return
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO ingest_events(event_type, payload_json, status, created_at, updated_at)
                VALUES (?, ?, 'pending', ?, ?)
                """,
                (event_type, json.dumps(dict(payload), ensure_ascii=False, sort_keys=True), now, now),
            )
            self._conn.commit()

    def _index_turn(self, user_content: str, assistant_content: str, *, session_id: str) -> None:
        if self._conn is None:
            return
        pieces = [("user", user_content or ""), ("assistant", assistant_content or "")]
        if any(self._looks_secret_like(content) for _, content in pieces if content):
            self._audit("ingest_blocked", result="secret_like_content", metadata={"source": "session", "session_id_present": bool(session_id)})
            return

        normalized = "\n".join(f"{role}: {content.strip()}" for role, content in pieces if content.strip())
        if not normalized:
            return

        now = time.time()
        doc_hash = self._sha256(f"session\0{session_id}\0{normalized}")
        doc_id = f"session:{doc_hash[:32]}"
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO documents(
                    doc_id, source, target, session_id, parent_session_id, lineage_root,
                    platform, profile, user_scope, trust_level, title, created_at, updated_at
                ) VALUES (?, 'session', NULL, ?, NULL, ?, ?, NULL, 'operator_private', 'trusted', ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (doc_id, session_id, session_id, self._platform, f"session turn {session_id or 'unknown'}", now, now),
            )
            for ordinal, (role, content) in enumerate(pieces):
                content = content.strip()
                if not content:
                    continue
                content_hash = self._sha256(content)
                chunk_id = f"{doc_id}:{role}:{ordinal}:{content_hash[:16]}"
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO chunks(
                        chunk_id, doc_id, vector_point_id, content, content_hash, role,
                        ordinal, token_count, importance, embedding_spec_hash,
                        chunker_version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.5, NULL, ?, ?, ?)
                    """,
                    (chunk_id, doc_id, self._point_id(chunk_id), content, content_hash, role, ordinal, len(content.split()), CHUNKER_VERSION, now, now),
                )
            self._conn.commit()

    def _search_fts(self, query: str, *, limit: int = 5) -> str:
        if self._conn is None or not query.strip():
            return ""
        rows = self._fts_rows(query, limit=limit) or self._like_rows(query, limit=limit)
        if not rows:
            return ""
        return self._format_recall("SQLite FTS", rows)

    def _search_vector(self, query: str, *, limit: int = 5) -> str:
        rows = self._vector_rows(query, limit=limit)
        return self._format_recall("Qdrant local", rows) if rows else ""

    def _vector_rows(self, query: str, *, limit: int = 5) -> List[sqlite3.Row]:
        if self._conn is None or not query.strip():
            return []
        client = self._ensure_qdrant_collection(recreate=False)
        if client is None:
            return []
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore[import-not-found]
            response = client.query_points(
                collection_name=self._collection,
                query=self.embed_text(query),
                query_filter=Filter(must=[FieldCondition(key="user_scope", match=MatchValue(value="operator_private"))]),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            chunk_ids = [point.payload.get("chunk_id") for point in response.points if point.payload and point.payload.get("chunk_id")]
        except Exception as exc:
            logger.debug("qdrant_local vector query failed: %s", exc)
            return []
        return self._hydrate_chunks(chunk_ids)

    def _format_recall(self, backend: str, rows: List[sqlite3.Row]) -> str:
        lines = [f"Local private memory recall ({backend}, scope=operator_private):"]
        for idx, row in enumerate(rows, start=1):
            lines.append(f"{idx}. [source=session doc={row['doc_id']}] {self._snippet(str(row['content']))}")
        return "\n".join(lines)

    def _fts_rows(self, query: str, *, limit: int) -> List[sqlite3.Row]:
        match_query = self._to_fts_query(query)
        if not match_query:
            return []
        try:
            with self._lock:
                return list(self._conn.execute(  # type: ignore[union-attr]
                    """
                    SELECT c.content, c.chunk_id, c.doc_id
                    FROM chunks_fts f
                    JOIN chunks c ON c.chunk_id = f.chunk_id
                    JOIN documents d ON d.doc_id = c.doc_id
                    WHERE chunks_fts MATCH ?
                      AND c.deleted_at IS NULL
                      AND d.deleted_at IS NULL
                      AND d.user_scope = 'operator_private'
                    ORDER BY bm25(chunks_fts)
                    LIMIT ?
                    """,
                    (match_query, limit),
                ))
        except sqlite3.OperationalError as exc:
            logger.debug("qdrant_local FTS query failed: %s", exc)
            return []

    def _like_rows(self, query: str, *, limit: int) -> List[sqlite3.Row]:
        tokens = [t.lower() for t in re.findall(r"[\w-]+", query, flags=re.UNICODE) if len(t) >= 2]
        if not tokens:
            return []
        clauses = " AND ".join("lower(c.content) LIKE ?" for _ in tokens)
        params = [f"%{token}%" for token in tokens]
        with self._lock:
            return list(self._conn.execute(  # type: ignore[union-attr]
                f"""
                SELECT c.content, c.chunk_id, c.doc_id
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.deleted_at IS NULL
                  AND d.deleted_at IS NULL
                  AND d.user_scope = 'operator_private'
                  AND {clauses}
                ORDER BY c.updated_at DESC
                LIMIT ?
                """,
                (*params, limit),
            ))

    def _all_active_chunks(self) -> List[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(  # type: ignore[union-attr]
                """
                SELECT c.chunk_id, c.doc_id, c.vector_point_id, c.content, c.content_hash, c.role,
                       d.source, d.user_scope, d.trust_level, d.session_id
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.deleted_at IS NULL
                  AND d.deleted_at IS NULL
                  AND d.user_scope = 'operator_private'
                ORDER BY d.updated_at DESC, c.ordinal ASC
                """
            ))

    def _hydrate_chunks(self, chunk_ids: List[str]) -> List[sqlite3.Row]:
        if not chunk_ids or self._conn is None:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._lock:
            rows = list(self._conn.execute(  # type: ignore[union-attr]
                f"""
                SELECT c.content, c.chunk_id, c.doc_id
                FROM chunks c
                JOIN documents d ON d.doc_id = c.doc_id
                WHERE c.chunk_id IN ({placeholders})
                  AND c.deleted_at IS NULL
                  AND d.deleted_at IS NULL
                  AND d.user_scope = 'operator_private'
                """,
                chunk_ids,
            ))
        by_id = {row["chunk_id"]: row for row in rows}
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    def _ensure_qdrant_collection(self, *, recreate: bool):
        if self._qdrant_path is None:
            return None
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-not-found]
            from qdrant_client.models import Distance, VectorParams  # type: ignore[import-not-found]
        except Exception as exc:
            logger.debug("qdrant-client unavailable: %s", exc)
            return None
        if self._qdrant_client is None:
            self._qdrant_path.mkdir(parents=True, exist_ok=True)
            self._qdrant_client = QdrantClient(path=str(self._qdrant_path))
        client = self._qdrant_client
        try:
            exists = client.collection_exists(self._collection)
            if recreate and exists:
                client.delete_collection(self._collection)
                exists = False
            if not exists:
                client.create_collection(collection_name=self._collection, vectors_config=VectorParams(size=self._embedding_dimensions, distance=Distance.COSINE))
            return client
        except Exception as exc:
            logger.debug("qdrant_local collection init failed: %s", exc)
            return None

    def _audit(self, event_type: str, *, result: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO audit_events(event_type, actor_scope, source_scope, result, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_type, self._agent_identity or None, self._platform or None, result, json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True), time.time()),
            )
            self._conn.commit()

    @staticmethod
    def _qdrant_import_available() -> bool:
        try:
            import qdrant_client  # type: ignore[import-not-found]  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def _to_fts_query(query: str) -> str:
        tokens = re.findall(r"[\w-]+", query, flags=re.UNICODE)
        tokens = [token.replace('"', '""') for token in tokens if len(token) >= 2]
        return " AND ".join(f'"{token}"' for token in tokens)

    @staticmethod
    def _snippet(content: str, *, limit: int = 240) -> str:
        content = " ".join(content.split())
        if len(content) <= limit:
            return content
        return content[: limit - 1].rstrip() + "…"

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hermes:qdrant_local:{chunk_id}"))

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _looks_secret_like(content: str) -> bool:
        return any(pattern.search(content) for pattern in SECRET_PATTERNS)


def register(ctx) -> None:
    ctx.register_memory_provider(QdrantLocalMemoryProvider())

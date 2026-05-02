"""SQLite semantic search helpers for the Hermes document MCP tool."""

from __future__ import annotations

import math
import sqlite3
import threading
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    DEFAULT_LIMIT,
    EMBED_MODEL,
    LOGGER,
    MAX_LIMIT,
    SQLITE_DB_PATH,
    TABLE_NAME,
    ensure_directories,
)


MODEL_LOCK = threading.Lock()
MODEL: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazily load the embedding model used by both indexing and querying."""
    global MODEL
    if MODEL is None:
        with MODEL_LOCK:
            if MODEL is None:
                LOGGER.info("Loading embedding model %s", EMBED_MODEL)
                MODEL = SentenceTransformer(EMBED_MODEL, device="cpu")
    return MODEL


def clamp_limit(limit: int | None) -> int:
    """Clamp user-provided result limits to the configured range."""
    if limit is None:
        return DEFAULT_LIMIT
    return max(1, min(int(limit), MAX_LIMIT))


def table_exists(conn: sqlite3.Connection) -> bool:
    """Return whether the CocoIndex target table exists."""
    row = conn.execute(
        "select name from sqlite_master where type = 'table' and name = ?",
        (TABLE_NAME,),
    ).fetchone()
    return row is not None


def embed_query(query: str) -> np.ndarray:
    """Embed a query into a normalized float32 vector."""
    vector = get_model().encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return np.asarray(vector, dtype=np.float32)


def decode_embedding(blob: bytes) -> np.ndarray:
    """Decode sqlite-vec float32 blobs written by CocoIndex."""
    return np.frombuffer(blob, dtype=np.float32)


def cosine_score(query_vector: np.ndarray, document_vector: np.ndarray) -> float:
    """Compute cosine similarity, guarding against malformed vectors."""
    if query_vector.shape != document_vector.shape:
        return float("-inf")
    denom = float(np.linalg.norm(query_vector) * np.linalg.norm(document_vector))
    if math.isclose(denom, 0.0):
        return float("-inf")
    return float(np.dot(query_vector, document_vector) / denom)


def fetch_candidate_rows(
    conn: sqlite3.Connection,
    *,
    topic: str | None,
) -> list[sqlite3.Row]:
    """Fetch rows, optionally narrowed by topic, path, or folder context."""
    base_query = (
        f"select id, absolute_path, relative_path, topic, folder_context, "
        f"file_name, file_type, chunk_start, chunk_end, text, embedding "
        f"from {TABLE_NAME}"
    )
    if topic:
        pattern = f"%{topic.lower()}%"
        return conn.execute(
            base_query
            + " where lower(topic) like ? "
            + "or lower(folder_context) like ? "
            + "or lower(relative_path) like ?",
            (pattern, pattern, pattern),
        ).fetchall()
    return conn.execute(base_query).fetchall()


def row_to_result(row: sqlite3.Row, score: float) -> dict[str, Any]:
    """Convert a SQLite row into an MCP-friendly result object."""
    text = str(row["text"])
    excerpt = text[:1200]
    return {
        "score": round(score, 4),
        "relative_path": row["relative_path"],
        "absolute_path": row["absolute_path"],
        "topic": row["topic"],
        "folder_context": row["folder_context"],
        "file_name": row["file_name"],
        "file_type": row["file_type"],
        "chunk_start": row["chunk_start"],
        "chunk_end": row["chunk_end"],
        "excerpt": excerpt,
    }


def search_documents(
    query: str,
    *,
    topic: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Search the local Hermes document index."""
    normalized_query = query.strip()
    if not normalized_query:
        return {"ok": False, "error": "query must not be empty", "results": []}

    ensure_directories()
    if not SQLITE_DB_PATH.exists():
        return {
            "ok": False,
            "error": f"index database does not exist yet: {SQLITE_DB_PATH}",
            "results": [],
        }

    result_limit = clamp_limit(limit)
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        if not table_exists(conn):
            return {
                "ok": False,
                "error": f"index table does not exist yet: {TABLE_NAME}",
                "results": [],
            }

        query_vector = embed_query(normalized_query)
        rows = fetch_candidate_rows(conn, topic=topic.strip() if topic else None)
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            try:
                score = cosine_score(query_vector, decode_embedding(row["embedding"]))
            except Exception as exc:
                LOGGER.warning("Skipping malformed embedding row %s: %s", row["id"], exc)
                continue
            if score != float("-inf"):
                scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        results = [row_to_result(row, score) for score, row in scored[:result_limit]]
        return {
            "ok": True,
            "query": normalized_query,
            "topic": topic,
            "limit": result_limit,
            "candidate_count": len(rows),
            "results": results,
        }
    finally:
        conn.close()

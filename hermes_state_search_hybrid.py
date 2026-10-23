"""Gateway session search: fuzzy index, optional semantic index, RRF when both exist.

The index lives next to ``state.db`` (gateway), never on a client. Fuzzy search
always runs (the existing FTS/trigram ``search_messages`` path). Semantic search
runs only when ``session_search.embedding_model`` is set; those two rankings are
merged with reciprocal rank fusion. No model configured means fuzzy only — the
semantic store is not read and no embedding call is made.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import struct
import urllib.error
import urllib.request
from array import array
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

RRF_K = 60
_INDEX_BATCH = 24
_EMBED_TEXT_CAP = 2000
_SIDECAR_NAME = "session-search-embeddings.db"

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def rrf_fuse(rankings: Sequence[Sequence[Any]], *, k: int = RRF_K) -> list[Any]:
    """Merge ranked id lists. Higher fused score first; ties keep the earlier list's order."""
    scores: dict[Any, float] = {}
    order: list[Any] = []
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            if item not in scores:
                order.append(item)
                scores[item] = 0.0
            scores[item] += 1.0 / (k + rank)
    return sorted(order, key=lambda item: (-scores[item], order.index(item)))


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = norm_l = norm_r = 0.0
    for a, b in zip(left, right):
        dot += a * b
        norm_l += a * a
        norm_r += b * b
    if norm_l <= 0.0 or norm_r <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_l) * math.sqrt(norm_r))


def embedding_settings(config: Optional[dict] = None) -> dict[str, str]:
    """Resolved embedding config. Empty ``model`` means fuzzy-only."""
    block = {}
    if config is None:
        try:
            from hermes_cli.config import load_config
            config = load_config()
        except Exception:
            logger.debug("session search embedding config unreadable", exc_info=True)
            config = {}
    raw = (config or {}).get("session_search") or {}
    if isinstance(raw, dict):
        block = raw
    model = str(block.get("embedding_model") or "").strip()
    base_url = str(block.get("embedding_base_url") or "").strip().rstrip("/")
    api_key = (
        os.environ.get("SESSION_SEARCH_EMBEDDING_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    return {"model": model, "base_url": base_url, "api_key": api_key}


def semantic_enabled(settings: Optional[dict[str, str]] = None) -> bool:
    cfg = settings if settings is not None else embedding_settings()
    return bool(cfg.get("model") and cfg.get("base_url") and cfg.get("api_key"))


def _pack(vector: Sequence[float]) -> bytes:
    buf = array("f", (float(v) for v in vector))
    if buf.itemsize != 4:
        return struct.pack(f"<{len(vector)}f", *vector)
    return buf.tobytes()


def _unpack(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob[: count * 4]))


class EmbeddingIndex:
    """Sidecar vector store beside the gateway session DB. Not opened when fuzzy-only."""

    def __init__(self, db_path: Path):
        self.path = Path(db_path).parent / _SIDECAR_NAME
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=5.0)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS message_embeddings (
                message_id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                model TEXT NOT NULL,
                vector BLOB NOT NULL
            )
            """
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def stored_ids(self, model: str) -> set[int]:
        rows = self._conn.execute(
            "SELECT message_id FROM message_embeddings WHERE model = ?", (model,)
        ).fetchall()
        return {int(row[0]) for row in rows}

    def upsert(self, model: str, rows: Sequence[tuple[int, str, Sequence[float]]]) -> None:
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO message_embeddings (message_id, session_id, model, vector)
            VALUES (?, ?, ?, ?)
            """,
            [(mid, sid, model, _pack(vec)) for mid, sid, vec in rows if vec],
        )
        self._conn.commit()

    def ranked_sessions(self, model: str, query_vector: Sequence[float], limit: int) -> list[str]:
        best: dict[str, float] = {}
        for sid, blob in self._conn.execute(
            "SELECT session_id, vector FROM message_embeddings WHERE model = ?", (model,)
        ):
            score = cosine(query_vector, _unpack(blob))
            if score > best.get(sid, 0.0):
                best[sid] = score
        ordered = sorted(best, key=lambda sid: (-best[sid], sid))
        return ordered[:limit]


def openai_embeddings(texts: Sequence[str], *, model: str, base_url: str, api_key: str) -> list[list[float]]:
    """OpenAI-compatible ``/embeddings``. Raises on HTTP or shape errors."""
    body = json.dumps({"model": model, "input": list(texts)}).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/embeddings",
        data=body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"embedding HTTP {exc.code}: {detail}") from exc
    data = payload.get("data") or []
    data.sort(key=lambda row: row.get("index", 0))
    return [list(row["embedding"]) for row in data]


def _message_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    text = str(raw)
    if text.startswith("[") or text.startswith("{"):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = text
        else:
            from hermes_state_search import _flatten_text
            text = _flatten_text(decoded)
    return text.strip()[:_EMBED_TEXT_CAP]


def _recent_messages(db, limit: int) -> list[tuple[int, str]]:
    path = getattr(db, "db_path", None)
    if not path:
        return []
    uri = f"file:{path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    except sqlite3.Error:
        logger.debug("session search index read failed", exc_info=True)
        return []
    try:
        rows = conn.execute(
            """
            SELECT id, session_id, content FROM messages
            WHERE role IN ('user', 'assistant') AND content IS NOT NULL AND content != ''
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    except sqlite3.Error:
        logger.debug("session search message scan failed", exc_info=True)
        return []
    finally:
        conn.close()
    out = []
    for mid, sid, content in rows:
        text = _message_text(content)
        if text and sid:
            out.append((int(mid), str(sid), text))
    return out


def _session_rows(db, session_ids: Sequence[str]) -> dict[str, dict]:
    if not session_ids:
        return {}
    path = getattr(db, "db_path", None)
    if not path:
        return {}
    placeholders = ",".join("?" for _ in session_ids)
    sql = f"""
        SELECT s.id AS session_id, s.source, s.model, s.started_at AS session_started,
               substr(m.content, 1, 240) AS snippet, m.role
        FROM sessions s
        LEFT JOIN messages m ON m.id = (
            SELECT id FROM messages WHERE session_id = s.id ORDER BY id DESC LIMIT 1
        )
        WHERE s.id IN ({placeholders})
    """
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, list(session_ids)).fetchall()
        conn.close()
    except sqlite3.Error:
        logger.debug("session search session hydrate failed", exc_info=True)
        return {}
    return {str(row["session_id"]): dict(row) for row in rows}


def content_search_matches(
    db,
    query: str,
    *,
    source_filter=None,
    exclude_sources=None,
    limit: int = 50,
    fields=None,
    settings: Optional[dict[str, str]] = None,
    embed: Optional[EmbedFn] = None,
    index: Optional[EmbeddingIndex] = None,
) -> list[dict]:
    """Fuzzy message hits, fused with semantic hits when an embedding model is configured."""
    fuzzy = db.search_messages(
        query=query,
        source_filter=source_filter,
        exclude_sources=exclude_sources,
        limit=limit,
        fields=fields,
    )
    cfg = settings if settings is not None else embedding_settings()
    if not semantic_enabled(cfg):
        return fuzzy
    embed_fn = embed or (
        lambda texts: openai_embeddings(
            texts, model=cfg["model"], base_url=cfg["base_url"], api_key=cfg["api_key"]
        )
    )
    owned = index is None
    store = index or EmbeddingIndex(db.db_path)
    try:
        _index_batch(store, db, cfg["model"], embed_fn)
        try:
            query_vec = embed_fn([query])[0]
        except Exception:
            logger.debug("session search query embed failed; fuzzy only", exc_info=True)
            return fuzzy
        semantic_sessions = store.ranked_sessions(cfg["model"], query_vec, limit)
    finally:
        if owned:
            store.close()
    if not semantic_sessions:
        return fuzzy
    fuzzy_sessions: list[str] = []
    by_session: dict[str, dict] = {}
    for row in fuzzy:
        sid = row.get("session_id")
        if not sid or sid in by_session:
            continue
        by_session[sid] = row
        fuzzy_sessions.append(sid)
    fused = rrf_fuse([fuzzy_sessions, semantic_sessions])[:limit]
    missing = [sid for sid in fused if sid not in by_session]
    by_session.update(_session_rows(db, missing))
    ordered = [by_session[sid] for sid in fused if sid in by_session]
    return ordered or fuzzy


def _index_batch(store: EmbeddingIndex, db, model: str, embed_fn: EmbedFn) -> None:
    have = store.stored_ids(model)
    pending = [
        (mid, sid, text)
        for mid, sid, text in _recent_messages(db, _INDEX_BATCH * 4)
        if mid not in have
    ][:_INDEX_BATCH]
    if not pending:
        return
    try:
        vectors = embed_fn([text for _, _, text in pending])
    except Exception:
        logger.debug("session search index embed failed", exc_info=True)
        return
    store.upsert(
        model,
        [(mid, sid, vec) for (mid, sid, _), vec in zip(pending, vectors)],
    )

"""Gateway session search: fuzzy index, optional semantic index, RRF when both exist.

The index lives next to ``state.db`` (gateway), never on a client. Fuzzy search
always runs (the existing FTS/trigram ``search_messages`` path). Semantic search
runs only when ``session_search.embedding_model`` is set; those two rankings are
merged with reciprocal rank fusion. No model configured means fuzzy only — the
semantic store is not read and no embedding call is made.

Semantic hits honor the same ``source`` / ``exclude_sources`` filters as the
fuzzy lane. Sidecar open/read failures degrade to the fuzzy ranking already
computed — search never 500s solely because embeddings are unavailable.

Indexing is incremental: each search embeds up to ``_INDEX_BATCH`` still-missing
messages, walking ``messages.id`` newest-first past already-stored rows so older
history is eventually covered rather than only the current tail window.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
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
# Drop embedding neighbors below this cosine. text-embedding-3-small keyword
# queries otherwise surface weak "top-N of everything" rows and the drawer
# looks like a recent-session dump after a multi-second index stall.
SEMANTIC_MIN_COSINE = 0.35
# Single-token typos like "logfeq" → "logseq": FTS prefix alone misses these.
_TYPO_MIN_TOKEN_LEN = 4
_TYPO_PREFIX_LEN = 3
_TYPO_CANDIDATE_HITS = 40
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


def edit_distance(left: str, right: str, *, max_distance: int = 2) -> int:
    """Bounded Levenshtein; returns max_distance+1 when farther than the cap."""
    if left == right:
        return 0
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    if not left:
        return len(right)
    if not right:
        return len(left)
    # two-row DP
    prev = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        curr = [i] + [0] * len(right)
        row_min = curr[0]
        for j, b in enumerate(right, start=1):
            cost = 0 if a == b else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            row_min = min(row_min, curr[j])
        if row_min > max_distance:
            return max_distance + 1
        prev = curr
    return prev[-1] if prev[-1] <= max_distance else max_distance + 1


def _strip_fts_markup(query: str) -> str:
    return re.sub(r"[*\"']+", " ", query or "").strip()


def _max_edits_for(token: str) -> int:
    # Short tokens: one key slip only. Longer: allow two (transpose-ish typos).
    return 1 if len(token) <= 6 else 2


def _words_in(*blobs: Any) -> set[str]:
    out: set[str] = set()
    for blob in blobs:
        if not blob:
            continue
        # FTS may wrap hits as >>>term<<< — strip markers before tokenizing.
        text = re.sub(r">>>+|<<<+", " ", str(blob))
        out.update(m.group(0).lower() for m in _WORD_RE.finditer(text))
    return out


def suggest_typo_terms(db, query: str, *, limit: int = 5) -> list[str]:
    """When FTS misses a single-token slip, propose near-neighbor terms from prefix hits.

    Uses a cheap 3-char FTS prefix probe, then keeps words within a small edit
    distance of the typed token. No external spell dictionary required.
    """
    tokens = [
        t.lower()
        for t in re.findall(r"[A-Za-z0-9_]+", _strip_fts_markup(query))
        if len(t) >= _TYPO_MIN_TOKEN_LEN
    ]
    if not tokens:
        return []
    search = getattr(db, "search_messages", None)
    if not callable(search):
        return []
    rescued: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        prefix = tok[:_TYPO_PREFIX_LEN] + "*"
        try:
            rows = search(
                query=prefix,
                limit=_TYPO_CANDIDATE_HITS,
                fields=("session_id", "snippet", "role", "source", "model", "session_started"),
            )
        except TypeError:
            try:
                rows = search(query=prefix, limit=_TYPO_CANDIDATE_HITS)
            except Exception:
                continue
        except Exception:
            continue
        max_d = _max_edits_for(tok)
        scored: list[tuple[int, str]] = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            for word in _words_in(row.get("snippet"), row.get("title"), row.get("content")):
                if word == tok or word in seen:
                    continue
                dist = edit_distance(tok, word, max_distance=max_d)
                if dist <= max_d:
                    scored.append((dist, word))
        scored.sort(key=lambda item: (item[0], item[1]))
        for _, word in scored:
            if word in seen:
                continue
            seen.add(word)
            rescued.append(word)
            if len(rescued) >= limit:
                return rescued
    return rescued


def _annotate_match_kind(rows: list[dict], kind: str) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        annotated = dict(row)
        annotated.setdefault("match_kind", kind)
        out.append(annotated)
    return out


def _highlight_term(snippet: str, term: str) -> str:
    """Wrap the first case-insensitive whole-ish match in >>>term<<< when missing."""
    text = (snippet or "").strip()
    if not text or not term:
        return text
    if ">>>" in text and "<<<" in text:
        return text
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    return pattern.sub(lambda m: f">>>{m.group(0)}<<<", text, count=1)


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

    def ranked_sessions(
        self,
        model: str,
        query_vector: Sequence[float],
        limit: int,
        *,
        min_score: float = SEMANTIC_MIN_COSINE,
    ) -> list[str]:
        best: dict[str, float] = {}
        for sid, blob in self._conn.execute(
            "SELECT session_id, vector FROM message_embeddings WHERE model = ?", (model,)
        ):
            score = cosine(query_vector, _unpack(blob))
            if score < min_score:
                continue
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


def _source_clause(source_filter=None, exclude_sources=None) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if source_filter is not None:
        allowed = list(source_filter)
        if not allowed:
            return "1=0", []
        clauses.append(f"s.source IN ({','.join('?' for _ in allowed)})")
        params.extend(allowed)
    if exclude_sources is not None:
        blocked = list(exclude_sources)
        if blocked:
            clauses.append(f"s.source NOT IN ({','.join('?' for _ in blocked)})")
            params.extend(blocked)
    if not clauses:
        return "1=1", []
    return " AND ".join(clauses), params


def _unindexed_messages(db, have: set[int], limit: int) -> list[tuple[int, str, str]]:
    """Newest-first unindexed user/assistant rows, paging past already-stored ids."""
    path = getattr(db, "db_path", None)
    if not path or limit <= 0:
        return []
    uri = f"file:{path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
    except sqlite3.Error:
        logger.debug("session search index read failed", exc_info=True)
        return []
    out: list[tuple[int, str, str]] = []
    # Page deeper than the batch so already-indexed recent rows do not hide older
    # still-missing history when embeddings are enabled after traffic has grown.
    page = max(limit * 4, _INDEX_BATCH * 4)
    before_id: Optional[int] = None
    try:
        while len(out) < limit:
            if before_id is None:
                sql = """
                    SELECT id, session_id, content FROM messages
                    WHERE role IN ('user', 'assistant')
                      AND content IS NOT NULL AND content != ''
                    ORDER BY id DESC
                    LIMIT ?
                """
                params: list[Any] = [page]
            else:
                sql = """
                    SELECT id, session_id, content FROM messages
                    WHERE role IN ('user', 'assistant')
                      AND content IS NOT NULL AND content != ''
                      AND id < ?
                    ORDER BY id DESC
                    LIMIT ?
                """
                params = [before_id, page]
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.Error:
                logger.debug("session search message scan failed", exc_info=True)
                return out
            if not rows:
                break
            before_id = int(rows[-1][0])
            for mid, sid, content in rows:
                mid_i = int(mid)
                if mid_i in have:
                    continue
                text = _message_text(content)
                if text and sid:
                    out.append((mid_i, str(sid), text))
                    if len(out) >= limit:
                        break
            if len(rows) < page:
                break
    finally:
        conn.close()
    return out


def _session_rows(
    db,
    session_ids: Sequence[str],
    *,
    source_filter=None,
    exclude_sources=None,
) -> dict[str, dict]:
    if not session_ids:
        return {}
    path = getattr(db, "db_path", None)
    if not path:
        return {}
    placeholders = ",".join("?" for _ in session_ids)
    source_sql, source_params = _source_clause(source_filter, exclude_sources)
    sql = f"""
        SELECT s.id AS session_id, s.source, s.model, s.started_at AS session_started,
               substr(m.content, 1, 240) AS snippet, m.role
        FROM sessions s
        LEFT JOIN messages m ON m.id = (
            SELECT id FROM messages WHERE session_id = s.id ORDER BY id DESC LIMIT 1
        )
        WHERE s.id IN ({placeholders}) AND ({source_sql})
    """
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, list(session_ids) + source_params).fetchall()
        conn.close()
    except sqlite3.Error:
        logger.debug("session search session hydrate failed", exc_info=True)
        return {}
    return {str(row["session_id"]): dict(row) for row in rows}


def _filter_session_ids(
    db,
    session_ids: Sequence[str],
    *,
    source_filter=None,
    exclude_sources=None,
) -> list[str]:
    if not session_ids or (source_filter is None and exclude_sources is None):
        return list(session_ids)
    allowed = _session_rows(
        db,
        session_ids,
        source_filter=source_filter,
        exclude_sources=exclude_sources,
    )
    return [sid for sid in session_ids if sid in allowed]


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
    """Fuzzy message hits, fused with semantic hits when an embedding model is configured.

    Each row carries ``match_kind``:
    - ``fts`` — direct lexical hit
    - ``expanded`` — typo/near-neighbor rescue when primary FTS was empty
    - ``semantic`` — embedding-only neighbor (no lexical hit for that session)
    """
    fuzzy = db.search_messages(
        query=query,
        source_filter=source_filter,
        exclude_sources=exclude_sources,
        limit=limit,
        fields=fields,
    )
    fuzzy_kind = "fts"
    if not fuzzy:
        # Primary FTS empty → try 1–2 edit neighbors discovered via a short prefix probe.
        for term in suggest_typo_terms(db, query):
            rescue_q = term if term.endswith("*") else f"{term}*"
            try:
                rescued = db.search_messages(
                    query=rescue_q,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    limit=limit,
                    fields=fields,
                )
            except Exception:
                continue
            if not rescued:
                continue
            fuzzy = []
            for row in rescued:
                annotated = dict(row)
                annotated["match_kind"] = "expanded"
                snip = annotated.get("snippet") or ""
                annotated["snippet"] = _highlight_term(str(snip), term)
                fuzzy.append(annotated)
            fuzzy_kind = "expanded"
            break
    fuzzy = _annotate_match_kind(list(fuzzy or []), fuzzy_kind)

    cfg = settings if settings is not None else embedding_settings()
    if not semantic_enabled(cfg):
        return fuzzy
    embed_fn = embed or (
        lambda texts: openai_embeddings(
            texts, model=cfg["model"], base_url=cfg["base_url"], api_key=cfg["api_key"]
        )
    )
    owned = index is None
    store = index
    try:
        if store is None:
            try:
                store = EmbeddingIndex(db.db_path)
            except Exception:
                logger.debug("session search sidecar open failed; fuzzy only", exc_info=True)
                return fuzzy
        try:
            # Warm sidecars: rank what is already embedded. Re-index is deferred so a
            # drawer keyword search is not blocked on a 24-row OpenRouter batch
            # (multi-second stall → empty UI → then a polluted result list).
            # Cold sidecars still bootstrap one batch so semantic can come online.
            try:
                if not store.stored_ids(cfg["model"]):
                    _index_batch(store, db, cfg["model"], embed_fn)
            except Exception:
                logger.debug("session search warm-check/index failed; continue", exc_info=True)
            try:
                query_vec = embed_fn([query])[0]
            except Exception:
                logger.debug("session search query embed failed; fuzzy only", exc_info=True)
                return fuzzy
            semantic_sessions = store.ranked_sessions(cfg["model"], query_vec, max(limit * 4, limit))
        except Exception:
            logger.debug("session search sidecar read failed; fuzzy only", exc_info=True)
            return fuzzy
    finally:
        if owned and store is not None:
            try:
                store.close()
            except Exception:
                logger.debug("session search sidecar close failed", exc_info=True)
    semantic_sessions = _filter_session_ids(
        db,
        semantic_sessions,
        source_filter=source_filter,
        exclude_sources=exclude_sources,
    )
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
    semantic_only = _session_rows(
        db,
        missing,
        source_filter=source_filter,
        exclude_sources=exclude_sources,
    )
    for sid, row in semantic_only.items():
        annotated = dict(row)
        annotated["match_kind"] = "semantic"
        # No lexical snippet — clients render a "Semantically similar" chip.
        if not (annotated.get("snippet") or "").strip():
            annotated["snippet"] = ""
        by_session[sid] = annotated
    ordered = [by_session[sid] for sid in fused if sid in by_session]
    return ordered or fuzzy


def _index_batch(store: EmbeddingIndex, db, model: str, embed_fn: EmbedFn) -> None:
    have = store.stored_ids(model)
    pending = _unindexed_messages(db, have, _INDEX_BATCH)
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

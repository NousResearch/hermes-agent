"""
Hybrid per-message skill retrieval (issue #34823).

Combines two complementary rankers and fuses their outputs with
Reciprocal Rank Fusion (RRF, k=60):

  Layer 1 — BM25 (Okapi BM25, k1=1.5, b=0.75)
      Always runs. Pure-Python, stdlib-only, ~1 ms. Scores skill
      name + description against the user's message. Catches exact
      and near-exact keyword matches ("docker skill" → docker skill).

  Layer 2 — Dense embeddings (optional)
      Runs only when ``embedding_model`` is configured in
      ``skills.semantic_search`` config and the SQLite index is warm.
      Catches semantic matches ("containerise my app" → docker skill).
      Embeddings are built in a background daemon thread so the agent
      loop is never blocked.

Public API
----------
retrieve_skills(query, skills, top_k, *, embedding_cfg) -> set[str]
    Return the names of the top-k most relevant skills.
    Short-circuits (returns all names) when len(skills) <= top_k.

get_index() -> EmbeddingIndex
    Return the process-singleton EmbeddingIndex (thread-safe).

EmbeddingIndex.invalidate(names=None)
    Drop cached embeddings for specific skills (or all). Called by
    skill_manager_tool, skills_sync, and skills_hub after mutations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer (shared by BM25 and similarity helpers)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word-boundary tokenizer. No stopword list needed —
    BM25's IDF naturally suppresses high-frequency terms."""
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# BM25 scorer
# ---------------------------------------------------------------------------


class BM25Scorer:
    """Okapi BM25 over an in-memory corpus.

    Parameters k1=1.5, b=0.75 are the standard Robertson et al. values and
    work well across short skill-description corpora without tuning.
    """

    k1: float = 1.5
    b: float = 0.75

    def __init__(self, corpus: list[str]) -> None:
        self._n = len(corpus)
        self._tokenized: list[list[str]] = [_tokenize(doc) for doc in corpus]
        total_len = sum(len(t) for t in self._tokenized)
        self._avg_dl: float = (total_len / self._n) if self._n else 1.0

        # Document-frequency map: how many docs contain each term
        df: dict[str, int] = {}
        for tokens in self._tokenized:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        self._df = df

    def _idf(self, term: str) -> float:
        """Robertson IDF — always positive, never negative."""
        df = self._df.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str) -> list[float]:
        """Return a BM25 score for each document in the corpus.

        Returned list is parallel to the ``corpus`` passed to __init__.
        """
        q_terms = _tokenize(query)
        if not q_terms:
            return [0.0] * self._n

        scores: list[float] = []
        for tokens in self._tokenized:
            dl = len(tokens)
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            s = 0.0
            for term in q_terms:
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                idf = self._idf(term)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * dl / self._avg_dl
                )
                s += idf * numerator / denominator
            scores.append(s)
        return scores


# ---------------------------------------------------------------------------
# Dense embedding index (SQLite-backed)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_embeddings (
    name         TEXT PRIMARY KEY,
    description  TEXT NOT NULL,
    embedding    TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    indexed_at   REAL NOT NULL
);
"""


def _content_hash(name: str, description: str) -> str:
    return hashlib.md5(f"{name}|{description}".encode("utf-8")).hexdigest()


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. Returns 0.0 for zero vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    val = dot / (na * nb)
    return max(-1.0, min(1.0, val))


class EmbeddingIndex:
    """Thread-safe SQLite-backed dense vector store.

    The DB lives at ``~/.hermes/.skill_index.db`` (inside HERMES_HOME,
    not inside the skills directory so it survives profile migrations).

    Write operations use WAL mode so concurrent readers are never blocked
    by an in-progress background upsert.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path or (get_hermes_home() / ".skill_index.db")
        self._lock = threading.Lock()
        self._init_db()

    # ------------------------------------------------------------------
    # Low-level DB helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        try:
            with self._lock:
                with self._connect() as conn:
                    conn.execute(_SCHEMA)
        except Exception as exc:
            logger.debug("EmbeddingIndex: DB init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_warm(self) -> bool:
        """True if at least one embedding row exists in the DB."""
        try:
            with self._lock:
                with self._connect() as conn:
                    row = conn.execute(
                        "SELECT 1 FROM skill_embeddings LIMIT 1"
                    ).fetchone()
                    return row is not None
        except Exception:
            return False

    def query(
        self,
        text: str,
        names: list[str],
        *,
        embedding_cfg: dict,
    ) -> dict[str, float]:
        """Embed *text* and return cosine similarity for each skill name.

        Only names already present in the DB are scored; missing names
        are silently absent from the returned dict (RRF treats them as
        ranked last). Returns {} on any error — caller falls back to
        BM25-only gracefully.
        """
        if not names:
            return {}
        try:
            vec = self._embed(text, embedding_cfg)
            if vec is None:
                return {}

            with self._lock:
                with self._connect() as conn:
                    placeholders = ",".join("?" * len(names))
                    rows = conn.execute(
                        f"SELECT name, embedding FROM skill_embeddings "
                        f"WHERE name IN ({placeholders})",
                        names,
                    ).fetchall()

            result: dict[str, float] = {}
            for name, emb_json in rows:
                emb = json.loads(emb_json)
                result[name] = _cosine(vec, emb)
            return result
        except Exception as exc:
            logger.debug("EmbeddingIndex.query failed: %s", exc)
            return {}

    def upsert(self, skills: list[dict], *, embedding_cfg: dict) -> None:
        """Insert or update embeddings for skills whose content changed.

        Each skill is checked against a content hash (md5 of name + description).
        Skills whose hash matches the stored value are skipped so we never
        re-embed unchanged entries.
        """
        import time

        for skill in skills:
            name = skill.get("name", "")
            description = skill.get("description", "")
            if not name:
                continue
            ch = _content_hash(name, description)

            # Skip if the stored hash is current
            try:
                with self._lock:
                    with self._connect() as conn:
                        row = conn.execute(
                            "SELECT content_hash FROM skill_embeddings WHERE name=?",
                            (name,),
                        ).fetchone()
                if row and row[0] == ch:
                    continue
            except Exception as exc:
                logger.debug(
                    "EmbeddingIndex: hash check failed for %s: %s", name, exc
                )

            # Obtain embedding
            vec = self._embed(f"{name} {description}", embedding_cfg)
            if vec is None:
                continue

            try:
                with self._lock:
                    with self._connect() as conn:
                        conn.execute(
                            "INSERT OR REPLACE INTO skill_embeddings "
                            "(name, description, embedding, content_hash, indexed_at) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (name, description, json.dumps(vec), ch, time.time()),
                        )
            except Exception as exc:
                logger.debug(
                    "EmbeddingIndex: upsert failed for %s: %s", name, exc
                )

    def invalidate(self, names: Optional[list[str]] = None) -> None:
        """Drop cached embeddings.

        When *names* is None, the entire index is cleared (full rebuild on
        the next turn). Otherwise only the named skills are removed.
        """
        try:
            with self._lock:
                with self._connect() as conn:
                    if names is None:
                        conn.execute("DELETE FROM skill_embeddings")
                        logger.debug("EmbeddingIndex: full invalidation")
                    else:
                        placeholders = ",".join("?" * len(names))
                        conn.execute(
                            f"DELETE FROM skill_embeddings WHERE name IN ({placeholders})",
                            names,
                        )
                        logger.debug("EmbeddingIndex: invalidated %s", names)
        except Exception as exc:
            logger.debug("EmbeddingIndex.invalidate failed: %s", exc)

    def build_async(self, skills: list[dict], *, embedding_cfg: dict) -> None:
        """Fire a daemon thread to upsert embeddings for *skills*.

        The agent loop never waits on this — it completes in the background.
        At most one build thread per process is spawned (the thread is named
        ``skill-index-build`` and is a daemon so it never prevents shutdown).
        """
        t = threading.Thread(
            target=self._build_worker,
            args=(skills, embedding_cfg),
            daemon=True,
            name="skill-index-build",
        )
        t.start()

    def _build_worker(self, skills: list[dict], embedding_cfg: dict) -> None:
        try:
            self.upsert(skills, embedding_cfg=embedding_cfg)
        except Exception as exc:
            logger.debug("EmbeddingIndex: background build error: %s", exc)

    # ------------------------------------------------------------------
    # Internal: provider embedding call
    # ------------------------------------------------------------------

    @staticmethod
    def _embed(text: str, cfg: dict) -> Optional[list[float]]:
        """Call the configured provider's embeddings endpoint.

        Uses the same ``auxiliary_client`` resolution chain as other
        side tasks (compression, vision, web_extract).  Returns None
        on any failure — callers treat None as "skip dense layer".
        """
        try:
            from agent.auxiliary_client import resolve_provider_client  # type: ignore

            client, _ = resolve_provider_client(
                task="embedding",
                provider=cfg.get("embedding_provider", "auto"),
                base_url=cfg.get("embedding_base_url", ""),
                api_key=cfg.get("embedding_api_key", ""),
            )
            resp = client.embeddings.create(
                model=cfg["embedding_model"],
                input=[text],
            )
            return resp.data[0].embedding  # type: ignore[return-value]
        except Exception as exc:
            logger.debug("EmbeddingIndex._embed failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Process-level singleton
# ---------------------------------------------------------------------------

_INDEX: Optional[EmbeddingIndex] = None
_INDEX_LOCK = threading.Lock()


def get_index() -> EmbeddingIndex:
    """Return the process-level EmbeddingIndex singleton (lazy init, thread-safe)."""
    global _INDEX
    with _INDEX_LOCK:
        if _INDEX is None:
            _INDEX = EmbeddingIndex()
        return _INDEX


# ---------------------------------------------------------------------------
# RRF fusion helpers
# ---------------------------------------------------------------------------


def _scores_to_ranks(scores: list[float], names: list[str]) -> dict[str, int]:
    """Convert a parallel float scores list to {name: rank} (rank 0 = highest score)."""
    paired = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
    return {name: rank for rank, (name, _) in enumerate(paired)}


def _dict_scores_to_ranks(
    score_dict: dict[str, float], all_names: list[str]
) -> dict[str, int]:
    """Convert a {name: score} dict to {name: rank}.

    Names absent from *score_dict* (e.g. not yet indexed) receive a rank of
    ``len(all_names)`` — last place — so they don't unfairly boost via RRF.
    """
    paired = sorted(
        ((name, score_dict.get(name, 0.0)) for name in all_names),
        key=lambda x: x[1],
        reverse=True,
    )
    return {name: rank for rank, (name, _) in enumerate(paired)}


def _rrf(
    rank_lists: list[dict[str, int]],
    names: list[str],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion over multiple rank dicts.

    score(d) = Σ_i  1 / (k + rank_i(d))

    k=60 is the standard value from the original RRF paper (Cormack et al.
    2009). Empty rank dicts are ignored so BM25-only still produces a valid
    ranking when the dense layer is absent.
    """
    fused: dict[str, float] = {name: 0.0 for name in names}
    for ranks in rank_lists:
        if not ranks:
            continue
        n = len(names)
        for name in names:
            r = ranks.get(name, n)  # absent → last place
            fused[name] += 1.0 / (k + r)
    return fused


# ---------------------------------------------------------------------------
# Main retrieval entry point
# ---------------------------------------------------------------------------


def retrieve_skills(
    query: str,
    skills: list[dict],
    top_k: int = 5,
    *,
    embedding_cfg: Optional[dict] = None,
) -> set[str]:
    """Return the names of the top-k most relevant skills via hybrid search.

    Algorithm
    ---------
    1. BM25 scores every skill's ``name + description`` against *query*.
    2. If *embedding_cfg* contains a non-empty ``embedding_model`` and the
       index is warm, dense cosine similarity scores are added.
    3. Ranks from both layers are fused with RRF (k=60).
    4. The top-k names by fused score are returned.

    When ``len(skills) <= top_k``, returns all names immediately (no ranking
    needed — the full-index render is unchanged for small skill sets).

    The dense index build is always triggered in a background thread when
    ``embedding_model`` is configured — this keeps the index warm for future
    turns without delaying the current one.

    Parameters
    ----------
    query:
        The user's message for the current turn.
    skills:
        List of dicts, each with at least ``"name"`` and ``"description"`` keys.
        Must be non-empty.
    top_k:
        Maximum number of skill names to return.
    embedding_cfg:
        The ``skills.semantic_search`` config sub-dict, or None for BM25-only.
    """
    if not skills:
        return set()

    # Short-circuit: no filtering needed when skill count fits in top_k
    if len(skills) <= top_k:
        return {s["name"] for s in skills}

    names = [s["name"] for s in skills]
    corpus = [f"{s['name']} {s.get('description', '')}" for s in skills]

    # ── Layer 1: BM25 (always) ───────────────────────────────────────────
    bm25 = BM25Scorer(corpus)
    bm25_raw = bm25.score(query)
    bm25_ranks = _scores_to_ranks(bm25_raw, names)

    # ── Layer 2: dense embeddings (optional) ─────────────────────────────
    dense_ranks: dict[str, int] = {}
    if embedding_cfg and embedding_cfg.get("embedding_model"):
        idx = get_index()
        if idx.is_warm():
            dense_scores = idx.query(query, names, embedding_cfg=embedding_cfg)
            dense_ranks = _dict_scores_to_ranks(dense_scores, names)
            logger.debug(
                "skill_retrieval: dense layer active (%d/%d skills scored)",
                len(dense_scores),
                len(names),
            )
        # Always queue a background rebuild so the index stays warm
        idx.build_async(skills, embedding_cfg=embedding_cfg)

    # ── RRF fusion ───────────────────────────────────────────────────────
    fused = _rrf([bm25_ranks, dense_ranks], names)
    top = sorted(fused, key=lambda n: fused[n], reverse=True)[:top_k]

    logger.debug(
        "skill_retrieval: query=%r top_%d=%s (dense=%s)",
        query[:60],
        top_k,
        top,
        bool(dense_ranks),
    )
    return set(top)

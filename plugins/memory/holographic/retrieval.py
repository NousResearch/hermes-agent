"""Hybrid keyword/BM25/embedding retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking, HRR vector scoring, and optional embedding
semantic similarity via sentence-transformers.
"""

from __future__ import annotations

import math
import struct
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding provider (optional, like numpy)
# ---------------------------------------------------------------------------

_HAS_SENTENCE_TRANSFORMERS: bool
try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False


def _embedding_to_bytes(vec: "np.ndarray") -> bytes:
    """Pack a float32 embedding vector into a compact BLOB."""
    return struct.pack(f"<{len(vec)}f", *vec.tolist())


def _bytes_to_embedding(data: bytes) -> "np.ndarray":
    """Unpack a BLOB back into a float32 numpy array."""
    import numpy as np
    n = len(data) // 4
    return np.array(struct.unpack(f"<{n}f", data), dtype=np.float32)


class EmbeddingProvider:
    """Wraps sentence-transformers for semantic embedding (optional dependency).

    Lazily loads the model on first use.  Provides encode() for single texts
    and cosine_similarity() for comparing vectors.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.model_name = model_name
        self.dim = dim
        self._model: "SentenceTransformer | None" = None
        self._loaded = False

    @property
    def available(self) -> bool:
        return _HAS_SENTENCE_TRANSFORMERS

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install with: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(self.model_name)
        self._loaded = True

    def encode(self, text: str) -> "np.ndarray":
        """Encode a single text into an embedding vector (float32)."""
        import numpy as np
        self._ensure_loaded()
        vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    @staticmethod
    def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
        """Cosine similarity between two vectors.  Assumes pre-normalized."""
        import numpy as np
        return float(np.dot(a, b))


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring.

    Four retrieval signals (when all available):
      - FTS5 full-text search  (fts_weight, default 0.25)
      - Jaccard token overlap   (jaccard_weight, default 0.20)
      - HRR vector similarity   (hrr_weight, default 0.15)
      - Embedding cosine sim    (embedding_weight, default 0.40)

    When a signal is unavailable its weight is redistributed proportionally
    among the remaining signals so totals always sum to 1.0.
    """

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.25,
        jaccard_weight: float = 0.20,
        hrr_weight: float = 0.15,
        hrr_dim: int = 1024,
        embedding_enabled: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        embedding_weight: float = 0.40,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim

        # --- Embedding provider (lazy, optional) ---
        self._embedding_enabled = embedding_enabled
        self._embedding_provider: EmbeddingProvider | None = None
        if embedding_enabled:
            self._embedding_provider = EmbeddingProvider(
                model_name=embedding_model, dim=embedding_dim,
            )

        # --- Auto-degrade unavailable signals and redistribute weights ---
        weights = {
            "fts": fts_weight,
            "jaccard": jaccard_weight,
            "hrr": hrr_weight,
            "embedding": embedding_weight if embedding_enabled else 0.0,
        }

        # Drop HRR if numpy unavailable
        if weights["hrr"] > 0 and not hrr._HAS_NUMPY:
            weights["hrr"] = 0.0

        # Drop embedding if sentence-transformers unavailable
        if weights["embedding"] > 0 and (
            self._embedding_provider is None or not self._embedding_provider.available
        ):
            weights["embedding"] = 0.0

        # Redistribute zeroed weights proportionally
        total = sum(weights.values())
        if total > 0 and total != 1.0:
            for key in weights:
                weights[key] /= total

        self.fts_weight = weights["fts"]
        self.jaccard_weight = weights["jaccard"]
        self.hrr_weight = weights["hrr"]
        self.embedding_weight = weights["embedding"]

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Hybrid search: FTS5 candidates -> Jaccard -> HRR -> Embedding -> trust.

        Pipeline:
        1. FTS5 search: Get limit*3 candidates from SQLite full-text search
        2. Jaccard boost: Token overlap between query and fact content
        3. HRR vector similarity (optional, requires numpy)
        4. Embedding cosine similarity (optional, requires sentence-transformers)
        5. Trust weighting: final_score = relevance * trust_score
        6. Temporal decay (optional): decay = 0.5^(age_days / half_life)

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        """
        # Stage 1: Get FTS5 candidates (more than limit for reranking headroom)
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)

        if not candidates:
            return []

        # Pre-compute query embedding once if embedding signal is active
        query_emb = None
        if self.embedding_weight > 0 and self._embedding_provider is not None:
            try:
                query_emb = self._embedding_provider.encode(query)
            except Exception:
                query_emb = None  # degrade gracefully

        # Stage 2: Rerank with Jaccard + HRR + Embedding + trust + optional decay
        query_tokens = self._tokenize(query)
        scored = []

        for fact in candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens

            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            fts_score = fact.get("fts_rank", 0.0)

            # HRR similarity
            if self.hrr_weight > 0 and fact.get("hrr_vector"):
                fact_vec = hrr.bytes_to_phases(fact["hrr_vector"])
                query_vec = hrr.encode_text(query, self.hrr_dim)
                hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0  # shift to [0,1]
            else:
                hrr_sim = 0.5  # neutral

            # Embedding similarity
            if self.embedding_weight > 0 and query_emb is not None and fact.get("embedding_vector"):
                try:
                    fact_emb = _bytes_to_embedding(fact["embedding_vector"])
                    emb_sim = self._embedding_provider.cosine_similarity(query_emb, fact_emb)
                    emb_sim = (emb_sim + 1.0) / 2.0  # shift to [0,1]
                except Exception:
                    emb_sim = 0.5  # neutral on failure
            else:
                emb_sim = 0.5  # neutral

            # Combine all four signals
            relevance = (self.fts_weight * fts_score
                        + self.jaccard_weight * jaccard
                        + self.hrr_weight * hrr_sim
                        + self.embedding_weight * emb_sim)

            # Trust weighting
            score = relevance * fact["trust_score"]

            # Optional temporal decay
            if self.half_life > 0:
                score *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

            fact["score"] = score
            scored.append(fact)

        # Stage 3: Apply engram strength boost (optional, degrades gracefully)
        try:
            fact_ids = [f["fact_id"] for f in scored]
            engram_map = self._engram_boost(fact_ids)
            if engram_map:
                for f in scored:
                    strength = engram_map.get(f["fact_id"], 1.0)
                    f["score"] *= (0.5 + 0.5 * strength)
        except Exception:
            pass  # degrade gracefully -- no engram boost applied

        # Sort by score descending, return top limit
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        # Strip raw HRR / embedding bytes — callers expect JSON-serializable dicts
        for fact in results:
            fact.pop("hrr_vector", None)
            fact.pop("embedding_vector", None)
        return results

    def _engram_boost(self, fact_ids: list[int]) -> dict[int, float]:
        """Look up engram strengths for candidate facts from pipeline_state.db.

        Queries the engram_strengths table for memory_ref entries matching
        'fact:<fact_id>' and returns a mapping of fact_id to strength value.
        Falls back to 1.0 for any fact not found.  Returns an empty dict
        (all defaults) if pipeline_state.db or the table is unavailable.
        """
        if not fact_ids:
            return {}
        try:
            pipeline_db = self.store.db_path.parent / "pipeline_state.db"
            if not pipeline_db.exists():
                return {}
            conn = sqlite3.connect(str(pipeline_db), timeout=5.0)
            conn.row_factory = sqlite3.Row
            try:
                placeholders = ",".join("?" for _ in fact_ids)
                refs = [f"fact:{fid}" for fid in fact_ids]
                rows = conn.execute(
                    f"SELECT memory_ref, strength FROM engram_strengths "
                    f"WHERE memory_ref IN ({placeholders})",
                    refs,
                ).fetchall()
                # Map back: 'fact:<id>' -> int id -> strength
                result: dict[int, float] = {}
                for row in rows:
                    ref = row["memory_ref"]
                    if ref.startswith("fact:"):
                        try:
                            fid = int(ref[5:])
                            result[fid] = row["strength"]
                        except (ValueError, KeyError):
                            pass
                return result
            finally:
                conn.close()
        except Exception as e:
            logger.debug("Engram boost lookup failed (graceful skip): %s", e)
            return {}

    def embedding_search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Pure embedding-based semantic search.

        Computes cosine similarity between the query embedding and every stored
        fact embedding.  Falls back to the hybrid search() if the embedding
        provider is unavailable.

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        """
        if self._embedding_provider is None or not self._embedding_provider.available:
            return self.search(query, category=category, min_trust=min_trust, limit=limit)

        try:
            query_emb = self._embedding_provider.encode(query)
        except Exception:
            return self.search(query, category=category, min_trust=min_trust, limit=limit)

        conn = self.store._conn

        where = "WHERE embedding_vector IS NOT NULL AND trust_score >= ?"
        params: list = [min_trust]
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   embedding_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            return []

        scored = []
        for row in rows:
            fact = dict(row)
            try:
                fact_emb = _bytes_to_embedding(fact.pop("embedding_vector"))
                sim = self._embedding_provider.cosine_similarity(query_emb, fact_emb)
            except Exception:
                continue
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def probe(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Compositional entity query using HRR algebra.

        Unbinds entity from memory bank to extract associated content.
        This is NOT keyword search — it uses algebraic structure to find facts
        where the entity plays a structural role.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            # Fallback to keyword search on entity name
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Encode entity as role-bound vector
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
        probe_key = hrr.bind(entity_vec, role_entity)

        # Try category-specific bank first, then all facts
        if category:
            bank_name = f"cat:{category}"
            bank_row = conn.execute(
                "SELECT vector FROM memory_banks WHERE bank_name = ?",
                (bank_name,),
            ).fetchone()
            if bank_row:
                bank_vec = hrr.bytes_to_phases(bank_row["vector"])
                extracted = hrr.unbind(bank_vec, probe_key)
                # Use extracted signal to score individual facts
                return self._score_facts_by_vector(
                    extracted, category=category, limit=limit
                )

        # Score against individual fact vectors directly
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            # Final fallback: keyword search
            return self.search(entity, category=category, limit=limit)

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
            # Unbind probe key from fact to see if entity is structurally present
            residual = hrr.unbind(fact_vec, probe_key)
            # Compare residual against content signal
            role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
            content_vec = hrr.bind(hrr.encode_text(fact["content"], self.hrr_dim), role_content)
            sim = hrr.similarity(residual, content_vec)
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def related(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Discover facts that share structural connections with an entity.

        Unlike probe (which finds facts *about* an entity), related finds
        facts that are connected through shared context — e.g., other entities
        mentioned alongside this one, or content that overlaps structurally.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, limit=limit)

        conn = self.store._conn

        # Encode entity as a bare atom (not role-bound — we want ANY structural match)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            return self.search(entity, category=category, limit=limit)

        # Score each fact by how much the entity's atom appears in its vector
        # This catches both role-bound entity matches AND content word matches
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))

            # Check structural similarity: unbind entity from fact
            residual = hrr.unbind(fact_vec, entity_vec)
            # A high-similarity residual to ANY known role vector means this entity
            # plays a structural role in the fact
            role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
            role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

            entity_role_sim = hrr.similarity(residual, role_entity)
            content_role_sim = hrr.similarity(residual, role_content)
            # Take the max — entity could appear in either role
            best_sim = max(entity_role_sim, content_role_sim)

            fact["score"] = (best_sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def reason(
        self,
        entities: list[str],
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Multi-entity compositional query — vector-space JOIN.

        Given multiple entities, algebraically intersects their structural
        connections to find facts related to ALL of them simultaneously.
        This is compositional reasoning that no embedding DB can do.

        Example: reason(["peppi", "backend"]) finds facts where peppi AND
        backend both play structural roles — without keyword matching.

        Falls back to FTS5 search if numpy unavailable.
        """
        if not hrr._HAS_NUMPY or not entities:
            # Fallback: search with all entities as keywords
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        conn = self.store._conn
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)

        # For each entity, compute what the bank "remembers" about it
        # by unbinding entity+role from each fact vector
        entity_residuals = []
        for entity in entities:
            entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
            probe_key = hrr.bind(entity_vec, role_entity)
            entity_residuals.append(probe_key)

        # Get all facts with vectors
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        if not rows:
            query = " ".join(entities)
            return self.search(query, category=category, limit=limit)

        # Score each fact by how much EACH entity is structurally present.
        # A fact scores high only if ALL entities have structural presence
        # (AND semantics via min, vs OR which would use mean/max).
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))

            entity_scores = []
            for probe_key in entity_residuals:
                residual = hrr.unbind(fact_vec, probe_key)
                sim = hrr.similarity(residual, role_content)
                entity_scores.append(sim)

            min_sim = min(entity_scores)
            fact["score"] = (min_sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def contradict(
        self,
        category: str | None = None,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Find potentially contradictory facts via entity overlap + content divergence.

        Two facts contradict when they share entities (same subject) but have
        low content-vector similarity (different claims). This is automated
        memory hygiene — no other memory system does this.

        Returns pairs of facts with a contradiction score.
        Falls back to empty list if numpy unavailable.
        """
        if not hrr._HAS_NUMPY:
            return []

        conn = self.store._conn

        # Get all facts with vectors and their linked entities
        where = "WHERE f.hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND f.category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                   f.created_at, f.updated_at, f.hrr_vector
            FROM facts f
            {where}
            """,
            params,
        ).fetchall()

        if len(rows) < 2:
            return []

        # Guard against O(n²) explosion on large fact stores.
        # At 500 facts, that's ~125K comparisons — acceptable.
        # Above that, only check the most recently updated facts.
        _MAX_CONTRADICT_FACTS = 500
        if len(rows) > _MAX_CONTRADICT_FACTS:
            rows = sorted(rows, key=lambda r: r["updated_at"] or r["created_at"], reverse=True)
            rows = rows[:_MAX_CONTRADICT_FACTS]

        # Build entity sets per fact
        fact_entities: dict[int, set[str]] = {}
        for row in rows:
            fid = row["fact_id"]
            entity_rows = conn.execute(
                """
                SELECT e.name FROM entities e
                JOIN fact_entities fe ON fe.entity_id = e.entity_id
                WHERE fe.fact_id = ?
                """,
                (fid,),
            ).fetchall()
            fact_entities[fid] = {r["name"].lower() for r in entity_rows}

        # Compare all pairs: high entity overlap + low content similarity = contradiction
        facts = [dict(r) for r in rows]
        contradictions = []

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                f1, f2 = facts[i], facts[j]
                ents1 = fact_entities.get(f1["fact_id"], set())
                ents2 = fact_entities.get(f2["fact_id"], set())

                if not ents1 or not ents2:
                    continue

                # Entity overlap (Jaccard)
                entity_overlap = len(ents1 & ents2) / len(ents1 | ents2) if (ents1 | ents2) else 0.0

                if entity_overlap < 0.3:
                    continue  # Not enough entity overlap to be contradictory

                # Content similarity via HRR vectors
                v1 = hrr.bytes_to_phases(f1["hrr_vector"])
                v2 = hrr.bytes_to_phases(f2["hrr_vector"])
                content_sim = hrr.similarity(v1, v2)

                # High entity overlap + low content similarity = potential contradiction
                # contradiction_score: higher = more contradictory
                contradiction_score = entity_overlap * (1.0 - (content_sim + 1.0) / 2.0)

                if contradiction_score >= threshold:
                    # Strip hrr_vector from output (not JSON serializable)
                    f1_clean = {k: v for k, v in f1.items() if k != "hrr_vector"}
                    f2_clean = {k: v for k, v in f2.items() if k != "hrr_vector"}
                    contradictions.append({
                        "fact_a": f1_clean,
                        "fact_b": f2_clean,
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_sim, 3),
                        "contradiction_score": round(contradiction_score, 3),
                        "shared_entities": sorted(ents1 & ents2),
                    })

        contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
        return contradictions[:limit]

    def _score_facts_by_vector(
        self,
        target_vec: "np.ndarray",
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Score facts by similarity to a target vector."""
        conn = self.store._conn

        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)

        rows = conn.execute(
            f"""
            SELECT fact_id, content, category, tags, trust_score,
                   retrieval_count, helpful_count, created_at, updated_at,
                   hrr_vector
            FROM facts
            {where}
            """,
            params,
        ).fetchall()

        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"))
            sim = hrr.similarity(target_vec, fact_vec)
            fact["score"] = (sim + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def _fts_candidates(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list[dict]:
        """Get raw FTS5 candidates from the store.

        Uses the store's database connection directly for FTS5 MATCH
        with rank scoring. Normalizes FTS5 rank to [0, 1] range.
        """
        conn = self.store._conn

        # Build query - FTS5 rank is negative (lower = better match)
        # We need to join facts_fts with facts to get all columns
        params: list = []
        where_clauses = ["facts_fts MATCH ?"]
        params.append(query)

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                   f.retrieval_count, f.helpful_count, f.created_at, f.updated_at,
                   f.hrr_vector, f.embedding_vector,
                   facts_fts.rank as fts_rank_raw
            FROM facts_fts
            JOIN facts f ON f.fact_id = facts_fts.rowid
            WHERE {where_sql}
            ORDER BY facts_fts.rank
            LIMIT ?
        """
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            # FTS5 MATCH can fail on malformed queries — fall back to empty
            return []

        if not rows:
            return []

        # Normalize FTS5 rank: rank is negative, lower = better
        # Convert to positive score in [0, 1] range
        raw_ranks = [abs(row["fts_rank_raw"]) for row in rows]
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        max_rank = max(max_rank, 1e-6)  # avoid div by zero

        results = []
        for row, raw_rank in zip(rows, raw_ranks):
            fact = dict(row)
            fact.pop("fts_rank_raw", None)
            fact["fts_rank"] = raw_rank / max_rank  # normalize to [0, 1]
            results.append(fact)

        return results

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple whitespace tokenization with lowercasing.

        Strips common punctuation. No stemming/lemmatization (Phase 1).
        """
        if not text:
            return set()
        # Split on whitespace, lowercase, strip punctuation
        tokens = set()
        for word in text.lower().split():
            cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
            if cleaned:
                tokens.add(cleaned)
        return tokens

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """Exponential decay: 0.5^(age_days / half_life_days).

        Returns 1.0 if decay is disabled or timestamp is missing.
        """
        if not self.half_life or not timestamp_str:
            return 1.0

        try:
            if isinstance(timestamp_str, str):
                # Parse ISO format timestamp from SQLite
                ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                ts = timestamp_str

            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            if age_days < 0:
                return 1.0

            return math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
class AdaptiveRetrieval(FactRetriever):
    """Extends FactRetriever with self-tuning retrieval weights.

    Learns from user feedback: when a retrieved fact is rated helpful or
    unhelpful, adjusts the four signal weights (fts, jaccard, hrr, embedding)
    to favour signals that scored high on good results and penalise signals
    that scored high on bad results.

    Weights are persisted to a JSON file co-located with the memory database
    so they survive restarts.

    Config:
        adaptive_weights_enabled: Enable weight adaptation (default False).
        adaptive_learning_rate: Step size per feedback event (default 0.01).
    """

    # Maps logical signal names to the parent class weight attributes.
    _SIGNAL_MAP: dict[str, str] = {
        "fts": "fts_weight",
        "jaccard": "jaccard_weight",
        "hrr": "hrr_weight",
        "embedding": "embedding_weight",
    }
    _WEIGHT_MIN: float = 0.05
    _WEIGHT_MAX: float = 0.60

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,
        fts_weight: float = 0.25,
        jaccard_weight: float = 0.20,
        hrr_weight: float = 0.15,
        hrr_dim: int = 1024,
        embedding_enabled: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        embedding_weight: float = 0.40,
        adaptive_weights_enabled: bool = False,
        adaptive_learning_rate: float = 0.01,
    ):
        super().__init__(
            store=store,
            temporal_decay_half_life=temporal_decay_half_life,
            fts_weight=fts_weight,
            jaccard_weight=jaccard_weight,
            hrr_weight=hrr_weight,
            hrr_dim=hrr_dim,
            embedding_enabled=embedding_enabled,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            embedding_weight=embedding_weight,
        )
        self._adaptive_enabled: bool = adaptive_weights_enabled
        self._learning_rate: float = adaptive_learning_rate
        # Snapshot the default weights (post-redistribution) for reset_weights().
        self._default_weights: dict[str, float] = self.get_weights()

        # Track which signals are actually usable (non-zero after redistribution).
        # Signals zeroed by the parent due to missing deps must not be revived.
        self._active_signals: set[str] = {
            name for name, attr in self._SIGNAL_MAP.items()
            if getattr(self, attr) > 0.0
        }

        # Derive persistence path from the database location.
        # e.g.  memory_store.db  ->  memory_store.adaptive_weights.json
        self._weights_path = (
            self.store.db_path.parent
            / (self.store.db_path.stem + ".adaptive_weights.json")
        )

        # Restore persisted weights when adaptive mode is on.
        if self._adaptive_enabled:
            self._load_persisted_weights()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weights(self) -> dict[str, float]:
        """Return the current signal weights as a dict."""
        return {
            name: getattr(self, attr)
            for name, attr in self._SIGNAL_MAP.items()
        }

    def reset_weights(self) -> None:
        """Restore weights to the original defaults computed at construction."""
        self._apply_weights(self._default_weights)
        if self._adaptive_enabled:
            self._persist_weights()

    def update_weights(
        self,
        query: str,
        selected_id: int,
        signal_scores: dict[str, float],
        feedback: bool,
    ) -> None:
        """Adjust signal weights based on retrieval feedback.

        Args:
            query: The original search query (for context / logging).
            selected_id: The fact ID of the result that was rated.
            signal_scores: Per-signal scores the selected fact received.
            feedback: True if helpful, False if unhelpful.

        Update rule: delta = learning_rate * signal_score * direction.
        Weights are clamped to [0.05, 0.60] and normalised to sum to 1.0.
        """
        if not self._adaptive_enabled:
            return

        direction = 1.0 if feedback else -1.0
        weights = self.get_weights()

        # Apply deltas -- only to signals that are actually available.
        for signal_name, score in signal_scores.items():
            if signal_name not in self._active_signals:
                continue
            if signal_name not in weights:
                continue
            delta = self._learning_rate * float(score) * direction
            weights[signal_name] += delta

        # Clamp active signals to [WEIGHT_MIN, WEIGHT_MAX].
        for name in self._active_signals:
            weights[name] = max(self._WEIGHT_MIN, min(self._WEIGHT_MAX, weights[name]))

        # Normalise active weights so they sum to 1.0.
        # Inactive signals remain at 0.0.
        active_total = sum(weights[n] for n in self._active_signals)
        if active_total > 0:
            for name in self._active_signals:
                weights[name] /= active_total

        self._apply_weights(weights)
        self._persist_weights()

        logger.debug(
            "Adaptive weights updated (feedback=%s, selected=%d): %s",
            feedback, selected_id, weights,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_weights(self, weights: dict[str, float]) -> None:
        """Push a weight dict onto the parent class attributes."""
        for name, attr in self._SIGNAL_MAP.items():
            if name in weights:
                setattr(self, attr, weights[name])

    def _load_persisted_weights(self) -> None:
        """Load adaptive weights from the JSON persistence file."""
        try:
            if not self._weights_path.exists():
                return
            with open(self._weights_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                return

            # Only adopt stored values for signals that are currently active.
            weights = self.get_weights()
            for name in self._active_signals:
                if name in data:
                    val = float(data[name])
                    weights[name] = max(self._WEIGHT_MIN, min(self._WEIGHT_MAX, val))

            # Re-normalise active weights.
            active_total = sum(weights[n] for n in self._active_signals)
            if active_total > 0:
                for name in self._active_signals:
                    weights[name] /= active_total

            self._apply_weights(weights)
            logger.debug("Loaded persisted adaptive weights: %s", weights)
        except (json.JSONDecodeError, OSError, KeyError, ValueError, TypeError) as exc:
            logger.debug("Could not load adaptive weights, using defaults: %s", exc)

    def _persist_weights(self) -> None:
        """Write current weights to the JSON persistence file."""
        try:
            self._weights_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._weights_path, "w", encoding="utf-8") as fh:
                json.dump(self.get_weights(), fh, indent=2)
        except OSError as exc:
            logger.debug("Could not persist adaptive weights: %s", exc)


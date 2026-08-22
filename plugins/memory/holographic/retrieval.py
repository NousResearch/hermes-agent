"""Hybrid keyword/BM25 retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking and trust-weighted scoring.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]


# Ranking policy constants — lexical dominance, HRR as a rerank tiebreak.
# Kept as module-level constants so the ranking policy is tunable and
# testable from a single place, rather than inline magic numbers.
_RELEVANCE_LEX_WEIGHT = 0.55   # lexical (Jaccard) weight in the blend
_RELEVANCE_HRR_WEIGHT = 0.45   # HRR structural weight in the blend
_REASON_LEX_WEIGHT = 0.6       # lexical weight in reason()'s AND blend
_REASON_HRR_WEIGHT = 0.4       # HRR weight in reason()'s AND blend


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.4,
        jaccard_weight: float = 0.3,
        hrr_weight: float = 0.3,
        hrr_dim: int = 1024,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim

        # Auto-redistribute weights if numpy unavailable
        if hrr_weight > 0 and not hrr._HAS_NUMPY:
            fts_weight = 0.6
            jaccard_weight = 0.4
            hrr_weight = 0.0

        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight

    def search(
        self,
        query: str,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
    ) -> list[dict]:
        """Hybrid search: FTS5 candidates → Jaccard rerank → trust weighting.

        Pipeline:
        1. FTS5 search: Get limit*3 candidates from SQLite full-text search
        2. Jaccard boost: Token overlap between query and fact content
        3. Trust weighting: final_score = relevance * trust_score
        4. Temporal decay (optional): decay = 0.5^(age_days / half_life)

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        """
        # Stage 1: Get FTS5 candidates (more than limit for reranking headroom)
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)

        if not candidates:
            return []

        # Stage 2: Rerank with Jaccard + trust + optional decay
        query_tokens = self._tokenize(query)
        # The query vector is loop-invariant — encode it at most once, on
        # the first candidate that actually carries an HRR vector. Lazy on
        # purpose: migrated stores can have FTS candidates whose hrr_vector
        # was never backfilled (MemoryStore._init_db adds the column
        # without backfilling), and those must not pay for an encode
        # nothing will use. encode_text is deterministic (SHA-256 counter
        # blocks), so the hoisted vector is bit-identical to what the
        # per-candidate calls produced.
        query_vec = None
        scored = []

        for fact in candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens

            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            fts_score = fact.get("fts_rank", 0.0)

            # HRR similarity
            if self.hrr_weight > 0 and fact.get("hrr_vector"):
                fact_vec = hrr.bytes_to_phases(fact["hrr_vector"], dim=self.hrr_dim)
                if query_vec is None:
                    query_vec = hrr.encode_text(query, self.hrr_dim)
                hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0  # shift to [0,1]
            else:
                hrr_sim = 0.5  # neutral

            # Combine FTS5 + Jaccard + HRR
            relevance = (self.fts_weight * fts_score
                        + self.jaccard_weight * jaccard
                        + self.hrr_weight * hrr_sim)

            # Trust weighting
            score = relevance * fact["trust_score"]

            # Optional temporal decay
            if self.half_life > 0:
                score *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

            fact["score"] = score
            scored.append(fact)

        # Sort by score descending, return top limit
        scored.sort(key=lambda x: x["score"], reverse=True)
        results = scored[:limit]
        # Strip raw HRR bytes — callers expect JSON-serializable dicts
        for fact in results:
            fact.pop("hrr_vector", None)
        return results

    def probe(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Compositional entity query.

        Uses a lexical prefilter (FTS5 + Jaccard) to surface candidate facts,
        then augments the ranking with HRR structure when available. This is
        the empirically-correct architecture: pure-HRR phase vectors are
        numerically too weak to isolate exact facts, while the lexical paths
        are precise. HRR is retained as a rerank signal, not the primary
        driver.
        """
        # Lexical prefilter (FTS5 + Jaccard), same pipeline as search()
        return self._lexical_anchor_query(entity, category=category, limit=limit, hrr_mode="probe")

    def related(
        self,
        entity: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Discover facts structurally related to an entity.

        Lexical prefilter, then scores facts by shared-context overlap.
        Where HRR is available, its structural signal augments the score;
        the lexical anchor is what makes recall reliable on terse facts.
        """
        return self._lexical_anchor_query(entity, category=category, limit=limit, hrr_mode="related")

    def reason(
        self,
        entities: list[str],
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Multi-entity compositional query.

        Facts that relate to ALL supplied entities simultaneously. Uses an
        AND intersection of per-entity lexical candidates, then a combined
        relevance score (lexical + HRR when available). This replaces the
        pure-HRR min() heuristic, which produced near-random ranking.

        NOTE: AND semantics depend on HRR being available. When HRR is
        disabled (no numpy, or hrr_weight <= 0), reason() degrades to a
        plain search(" ".join(entities)) — an OR-ish union, not an AND
        intersection. Callers relying on strict AND behavior must ensure
        HRR is enabled.
        """
        if not entities:
            return []
        if self.hrr_weight <= 0 or not hrr._HAS_NUMPY:
            # No HRR signal — plain search over the joined terms (OR-ish
            # union, NOT an AND intersection — see docstring note).
            return self.search(" ".join(entities), category=category, limit=limit)

        # Lexical candidates per entity, then intersect (AND semantics)
        per_entity = {}
        min_candidate_span = 1
        for ent in entities:
            cands = self._fts_candidates(ent, category, 0.0, limit * 3)
            per_entity[ent] = {c["fact_id"] for c in cands}
            min_candidate_span = min(min_candidate_span, len(cands))

        # If any entity has no lexical candidates, fall back to search
        if min_candidate_span == 0:
            return self.search(" ".join(entities), category=category, limit=limit)

        # Intersection: facts matching ALL entities
        intersection = set.intersection(*per_entity.values())
        if not intersection:
            return self.search(" ".join(entities), category=category, limit=limit)

        conn = self.store._conn
        where = "WHERE fact_id IN (%s)" % ",".join("?" * len(intersection))
        if category:
            where += f" AND category = ?"
            params: list = [*intersection, category]
        else:
            params = [*intersection]
        rows = conn.execute(
            f"SELECT fact_id, content, category, tags, trust_score, "
            f"retrieval_count, helpful_count, created_at, updated_at, hrr_vector FROM facts {where}",
            params,
        ).fetchall()
        if not rows:
            return self.search(" ".join(entities), category=category, limit=limit)

        # Recompute the combined score for the intersection facts
        query_tokens = set().union(*[self._tokenize(e) for e in entities]) if entities else set()
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
        probe_keys = []
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        for e in entities:
            ev = hrr.encode_atom(e.lower(), self.hrr_dim)
            probe_keys.append(hrr.bind(ev, role_entity))

        scored = []
        for row in rows:
            fact = dict(row)
            fv = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)
            # HRR structural score: proximity of residual to content role.
            # reason() has AND semantics, so ALL supplied entities must
            # contribute; aggregate across every probe key and use the weakest
            # (min) entity match so the structural score is entity-order-
            # invariant and cannot be gamed by listing a matched entity first.
            hrr_sim = 0.5
            if hrr._HAS_NUMPY and self.hrr_weight > 0:
                entity_hrr_scores = []
                for probe_key in probe_keys:
                    residual = hrr.unbind(fv, probe_key)
                    entity_hrr_scores.append(
                        (hrr.similarity(residual, role_content) + 1.0) / 2.0
                    )
                hrr_sim = min(entity_hrr_scores)

            # Lexical relevance: Jaccard between query entities and fact
            ft = self._tokenize(fact["content"]) | self._tokenize(fact.get("tags", ""))
            jac = self._jaccard_similarity(query_tokens, ft)
            # Blend: lexical dominant, HRR as tiebreak (matching search philosophy)
            relevance = (_REASON_LEX_WEIGHT * jac + _REASON_HRR_WEIGHT * hrr_sim)
            fact["score"] = relevance * fact["trust_score"]
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        # Strip raw HRR bytes — callers expect JSON-serializable dicts
        for fact in scored[:limit]:
            fact.pop("hrr_vector", None)
        return scored[:limit]

    def _lexical_anchor_query(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
        hrr_mode: str = "probe",
    ) -> list[dict]:
        """Shared lexical-prefilter + HRR-augment pipeline for probe/related.

        Uses the proven FTS5+Jaccard prefilter (which ranks ground-truth facts
        correctly on this corpus) then augments with HRR structure when
        numpy is available. Falls back to pure search() if numpy is missing.
        """
        if self.hrr_weight <= 0 or not hrr._HAS_NUMPY:
            # No HRR enhancement — plain lexical search is the reliable path
            return self.search(query, category=category, limit=limit)

        # Lexical prefilter
        cands = self._fts_candidates(query, category, 0.0, limit * 3)
        if not cands:
            # Preserve probe/related's historical fallback contract.
            # search() currently shares the same FTS candidate source, but
            # keeping the fallback avoids coupling these retrieval paths
            # if they diverge later.
            return self.search(query, category=category, limit=limit)

        query_tokens = self._tokenize(query)
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
        entity_vec = hrr.encode_atom(query.lower(), self.hrr_dim)

        scored = []
        for f in cands:
            fact = dict(f)
            ft = self._tokenize(fact["content"]) | self._tokenize(fact.get("tags", ""))
            jac = self._jaccard_similarity(query_tokens, ft)
            # HRR structural signal (probe: entity role; related: any role)
            hrr_score = 0.5  # neutral
            if fact.get("hrr_vector"):
                fv = hrr.bytes_to_phases(fact["hrr_vector"], dim=self.hrr_dim)
                if hrr_mode == "probe":
                    residual = hrr.unbind(fv, hrr.bind(entity_vec, role_entity))
                    sim = hrr.similarity(residual, role_content)
                    hrr_score = (sim + 1.0) / 2.0
                else:  # related
                    residual = hrr.unbind(fv, entity_vec)
                    best = max(hrr.similarity(residual, role_entity),
                               hrr.similarity(residual, role_content))
                    hrr_score = (best + 1.0) / 2.0
            # Lexical dominant, HRR as tiebreak — NOT the primary signal
            relevance = _RELEVANCE_LEX_WEIGHT * jac + _RELEVANCE_HRR_WEIGHT * hrr_score
            # Trust weighting: linear, same formula as search() and reason()
            relevance *= fact["trust_score"]
            fact["score"] = relevance
            scored.append(fact)

        scored.sort(key=lambda x: x["score"], reverse=True)
        # Strip raw HRR bytes — callers expect JSON-serializable dicts
        for fact in scored[:limit]:
            fact.pop("hrr_vector", None)
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
                v1 = hrr.bytes_to_phases(f1["hrr_vector"], dim=self.hrr_dim)
                v2 = hrr.bytes_to_phases(f2["hrr_vector"], dim=self.hrr_dim)
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
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)
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
        # FTS5 defaults to AND-between-tokens, which kills recall on
        # natural-language queries ("what happened with the deployment
        # rollback"). Sanitize: drop stopwords, OR-join content tokens, so
        # any significant term can match.
        params.append(self._sanitize_fts_query(query))

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.*, facts_fts.rank as fts_rank_raw
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

    # Stopwords dropped before FTS5 OR-expansion. Short English function
    # words that carry no retrieval signal and force false-negative AND
    # matches when left in the query.
    _FTS_STOPWORDS = frozenset({
        "a", "about", "above", "after", "again", "all", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being",
        "between", "both", "but", "by", "can", "could", "did", "do", "does",
        "doing", "don", "down", "during", "each", "few", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here",
        "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
        "into", "is", "it", "its", "itself", "just", "me", "more", "most",
        "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over",
        "own", "same", "she", "should", "so", "some", "such", "than", "that",
        "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under",
        "until", "up", "very", "was", "we", "were", "what", "when", "where",
        "which", "while", "who", "whom", "why", "will", "with", "would",
        "you", "your", "yours", "yourself", "yourselves",
    })

    @classmethod
    def _sanitize_fts_query(cls, query: str) -> str:
        """Convert a natural-language query to an FTS5-safe OR expression.

        FTS5 treats a multi-word MATCH argument as AND-joined by default,
        which tanks recall on prose queries. This helper:
          - tokenizes the query
          - drops stopwords and short (<2 char) tokens
          - strips FTS5 special characters from each token
          - OR-joins the survivors

        If nothing remains (pathological query), falls back to the raw
        query so the caller sees zero results instead of a SQL error.
        """
        if not query:
            return ""
        # Strip FTS5 operator characters from EACH token to avoid
        # accidentally creating a malformed query.
        _FTS_SPECIAL = '"()*^:-+'
        tokens: list[str] = []
        for raw in query.lower().split():
            cleaned = raw.strip(".,;:!?\"'()[]{}#@<>") .translate(
                str.maketrans("", "", _FTS_SPECIAL)
            )
            if len(cleaned) < 2:
                continue
            if cleaned in cls._FTS_STOPWORDS:
                continue
            # FTS5 phrase-literal each token to ensure no special chars
            # sneak through as operators.
            tokens.append(f'"{cleaned}"')
        if not tokens:
            # Fallback: raw query (likely returns 0, but never crashes)
            return query
        return " OR ".join(tokens)

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

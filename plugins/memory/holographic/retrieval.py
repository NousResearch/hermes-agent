"""Hybrid keyword/BM25 retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking and trust-weighted scoring.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]


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
        2. If FTS5 returns < limit candidates, supplement with Jaccard scan
           of remaining facts (handles cross-sentence CJK queries).
        3. Rerank with Jaccard + trust + optional HRR + temporal decay.

        Returns list of dicts with fact data + 'score' field, sorted by score desc.
        """
        # Stage 1: Get FTS5 candidates (more than limit for reranking headroom)
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)

        # Stage 1b: FTS5 undershoot — supplement with Jaccard fallback.
        if len(candidates) < limit:
            existing_ids = {c["fact_id"] for c in candidates} if candidates else set()
            needed = (limit * 3) - len(candidates)
            jaccard_extras = self._jaccard_fallback(
                query, category, min_trust, needed, exclude_ids=existing_ids,
            )
            if candidates:
                candidates.extend(jaccard_extras)
            else:
                candidates = jaccard_extras

        # Stage 1b: FTS5 undershoot — supplement with Jaccard fallback.
        # Ordered bigram phrases can miss cross-sentence CJK matches
        # (e.g. "한글입니다" vs "한글은 ...입니다") where intervening
        # characters break bigram contiguity.  Jaccard handles these
        # because it's order-independent set overlap.
        if len(candidates) < limit:
            existing_ids = {c["fact_id"] for c in candidates} if candidates else set()
            needed = (limit * 3) - len(candidates)
            jaccard_extras = self._jaccard_fallback(
                query, category, min_trust, needed, exclude_ids=existing_ids,
            )
            if candidates:
                candidates.extend(jaccard_extras)
            else:
                candidates = jaccard_extras

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
                bank_vec = hrr.bytes_to_phases(bank_row["vector"], dim=self.hrr_dim)
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

        # role_content is loop-invariant — encode it once (deterministic
        # SHA-256-based atom) instead of once per fact row.
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)
            # Unbind probe key from fact to see if entity is structurally present
            residual = hrr.unbind(fact_vec, probe_key)
            # Compare residual against content signal
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
        # Both role atoms are loop-invariant — encode them once here
        # (deterministic SHA-256-based atoms) instead of twice per fact row.
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)

            # Check structural similarity: unbind entity from fact
            residual = hrr.unbind(fact_vec, entity_vec)
            # A high-similarity residual to ANY known role vector means this entity
            # plays a structural role in the fact

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
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)

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

    def _jaccard_fallback(
        self,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
        exclude_ids: set[int] | None = None,
    ) -> list[dict]:
        """Jaccard-only candidate retrieval when FTS5 undershoots.

        FTS5 ordered bigram phrases can miss cross-sentence CJK matches
        where intervening characters break bigram contiguity.  This
        fallback scans facts that FTS5 didn't return and ranks them by
        pure Jaccard token overlap — order-independent, so ``한글입니다``
        can match ``한글은 ...입니다`` across sentence boundaries.

        Cost: ~13 µs / fact (tokenize + set intersection).  For a
        typical 1 000-fact store this is ~13 ms — triggered only when
        FTS5 returns fewer than ``limit`` candidates.
        """
        exclude_ids = exclude_ids or set()
        conn = self.store._conn

        params: list = []
        wheres: list[str] = ["1=1"]

        if category:
            wheres.append("category = ?")
            params.append(category)

        wheres.append("trust_score >= ?")
        params.append(min_trust)

        if exclude_ids:
            placeholders = ",".join("?" for _ in exclude_ids)
            wheres.append(f"fact_id NOT IN ({placeholders})")
            params.extend(exclude_ids)

        sql = f"SELECT * FROM facts WHERE {' AND '.join(wheres)}"
        rows = conn.execute(sql, params).fetchall()

        if not rows:
            return []

        query_tokens = self._tokenize(query)

        scored: list[tuple[float, dict]] = []
        for row in rows:
            fact = dict(row)
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens

            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            if jaccard <= 0:
                continue

            # Normalize Jaccard into the same [0,1] range as fts_rank
            fact["fts_rank"] = jaccard
            scored.append((jaccard, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for _, fact in scored[:limit]]
    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text into a set of indexable terms.

        English/ASCII: whitespace-split, lowercased, punctuation-stripped.
        CJK (Chinese/Japanese/Korean): character 2-grams + standalone
        alphanumeric substrings extracted from CJK runs.

        This hybrid approach ensures both English keywords and Chinese
        phrases are searchable with Jaccard overlap — without requiring
        a segmenter or dictionary.
        """
        if not text:
            return set()
        # NFC-normalize: macOS pastes Hangul as NFD (e.g. 한 for 한),
        # which would produce different bigrams than the indexed NFC form.
        text = unicodedata.normalize("NFC", text)
        tokens: set[str] = set()
        # ── CJK 2-gram extraction ──
        # Split into alternating runs: ASCII-or-digit spans vs CJK spans.
        import re as _re
        cjk = _re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+")
        prev_end = 0
        for m in cjk.finditer(text):
            # Emit ASCII tokens before this CJK span (whitespace-split)
            ascii_span = text[prev_end : m.start()]
            for word in ascii_span.lower().split():
                cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
                if cleaned:
                    tokens.add(cleaned)
            # CJK run → character 2-grams
            run = m.group()
            for i in range(len(run) - 1):
                tokens.add(run[i : i + 2])
            # Also add single characters for short matches
            for ch in run:
                tokens.add(ch)
            prev_end = m.end()
        # Remaining ASCII after last CJK run
        for word in text[prev_end:].lower().split():
            cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
            if cleaned:
                tokens.add(cleaned)
        return tokens
    # CJK ranges for FTS5 pre-tokenization — same ranges as _tokenize()
    _CJK_RE = re.compile(
        r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
        r"\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+"
    )

    # Regex for splitting ASCII tokens on letter↔digit and underscore
    # boundaries.  "Python3.12" → "Python 3 12", "snake_case" → "snake case".
    # Does NOT split camelCase (would break DeepSeek, GitHub, etc.).
    _ASCII_BOUNDARY_RE = re.compile(r"([a-zA-Z])(\d)|(\d)([a-zA-Z])|_")

    @staticmethod
    def _split_ascii_for_fts(text: str) -> str:
        """Split ASCII text on letter-digit and underscore boundaries.

        FTS5's unicode61 treats ``Python3.12`` as one token.  Without
        splitting, a query for ``Python`` cannot match it.  This helper
        inserts spaces at boundaries so both the index and query sides
        see ``Python``, ``3``, ``12`` as independent tokens.
        """
        # letter→digit  and  digit→letter
        text = FactRetriever._ASCII_BOUNDARY_RE.sub(
            lambda m: (m.group(1) or m.group(3)) + " " + (m.group(2) or m.group(4))
            if m.group(1) or m.group(3) else " ",
            text,
        )
        return text

    @classmethod
    def _cjk_to_fts5_bigrams(cls, text: str) -> str:
        """Pre-tokenize CJK text for FTS5: single chars + character 2-grams.

        SQLite FTS5's ``unicode61`` tokenizer treats a continuous run of CJK
        characters as ONE indivisible token — a query for a substring (e.g.
        ``배포`` inside ``배포는``, or ``刚才`` inside ``小明刚才说了什么``)
        never matches.  This helper inserts spaces between character 2-grams
        AND individual characters so FTS5 indexes both.  Single-character
        tokens are needed so single-CJK-char queries (``书``, ``口``, ``日``)
        match facts where the character appears only inside longer bigrams.

        ASCII spans pass through unchanged.
        """
        if not text:
            return text
        # NFC-normalize: ensures the FTS5 index always stores composed
        # forms, matching the normalization applied on the query side.
        text = unicodedata.normalize("NFC", text)
        prev_end = 0
        parts: list[str] = []
        for m in cls._CJK_RE.finditer(text):
            if m.start() > prev_end:
                # Split ASCII span at letter-digit / underscore boundaries
                # so "Python3.12" → "Python 3 12" tokens in FTS5 index.
                parts.append(cls._split_ascii_for_fts(text[prev_end : m.start()]))
            run = m.group()
            if len(run) >= 2:
                # Emit single characters AND 2-grams so both
                # single-char and multi-char queries match.
                chars = [run[i] for i in range(len(run))]
                bigrams = [run[i : i + 2] for i in range(len(run) - 1)]
                parts.append(" " + " ".join(chars + bigrams) + " ")
            else:
                parts.append(" " + run + " ")
            prev_end = m.end()
        if prev_end < len(text):
            parts.append(cls._split_ascii_for_fts(text[prev_end:]))
        return "".join(parts).strip()
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

          - Splits CJK runs into character 2-gram prefix tokens so they
            match bigram-spaced FTS5 content (see _cjk_to_fts5_bigrams).
          - For ASCII spans: tokenizes, drops stopwords and short tokens,
            strips FTS5 special characters.
          - OR-joins all tokens.

        If nothing remains, falls back to the raw query so the caller sees
        zero results instead of a SQL error.
        """
        if not query:
            return ""
        # NFC-normalize: macOS pastes Hangul as NFD, which would
        # produce queries that don't match the NFC-indexed content.
        query = unicodedata.normalize("NFC", query)
        _FTS_SPECIAL = '\"()*^:+'  # NOTE: '-' deliberately kept — safe inside phrase
        # literals and needed for hyphenated terms (bong-u, verify-patches, etc.)
        tokens: list[str] = []

        # Split into alternating CJK / non-CJK spans
        prev_end = 0
        for m in cls._CJK_RE.finditer(query):
            # ASCII span before this CJK run — split at letter-digit and
            # underscore boundaries so "Python3.12" → "Python 3 12".
            ascii_span = cls._split_ascii_for_fts(query[prev_end : m.start()])
            for raw in ascii_span.lower().split():
                cleaned = raw.strip(".,;:!?\"'()[]{}#@<>").translate(
                    str.maketrans("", "", _FTS_SPECIAL)
                )
                if len(cleaned) >= 1 and cleaned not in cls._FTS_STOPWORDS:
                    # Prefix match for short ASCII tokens (≤6 chars):
                    # "python"* matches Python3, Python3.12, python2.7, etc.
                    if len(cleaned) <= 6 and cleaned.isascii() and cleaned.isalpha():
                        tokens.append(f'"{cleaned}"*')
                    else:
                        tokens.append(f'"{cleaned}"')
            # CJK run → ordered bigram phrase so FTS5 matches the
            # run as a contiguous span.  OR-ing bigrams would
            # collapse precision on languages where a single
            # bigram (e.g. Korean ``한다``) terminates nearly
            # every declarative verb.
            run = m.group()
            if len(run) >= 2:
                bigrams = [run[i : i + 2] for i in range(len(run) - 1)]
                tokens.append('"' + " ".join(bigrams) + '"')
            elif len(run) == 1:
                tokens.append(f'"{run}"')
            prev_end = m.end()

        # Trailing ASCII after last CJK run — split boundaries + prefix
        trailing = cls._split_ascii_for_fts(query[prev_end:])
        for raw in trailing.lower().split():
            cleaned = raw.strip(".,;:!?\"'()[]{}#@<>").translate(
                str.maketrans("", "", _FTS_SPECIAL)
            )
            if len(cleaned) >= 1 and cleaned not in cls._FTS_STOPWORDS:
                if len(cleaned) <= 6 and cleaned.isascii() and cleaned.isalpha():
                    tokens.append(f'"{cleaned}"*')
                else:
                    tokens.append(f'"{cleaned}"')

        if not tokens:
            # Fallback: quote the raw query so FTS5 never sees bare
            # operator characters that would cause a parse error.
            return f'"{query}"'
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

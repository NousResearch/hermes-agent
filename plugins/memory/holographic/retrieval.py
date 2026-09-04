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

        # CJK short-token fallback: the trigram tokenizer cannot match
        # queries shorter than 3 CJK characters (e.g. "芯片", "智谱").
        # When the query contains CJK and MATCH returned nothing, retry
        # with substring LIKE on content+tags — the same production-proven
        # pattern the CLS news database uses for 2-char Chinese queries.
        if not rows and self._has_cjk(query):
            rows = self._like_fallback(conn, query, category, min_trust, limit)

        # CJK supplement: a query mixing 3+ char runs with 2-char runs
        # ("智谱 港股代码") produces a non-empty MATCH result that silently
        # drops the 2-char signal — merge LIKE hits for those runs into the
        # candidate pool so the rerank sees them (dedup by fact_id).
        sub_runs = self._sub_trigram_cjk_runs(query)
        if sub_runs and self._has_cjk(query):
            seen = {r["fact_id"] for r in rows}
            for extra in self._like_fallback_runs(conn, sub_runs, category, min_trust, limit):
                if extra["fact_id"] not in seen:
                    rows = list(rows) + [extra]
                    seen.add(extra["fact_id"])

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

    def _like_fallback(
        self,
        conn,
        query: str,
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list:
        """Substring fallback for short CJK tokens (<3 chars) that trigram
        FTS5 cannot index. Scans content+tags with LIKE; rank by number of
        matched substrings so the caller's rerank stages still see signal.
        """
        # Extract CJK runs from the sanitized query
        runs: list[str] = []
        current: list[str] = []
        for ch in query:
            if self._is_cjk_char(ch):
                current.append(ch)
            else:
                if current:
                    runs.append("".join(current))
                    current = []
        if current:
            runs.append("".join(current))
        if not runs:
            return []

        # Use the longest run as the primary LIKE pattern (most specific)
        primary = max(runs, key=len)
        like_clause = " OR ".join(
            ["(f.content LIKE ? ESCAPE '\\')", "(f.tags LIKE ? ESCAPE '\\')"]
        )
        pattern = f"%{primary.replace('%', chr(92) + '%')}%"

        sql = f"""
            SELECT f.*, 1.0 as fts_rank_raw
            FROM facts f
            WHERE ({like_clause})
        """
        params: list = [pattern, pattern]
        if category:
            sql += " AND f.category = ?"
            params.append(category)
        sql += " AND f.trust_score >= ?"
        params.append(min_trust)
        sql += " ORDER BY f.updated_at DESC LIMIT ?"
        params.append(limit)

        try:
            return conn.execute(sql, params).fetchall()
        except Exception:
            return []

    def _like_fallback_runs(
        self,
        conn,
        runs: list[str],
        category: str | None,
        min_trust: float,
        limit: int,
    ) -> list:
        """LIKE supplement for explicit 2-char CJK runs. Each run gets its
        own scan; results are capped per run to keep a common 2-char word
        from flooding the candidate pool."""
        out: list = []
        for run in runs[:4]:  # cap runs considered
            pattern = f"%{run}%"
            sql = """
                SELECT f.*, 0.5 as fts_rank_raw
                FROM facts f
                WHERE (f.content LIKE ? ESCAPE '\\' OR f.tags LIKE ? ESCAPE '\\')
            """
            params: list = [pattern, pattern]
            if category:
                sql += " AND f.category = ?"
                params.append(category)
            sql += " AND f.trust_score >= ? ORDER BY f.updated_at DESC LIMIT ?"
            params.extend([min_trust, max(limit, 10)])
            try:
                out.extend(conn.execute(sql, params).fetchall())
            except Exception:
                continue
        return out

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Whitespace tokenization with lowercasing + CJK bigram expansion.

        Strips common punctuation. No stemming/lemmatization (Phase 1).
        For CJK text (no whitespace boundaries), falls back to character
        bigrams so Jaccard reranking sees Chinese signal — mirrors the
        SimHash trigram approach used by the CLS news pipeline.
        """
        if not text:
            return set()
        tokens: set[str] = set()
        for word in text.lower().split():
            cleaned = word.strip(".,;:!?\"'()[]{}#@<>")
            if not cleaned:
                continue
            cjk_chars = [ch for ch in cleaned if FactRetriever._is_cjk_char(ch)]
            non_cjk = "".join(ch for ch in cleaned if not FactRetriever._is_cjk_char(ch))
            # Keep latin/digit runs as whole tokens
            for part in non_cjk.replace("/", " ").replace("=", " ").split():
                if part:
                    tokens.add(part)
            # CJK: character bigrams (unigrams too for 1-char entities)
            if cjk_chars:
                run = "".join(cjk_chars)
                if len(run) == 1:
                    tokens.add(run)
                else:
                    tokens.update(run[i:i+2] for i in range(len(run) - 1))
        return tokens

    @classmethod
    def _is_cjk_char(cls, ch: str) -> bool:
        return any(lo <= ord(ch) <= hi for lo, hi in cls._CJK_RANGES)

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

    # --- CJK support -------------------------------------------------------
    # The trigram tokenizer indexes overlapping 3-char windows, so a query
    # shorter than 3 CJK characters cannot match anything through FTS5
    # ("芯片" -> 0 hits). CJK detection + a LIKE fallback mirror the
    # production-proven pattern used by the CLS news database: >=3 char
    # tokens go through MATCH, shorter ones through substring LIKE.
    _CJK_RANGES = (
        (0x4E00, 0x9FFF),    # CJK Unified Ideographs
        (0x3400, 0x4DBF),    # Extension A
        (0xF900, 0xFAFF),    # Compatibility Ideographs
    )

    @classmethod
    def _has_cjk(cls, text: str) -> bool:
        if not text:
            return False
        return any(
            any(lo <= ord(ch) <= hi for lo, hi in cls._CJK_RANGES)
            for ch in text
        )

    @staticmethod
    def _split_cjk_run(run: str) -> list[str]:
        """Split a punctuation-free CJK run into trigram-queryable pieces.

        For runs >=3 chars, a single token works (its 3-char windows overlap
        the indexed windows). For 2-char runs we keep the pair verbatim —
        the LIKE fallback handles it. Mixed CJK+latin runs are split so the
        latin part keeps working through the normal tokenizer.
        """
        if len(run) >= 3:
            return [run]
        return [run] if run else []

    @classmethod
    def _sub_trigram_cjk_runs(cls, query: str) -> list[str]:
        """2-char CJK runs in the query — impossible to match through the
        trigram FTS index, so callers must supplement with LIKE."""
        runs: list[str] = []
        current: list[str] = []
        for ch in query:
            if cls._is_cjk_char(ch):
                current.append(ch)
            else:
                if len(current) == 2:
                    runs.append("".join(current))
                current = []
        if len(current) == 2:
            runs.append("".join(current))
        return runs

    @classmethod
    def _sanitize_fts_query(cls, query: str) -> str:
        """Convert a natural-language query to an FTS5-safe OR expression.

        FTS5 treats a multi-word MATCH argument as AND-joined by default,
        which tanks recall on prose queries. This helper:
          - tokenizes the query
          - drops stopwords and short (<2 char) tokens
          - strips FTS5 special characters from each token
          - OR-joins the survivors

        For CJK-containing queries, the trigram tokenizer can only match
        contiguous 3-char windows, so a whole natural-language question
        almost never appears verbatim in a document. CJK text is therefore
        expanded into overlapping 3-char shingles and OR-joined — any
        shingle hit retrieves the fact, and the Jaccard rerank (with CJK
        bigrams) orders the candidates. Queries whose CJK runs are all
        shorter than 3 chars return "" so the caller's LIKE fallback runs.

        If nothing remains (pathological query), falls back to the raw
        query so the caller sees zero results instead of a SQL error.
        """
        if not query:
            return ""

        # CJK path: shingle expansion
        if cls._has_cjk(query):
            cjk_runs: list[str] = []
            current: list[str] = []
            for ch in query:
                if cls._is_cjk_char(ch):
                    current.append(ch)
                else:
                    if current:
                        cjk_runs.append("".join(current))
                        current = []
            if current:
                cjk_runs.append("".join(current))

            shingles: set[str] = set()
            sub_trigram_runs: list[str] = []
            latin_tokens: list[str] = []
            for run in cjk_runs:
                if len(run) >= 3:
                    shingles.update(run[i:i+3] for i in range(len(run) - 2))
                elif len(run) == 2:
                    # Trigram indexes only 3-char windows, so a 2-char run
                    # can NEVER match through FTS5 (phrase or bare). It is
                    # routed to the LIKE supplement in _fts_candidates.
                    sub_trigram_runs.append(run)
                # len(run) == 1: single char, too noisy — skip
            # latin/digit tokens in a mixed query still go through the
            # normal phrase-literal path. Keep hyphens: trigram indexes
            # them verbatim and stripping them breaks exact-term recall
            # ("daemon-reexec" -> "daemonreexec" matches nothing).
            for raw in query.split():
                cleaned = raw.strip(".,;:!?\"'()[]{}#@<>").translate(
                    str.maketrans("", "", '"()*^:+'))
                if len(cleaned) >= 2 and cleaned not in cls._FTS_STOPWORDS \
                        and not cls._has_cjk(cleaned):
                    latin_tokens.append(f'"{cleaned}"')

            or_parts = [f'"{s}"' for s in shingles] + latin_tokens
            if or_parts:
                return " OR ".join(or_parts)
            # all CJK runs < 3 chars → let LIKE fallback handle it
            return query

        # Latin-only path (original behaviour)
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

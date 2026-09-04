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

        # Cache trigram-tokenizer availability: depends only on the SQLite
        # build, which never changes at runtime. The probe (CREATE/DROP DDL)
        # runs at most once instead of on every search.
        self._trigram_available_cache: bool | None = None

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

            # Trigram window-hit boost: when candidates came from the trigram
            # path, facts sharing more 3-char windows with the query are more
            # relevant (window_hits added by _fts_candidates).  Bias toward
            # the query's total window count so a fact hitting all windows
            # ranks above one hitting a single window.
            window_hits = fact.get("window_hits", 0)
            if window_hits:
                total_windows = len(self._window_tokens(query))
                if total_windows:
                    relevance += window_hits / total_windows

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

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        """True when text contains CJK (Chinese/Japanese/Korean) chars.

        Same ranges used by L3 session search (hermes_state_search.py):
        CJK Unified Ideographs, Extension A/B, CJK Symbols.
        """
        for ch in text:
            cp = ord(ch)
            if (0x4E00 <= cp <= 0x9FFF or    # CJK Unified Ideographs
                    0x3400 <= cp <= 0x4DBF or    # CJK Extension A
                    0x20000 <= cp <= 0x2A6DF or  # CJK Extension B
                    0x3000 <= cp <= 0x303F):     # CJK Symbols
                return True
        return False

    def _trigram_available(self) -> bool:
        """True when this SQLite build has the FTS5 trigram tokenizer.

        Cached: the probe (CREATE/DROP DDL on the live connection) is
        expensive and the build never changes at runtime, so it runs at most
        once per FactRetriever instance.
        """
        if self._trigram_available_cache is not None:
            return self._trigram_available_cache
        try:
            conn = self.store._conn
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS _tgram_probe USING fts5(x, tokenize='trigram')"
            )
            conn.execute("DROP TABLE IF EXISTS _tgram_probe")
            self._trigram_available_cache = True
        except Exception:
            self._trigram_available_cache = False
        return self._trigram_available_cache

    @staticmethod
    def _or_tokens_trigram(query: str) -> str:
        """Split a query into overlapping 3-char tokens, OR-joined.

        The trigram tokenizer indexes overlapping 3-char sequences for BOTH
        CJK and Latin text ("deployment" → "dep"/"epl"/"loy"/...), so a
        natural-language query like "VPS内存多大" as a whole won't appear
        verbatim in a fact, but its 3-char windows overlap with fact content.
        OR-join so any window hit returns candidates, then Jaccard rerank in
        `search()` picks the best.  Latin runs are windowed too: keeping a
        whole word such as "deployment" would produce a MATCH token the
        trigram index does not contain, silently dropping recall for that
        term in mixed CJK/English queries.
        """
        q = query.strip()
        if not q:
            return q
        tokens = []
        i = 0
        n = len(q)
        while i < n:
            # collect a run: CJK run windowed, else latin/digit run windowed
            j = i
            while j < n and (0x4E00 <= ord(q[j]) <= 0x9FFF or 0x3000 <= ord(q[j]) <= 0x303F):
                j += 1
            if j > i:
                run = q[i:j]
                for k in range(len(run) - 2):
                    tokens.append(run[k:k + 3])
                i = j
            else:
                j = i
                while j < n and not (0x4E00 <= ord(q[j]) <= 0x9FFF or 0x3000 <= ord(q[j]) <= 0x303F):
                    j += 1
                w = q[i:j]
                for k in range(len(w) - 2):
                    tokens.append(w[k:k + 3])
                i = j
        if not tokens:
            return q
        # Dedupe, keep order, drop tokens that cannot match (len<3 runs)
        seen = set()
        out = []
        for t in tokens:
            if t in seen:
                continue
            seen.add(t)
            out.append(f'"{t}"')
        return " OR ".join(out)

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

        # CJK (Chinese/Japanese/Korean) queries route to the trigram index
        # (2026-08-27 patch): unicode61 tokenizes CJK into single chars, so
        # natural-language Chinese queries never match phrases.  The trigram
        # tokenizer indexes overlapping 3-char sequences, giving substring
        # match for any >=3-char query.  Same pattern as L3 session search
        # (messages_fts_trigram in hermes_state_common.py).
        cjk = self._contains_cjk(query)
        if cjk and self._trigram_available():
            table = "facts_fts_trigram"
            # trigram does substring match on >=3-char tokens.  For
            # natural-language queries ("VPS内存多大"), OR-join the
            # 3-char sliding-window tokens so any overlapping phrase can
            # match, instead of requiring the whole sentence verbatim.
            match_arg = self._or_tokens_trigram(query)
        else:
            table = "facts_fts"
            # FTS5 defaults to AND-between-tokens, which kills recall on
            # natural-language queries ("what happened with the deployment
            # rollback"). Sanitize: drop stopwords, OR-join content tokens, so
            # any significant term can match.
            match_arg = self._sanitize_fts_query(query)

        params: list = []
        where_clauses = [f"{table} MATCH ?"]
        params.append(match_arg)

        if category:
            where_clauses.append("f.category = ?")
            params.append(category)

        where_clauses.append("f.trust_score >= ?")
        params.append(min_trust)

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT f.*, {table}.rank as fts_rank_raw
            FROM {table}
            JOIN facts f ON f.fact_id = {table}.rowid
            WHERE {where_sql}
            ORDER BY {table}.rank
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

        # For the trigram path, augment ranking with per-fact window-token
        # hit count: count how many of the query's 3-char windows appear in
        # this fact's content.  More hits = more relevant (natural-language
        # queries rarely match verbatim, so this beats raw FTS rank).
        if cjk and self._trigram_available():
            windows = self._window_tokens(query)
            out = []
            for row in rows:
                fact = dict(row)
                fact.pop("fts_rank_raw", None)
                content = fact.get("content", "")
                if windows:
                    hits = sum(1 for w in windows if w in content)
                    fact["window_hits"] = hits
                    # fts_rank keeps its normalized position as a tiebreak
                    fact["fts_rank"] = abs(row["fts_rank_raw"]) / max(
                        max(abs(r["fts_rank_raw"]) for r in rows), 1e-6
                    )
                else:
                    fact["window_hits"] = 0
                    fact["fts_rank"] = 0.5
                out.append(fact)
            return out

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
    def _window_tokens(query: str) -> list[str]:
        """Return the overlapping 3-char windows of a query (for ranking).

        Mirrors `_or_tokens_trigram`'s tokenization (CJK and Latin runs both
        windowed) but returns raw window strings (no quoting / OR) for
        content-hit counting.  Keeping both in sync matters: the MATCH tokens
        and the ranking windows must come from the same splitter or a mixed
        CJK/English query's window_hits diverge from what actually matched.
        """
        q = query.strip()
        tokens: list[str] = []
        i = 0
        n = len(q)
        while i < n:
            j = i
            while j < n and (0x4E00 <= ord(q[j]) <= 0x9FFF or 0x3000 <= ord(q[j]) <= 0x303F):
                j += 1
            if j > i:
                run = q[i:j]
                for k in range(len(run) - 2):
                    tokens.append(run[k:k + 3])
                i = j
            else:
                j = i
                while j < n and not (0x4E00 <= ord(q[j]) <= 0x9FFF or 0x3000 <= ord(q[j]) <= 0x303F):
                    j += 1
                w = q[i:j]
                for k in range(len(w) - 2):
                    tokens.append(w[k:k + 3])
                i = j
        seen = set()
        return [t for t in tokens if not (t in seen or seen.add(t))]

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

"""Hybrid keyword/BM25 retrieval for the memory store: FTS5 candidates reranked with
Jaccard similarity and HRR vector similarity, trust-weighted (ported from KIK memory_agent.py)."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

from . import holographic as hrr

_FACT_COLUMNS = "fact_id, content, category, tags, trust_score, retrieval_count, helpful_count, created_at, updated_at"
_ROLE_ENTITY, _ROLE_CONTENT = hrr.ROLE_ENTITY, hrr.ROLE_CONTENT
_PUNCT = ".,;:!?\"'()[]{}#@<>"
_FTS_OPERATORS = str.maketrans("", "", '"()*^:-+')
# Stopwords dropped before FTS5 OR-expansion: short English function words that
# carry no retrieval signal and force false-negative AND matches.
_FTS_STOPWORDS = frozenset("""
    a about above after again all am an and any are as at be because been before being between both but by can could
    did do does doing don down during each few for from further had has have having he her here hers herself him himself
    his how i if in into is it its itself just me more most my myself no nor not now of off on once only or other our
    ours ourselves out over own same she should so some such than that the their theirs them themselves then there these
    they this those through to too under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves""".split())


def _shift(sim: float) -> float:
    """Cosine similarity [-1, 1] -> [0, 1]."""
    return (sim + 1.0) / 2.0


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(self, store: MemoryStore, temporal_decay_half_life: int = 0,  # days, 0 = disabled
                 fts_weight: float = 0.4, jaccard_weight: float = 0.3, hrr_weight: float = 0.3, hrr_dim: int = 1024):
        self.store, self.half_life, self.hrr_dim = store, temporal_decay_half_life, hrr_dim
        if hrr_weight > 0 and not hrr._HAS_NUMPY:  # redistribute weights without numpy
            fts_weight, jaccard_weight, hrr_weight = 0.6, 0.4, 0.0
        self.fts_weight, self.jaccard_weight, self.hrr_weight = fts_weight, jaccard_weight, hrr_weight
        # Cache trigram-tokenizer availability: depends only on the SQLite
        # build, which never changes at runtime. The probe (CREATE/DROP DDL)
        # runs at most once instead of on every search.
        self._trigram_available_cache: bool | None = None

    def _atom(self, word: str):
        return hrr.encode_atom(word, self.hrr_dim)

    def _phases(self, blob: bytes):
        return hrr.bytes_to_phases(blob, dim=self.hrr_dim)

    def search(self, query: str, category: str | None = None, min_trust: float = 0.3, limit: int = 10) -> list[dict]:
        """FTS5 candidates (limit*3) → Jaccard + HRR rerank → trust weighting → optional temporal decay
        0.5^(age_days / half_life). Returns fact dicts with 'score', sorted desc."""
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)
        query_tokens = self._tokenize(query)
        # Query vector is loop-invariant; encode lazily on the first candidate that carries an HRR vector
        # so stores whose hrr_vector was never backfilled don't pay for it.
        query_vec = None
        for fact in candidates:
            jaccard = self._jaccard_similarity(query_tokens, self._tokenize(fact["content"]) | self._tokenize(fact.get("tags", "")))
            hrr_sim = 0.5  # neutral
            if self.hrr_weight > 0 and fact.get("hrr_vector"):
                fact_vec = self._phases(fact["hrr_vector"])
                if query_vec is None:
                    query_vec = hrr.encode_text(query, self.hrr_dim)
                hrr_sim = _shift(hrr.similarity(query_vec, fact_vec))
            relevance = self.fts_weight * fact.get("fts_rank", 0.0) + self.jaccard_weight * jaccard + self.hrr_weight * hrr_sim
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
            fact["score"] = relevance * fact["trust_score"]
            if self.half_life > 0:
                fact["score"] *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))
        results = sorted(candidates, key=lambda x: x["score"], reverse=True)[:limit]
        for fact in results:
            fact.pop("hrr_vector", None)  # callers expect JSON-serializable dicts
        return results

    def _vector_query(self, fallback: str, category: str | None, limit: int, sim_fn: Callable) -> list[dict]:
        """Rank every fact vector (optionally per category) by sim_fn; FTS5 fallback when no vectors exist."""
        rows = self._vector_rows(category)
        return self._rank_by_vector(rows, sim_fn, limit) if rows else self.search(fallback, category=category, limit=limit)

    def probe(self, entity: str, category: str | None = None, limit: int = 10) -> list[dict]:
        """Compositional entity query: unbind bind(entity, ROLE_ENTITY) from the category bank (or each fact vector)
        to find facts where the entity plays a structural role. Not keyword search. Falls back to FTS5 without numpy."""
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, limit=limit)
        probe_key = hrr.bind(self._atom(entity.lower()), self._atom(_ROLE_ENTITY))
        if category:  # category bank first, then individual fact vectors
            bank_row = self.store._conn.execute("SELECT vector FROM memory_banks WHERE bank_name = ?", (f"cat:{category}",)).fetchone()
            if bank_row:
                extracted = hrr.unbind(self._phases(bank_row["vector"]), probe_key)
                return self._rank_by_vector(self._vector_rows(category), lambda _f, fact_vec: hrr.similarity(extracted, fact_vec), limit)
        role_content = self._atom(_ROLE_CONTENT)  # loop-invariant: encode once, not per row
        # Does unbinding the probe key leave the fact's content signal?
        return self._vector_query(entity, category, limit, lambda fact, fact_vec: hrr.similarity(
            hrr.unbind(fact_vec, probe_key), hrr.bind(hrr.encode_text(fact["content"], self.hrr_dim), role_content)))

    def related(self, entity: str, category: str | None = None, limit: int = 10) -> list[dict]:
        """Facts structurally connected to an entity (shared context), not just facts *about* it as in probe.
        Falls back to FTS5 without numpy."""
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, limit=limit)
        entity_vec = self._atom(entity.lower())  # bare atom, not role-bound: ANY structural match
        roles = (self._atom(_ROLE_ENTITY), self._atom(_ROLE_CONTENT))  # loop-invariant: encode once
        # A residual similar to ANY role vector means the entity plays a structural role in the fact.
        return self._vector_query(entity, category, limit, lambda _f, fact_vec: max(
            hrr.similarity(hrr.unbind(fact_vec, entity_vec), role) for role in roles))

    def reason(self, entities: list[str], category: str | None = None, limit: int = 10) -> list[dict]:
        """Multi-entity compositional query (vector-space JOIN): facts where ALL entities play structural roles.
        Falls back to FTS5 without numpy."""
        if not hrr._HAS_NUMPY or not entities:
            return self.search(" ".join(entities), category=category, limit=limit)
        role_entity, role_content = self._atom(_ROLE_ENTITY), self._atom(_ROLE_CONTENT)
        probe_keys = [hrr.bind(self._atom(entity.lower()), role_entity) for entity in entities]
        # AND semantics via min: high only if EVERY entity is structurally present.
        return self._vector_query(" ".join(entities), category, limit, lambda _f, fact_vec: min(
            hrr.similarity(hrr.unbind(fact_vec, key), role_content) for key in probe_keys))

    def contradict(self, category: str | None = None, threshold: float = 0.3, limit: int = 10) -> list[dict]:
        """Pairs of facts sharing entities (same subject) with low content-vector similarity (different claims). Empty without numpy."""
        if not hrr._HAS_NUMPY:
            return []
        rows = self._vector_rows(category, columns="fact_id, content, category, tags, trust_score, created_at, updated_at, hrr_vector")
        if len(rows) < 2:
            return []
        if len(rows) > 500:  # O(n²) guard: only compare the most recently updated facts
            rows = sorted(rows, key=lambda r: r["updated_at"] or r["created_at"], reverse=True)[:500]
        facts = []  # (public dict, lower-cased entity names, phase vector)
        for row in rows:
            fact = dict(row)
            entity_rows = self.store._conn.execute(
                "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
                (fact["fact_id"],),
            ).fetchall()
            facts.append((fact, {r["name"].lower() for r in entity_rows}, self._phases(fact.pop("hrr_vector"))))
        contradictions = []
        for i, (f1, ents1, vec1) in enumerate(facts):
            for f2, ents2, vec2 in facts[i + 1:]:
                if not ents1 or not ents2:
                    continue
                entity_overlap = len(ents1 & ents2) / len(ents1 | ents2)
                if entity_overlap < 0.3:
                    continue  # not enough shared subject to be contradictory
                content_sim = hrr.similarity(vec1, vec2)
                contradiction_score = entity_overlap * (1.0 - _shift(content_sim))  # high overlap + low similarity
                if contradiction_score >= threshold:
                    contradictions.append({
                        "fact_a": f1, "fact_b": f2,
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_sim, 3),
                        "contradiction_score": round(contradiction_score, 3),
                        "shared_entities": sorted(ents1 & ents2),
                    })
        return sorted(contradictions, key=lambda x: x["contradiction_score"], reverse=True)[:limit]

    def _vector_rows(self, category: str | None, columns: str = _FACT_COLUMNS + ", hrr_vector") -> list:
        """All facts that carry an HRR vector, optionally filtered by category."""
        where = "WHERE hrr_vector IS NOT NULL" + (" AND category = ?" if category else "")
        return self.store._conn.execute(f"SELECT {columns} FROM facts {where}", [category] if category else []).fetchall()

    def _rank_by_vector(self, rows: list, sim_fn: Callable[[dict, object], float], limit: int) -> list[dict]:
        """Score each row as (sim + 1) / 2 * trust_score (sim shifted to [0, 1]), sorted desc."""
        scored = [dict(row) for row in rows]
        for fact in scored:
            fact["score"] = _shift(sim_fn(fact, self._phases(fact.pop("hrr_vector")))) * fact["trust_score"]
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]

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

    def _fts_candidates(self, query: str, category: str | None, min_trust: float, limit: int) -> list[dict]:
        """Raw FTS5 MATCH candidates with rank normalized to [0, 1] as 'fts_rank'."""
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
            # natural-language queries. Sanitize: drop stopwords, OR-join
            # content tokens, so any significant term can match.
            match_arg = self._sanitize_fts_query(query)
        category_clause = "AND f.category = ? " if category else ""
        params = [match_arg] + ([category] if category else []) + [min_trust, limit]
        sql = (f"SELECT f.*, {table}.rank as fts_rank_raw FROM {table} JOIN facts f ON f.fact_id = {table}.rowid "
               f"WHERE {table} MATCH ? {category_clause}AND f.trust_score >= ? ORDER BY {table}.rank LIMIT ?")
        try:
            rows = [dict(row) for row in self.store._conn.execute(sql, params).fetchall()]
        except Exception:
            return []  # FTS5 MATCH can fail on malformed queries

        # For the trigram path, augment ranking with per-fact window-token
        # hit count: count how many of the query's 3-char windows appear in
        # this fact's content.  More hits = more relevant (natural-language
        # queries rarely match verbatim, so this beats raw FTS rank).
        if cjk and self._trigram_available():
            windows = self._window_tokens(query)
            out = []
            raw_ranks = [abs(r["fts_rank_raw"]) for r in rows]
            max_rank = max(raw_ranks + [1e-6])
            for row in rows:
                fact = dict(row)
                fact.pop("fts_rank_raw", None)
                content = fact.get("content", "")
                if windows:
                    hits = sum(1 for w in windows if w in content)
                    fact["window_hits"] = hits
                    # fts_rank keeps its normalized position as a tiebreak
                    fact["fts_rank"] = abs(row["fts_rank_raw"]) / max_rank
                else:
                    fact["window_hits"] = 0
                    fact["fts_rank"] = 0.5
                out.append(fact)
            return out

        # FTS5 rank is negative (lower = better); normalize |rank| / max to [0, 1] (1e-6 floor avoids div by zero)
        max_rank = max([abs(f["fts_rank_raw"]) for f in rows] + [1e-6])
        for fact in rows:
            fact["fts_rank"] = abs(fact.pop("fts_rank_raw")) / max_rank
        return rows

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
        """Lowercase whitespace tokens with surrounding punctuation stripped (no stemming)."""
        return {c for c in (w.strip(_PUNCT) for w in text.lower().split()) if c} if text else set()

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Natural-language query -> FTS5-safe OR expression of quoted tokens. FTS5 AND-joins a multi-word
        MATCH by default, which tanks recall on prose: drop stopwords and <2-char tokens, strip FTS5 operator
        chars, phrase-quote each survivor. If nothing survives, return the raw query (zero results, not a SQL error)."""
        if not query:
            return ""
        tokens = [f'"{c}"' for c in (raw.strip(_PUNCT).translate(_FTS_OPERATORS) for raw in query.lower().split())
                  if len(c) >= 2 and c not in _FTS_STOPWORDS]
        return " OR ".join(tokens) if tokens else query

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        return len(set_a & set_b) / len(set_a | set_b) if set_a and set_b else 0.0

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """0.5^(age_days / half_life); 1.0 if disabled, missing, unparseable, or in the future."""
        if not self.half_life or not timestamp_str:
            return 1.0
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00")) if isinstance(timestamp_str, str) else timestamp_str
            age_days = (datetime.now(timezone.utc) - (ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc))).total_seconds() / 86400
            return 1.0 if age_days < 0 else math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0

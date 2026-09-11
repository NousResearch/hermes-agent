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

    def _fts_candidates(self, query: str, category: str | None, min_trust: float, limit: int) -> list[dict]:
        """Raw FTS5 MATCH candidates with rank normalized to [0, 1] as 'fts_rank'."""
        category_clause = "AND f.category = ? " if category else ""
        params = [self._sanitize_fts_query(query)] + ([category] if category else []) + [min_trust, limit]
        sql = ("SELECT f.*, facts_fts.rank as fts_rank_raw FROM facts_fts JOIN facts f ON f.fact_id = facts_fts.rowid "
               f"WHERE facts_fts MATCH ? {category_clause}AND f.trust_score >= ? ORDER BY facts_fts.rank LIMIT ?")
        try:
            results = [dict(row) for row in self.store._conn.execute(sql, params).fetchall()]
        except Exception:
            return []  # FTS5 MATCH can fail on malformed queries
        # FTS5 rank is negative (lower = better); normalize |rank| / max to [0, 1] (1e-6 floor avoids div by zero)
        max_rank = max([abs(f["fts_rank_raw"]) for f in results] + [1e-6])
        for fact in results:
            fact["fts_rank"] = abs(fact.pop("fts_rank_raw")) / max_rank
        return results

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

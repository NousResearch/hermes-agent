"""Hybrid keyword/BM25 retrieval for the memory store: FTS5 candidates reranked with
Jaccard similarity and HRR vector similarity, trust-weighted (ported from KIK memory_agent.py)."""

from __future__ import annotations

import logging
import math
import unicodedata
from collections.abc import Callable
from datetime import datetime, timezone
from functools import wraps
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .store import MemoryStore

from . import holographic as hrr

_FACT_COLUMNS = "fact_id, content, category, tags, trust_score, retrieval_count, helpful_count, created_at, updated_at, lifecycle, verified_at"
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


def _locked(fn: Callable) -> Callable:
    """Run a retriever method under the store's shared lock (single SQLite
    connection across threads — unlocked concurrent reads race writers)."""
    @wraps(fn)
    def _wrapper(self, *args, **kwargs):
        with self.store._lock:
            return fn(self, *args, **kwargs)
    return _wrapper


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
        0.5^(age_days / half_life). Returns fact dicts with 'score', sorted desc.

        Hardening: entity-alias expansion widens lexical recall; when FTS5
        yields zero candidates, a bounded HRR vector scan backs it up instead
        of returning nothing (tagged ``_fallback_hrr``)."""
        with self.store._lock:
            return self._search_locked(query, category, min_trust, limit)

    def _search_locked(self, query: str, category: str | None, min_trust: float, limit: int) -> list[dict]:
        """search() body; runs under the store's shared lock (single SQLite
        connection across threads — unlocked concurrent reads race writers).

        Candidate fetch never raises: a degraded store degrades to no
        results (logged) instead of crashing retrieval."""
        try:
            alias_extra = self._alias_tokens(query)  # single entities scan per search
            expanded_query = query + (" " + " ".join(sorted(alias_extra)) if alias_extra else "")
            candidates = self._fts_candidates(expanded_query, category, min_trust, limit * 3)
            if not candidates:
                candidates = self._hrr_fallback(query, category, min_trust, limit * 3)
            if len(candidates) < limit:  # Thai bigram supplement for unsegmented queries
                seen_ids = {f["fact_id"] for f in candidates}
                for row in self._thai_bigram_candidates(query, category, min_trust, limit):
                    if row["fact_id"] not in seen_ids:
                        seen_ids.add(row["fact_id"])
                        candidates.append(row)
        except Exception as e:
            logger.debug("Holographic candidate fetch failed: %s", e)
            return []
        query_tokens = self._tokenize(query) | alias_extra | self._thai_bigrams(query)
        # Query vector is loop-invariant; encode lazily on the first candidate that carries an HRR vector
        # so stores whose hrr_vector was never backfilled don't pay for it.
        query_vec = None
        for fact in candidates:
            jaccard = self._jaccard_similarity(query_tokens, self._tokenize(fact["content"]) | self._tokenize(fact.get("tags", "")) | self._thai_bigrams(fact["content"]) | self._thai_bigrams(fact.get("tags", "")))
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
        results = sorted(candidates, key=self._recency_key, reverse=True)
        results = self._apply_lifecycle_partition(results)
        results = results[:limit]
        for fact in results:
            fact.pop("hrr_vector", None)  # callers expect JSON-serializable dicts
        return results

    @staticmethod
    def _thai_bigrams(text: str) -> set[str]:
        """Overlapping 2-char grams of Thai-script runs (U+0E00-U+0E7F).

        Deterministic stdlib fallback for unsegmented Thai (whitespace
        tokenization yields one giant token). No external dependency."""
        import re as _re
        norm = unicodedata.normalize("NFKC", text or "")
        grams: set[str] = set()
        for run in _re.findall(r"[\u0e00-\u0e7f]{2,}", norm):
            for i in range(len(run) - 1):
                grams.add(run[i:i + 2])
        return grams

    def _thai_bigram_candidates(self, query: str, category: str | None,
                                min_trust: float, limit: int) -> list[dict]:
        """LIKE scan over query Thai bigrams (bounded: first 8 grams). Only
        used when lexical candidates run short. Tagged ``_thai_bigram``."""
        grams = sorted(self._thai_bigrams(query))[:8]
        if not grams:
            return []
        clauses, params = [], []
        for gram in grams:
            esc = gram.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("content LIKE ? ESCAPE '\\'")
            params.append(f"%{esc}%")
        where = "(" + " OR ".join(clauses) + ")"
        if category:
            where += " AND category = ?"
            params.append(category)
        params.extend([min_trust, limit])
        sql = ("SELECT fact_id, content, category, tags, trust_score, retrieval_count,"
               " helpful_count, created_at, updated_at, lifecycle, verified_at FROM facts "
               f"WHERE {where} AND trust_score >= ? ORDER BY trust_score DESC LIMIT ?")
        try:
            rows = [dict(r) for r in self.store._conn.execute(sql, params).fetchall()]
        except Exception:
            return []
        for row in rows:
            row["fts_rank"] = 0.3
            row["_thai_bigram"] = True
        return rows

    # R4 demotion weights for non-search vector paths (probe/related/reason).
    # Search uses the hard partition instead; these multipliers keep history
    # demoted wherever scores are compared.
    _LIFECYCLE_DEMOTE = {"active": 1.0, "conflict": 0.4, "stale": 0.5,
                         "superseded": 0.25, "revoked": 0.1}

    @staticmethod
    def _lifecycle_of(fact: dict) -> str:
        try:
            return (fact.get("lifecycle") or "active").lower()
        except Exception:
            return "active"

    def _apply_lifecycle_partition(self, results: list[dict]) -> list[dict]:
        """Flag non-active rows and stable-partition verified/current above
        history. Shared by search and vector-query paths. The computed
        ``stale`` key exists ONLY on non-active rows (active rows keep the
        exact pre-R4 dict shape required by the parity contract) — callers
        must use ``fact.get("stale", False)``."""
        try:
            from .store import LIFECYCLE_RANK as _RANK, _VERIFIED_RANK as _VRANK
        except Exception:
            return results

        def _class(fact: dict) -> int:
            lc = self._lifecycle_of(fact)
            if lc == "active" and (fact.get("verified_at") or ""):
                return _VRANK
            return _RANK.get(lc, 1)

        for fact in results:
            if self._lifecycle_of(fact) != "active":
                fact["lifecycle"] = self._lifecycle_of(fact)
                fact["stale"] = True
        results.sort(key=_class)
        return results

    @staticmethod
    def _recency_key(fact: dict) -> tuple:
        """Explicit tie-break (trust already dominates via score): newer
        updated_at first, then higher fact_id (insertion order). Timestamps
        have second resolution, so fact_id is the deterministic decider."""
        return (fact.get("score", 0.0), fact.get("updated_at") or "", fact.get("fact_id") or 0)

    def _vector_query(self, fallback: str, category: str | None, limit: int, sim_fn: Callable) -> list[dict]:
        """Rank every fact vector (optionally per category) by sim_fn; FTS5 fallback when no vectors exist."""
        rows = self._vector_rows(category)
        return self._rank_by_vector(rows, sim_fn, limit) if rows else self.search(fallback, category=category, limit=limit)

    @_locked
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

    @_locked
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

    @_locked
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

    @_locked
    def contradict(self, category: str | None = None, threshold: float = 0.3, limit: int = 10) -> list[dict]:
        """Pairs of facts sharing entities (same subject) with low content-vector similarity (different claims). Empty without numpy.

        Hardened: a deterministic slot pass (same subject+predicate, different
        value) covers entity-less key/value facts the HRR pass skips. Slot
        hits carry ``method: 'slot'``; HRR hits carry no method key."""
        slot_hits = self._slot_contradictions(category, limit)
        if not hrr._HAS_NUMPY:
            return slot_hits
        rows = self._vector_rows(category, columns="fact_id, content, category, tags, trust_score, created_at, updated_at, lifecycle, hrr_vector")
        try:
            rows = [r for r in rows if (r["lifecycle"] or "active").lower() != "revoked"]
        except Exception:
            pass
        if len(rows) < 2:
            return slot_hits[:limit]
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
        merged = list(slot_hits) + sorted(
            contradictions, key=lambda x: x["contradiction_score"], reverse=True)
        return merged[:limit]

    _SLOT_SPLIT = None  # compiled lazily (module import must stay cheap)

    @staticmethod
    def _split_slot(content: str) -> tuple[str, str, str]:
        """Split 'subject predicate value' on = -> → คือ (first occurrence).

        Bare ':' is deliberately excluded: it is the prose-heavy separator
        ('notes: buy milk' vs 'notes: call bob' are not contradictions).
        Returns (subject, predicate, value) normalized; empty subject means
        'no parseable slot' (never a conflict signal by itself)."""
        import re as _re
        import unicodedata as _ud
        norm = _ud.normalize("NFKC", (content or "").strip().lower())
        parts = _re.split(r"\s*(=|->|→|คือ)\s*", norm, maxsplit=1)
        if len(parts) < 3 or not parts[0].strip() or not parts[2].strip():
            return "", "", ""
        return parts[0].strip(), parts[1].strip(), parts[2].strip()

    def _slot_contradictions(self, category: str | None, limit: int) -> list[dict]:
        """Entity-less contradiction: same (subject, predicate) with different
        normalized values. Same value is NEVER a conflict (timeout=30 twice).
        Pure lexical pass — complements the HRR entity-overlap pass. Capped
        at 500 rows like the HRR pass (O(pairs) grouping guard)."""
        try:
            rows = self.store._conn.execute(
                "SELECT fact_id, content, category, tags, trust_score, created_at, updated_at "
                "FROM facts WHERE lifecycle != 'revoked'" + (" AND category = ?" if category else "") +
                " ORDER BY updated_at DESC LIMIT 500",
                [category] if category else [],
            ).fetchall()
        except Exception:
            return []
        groups: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            try:
                fact = dict(row)
                subj, pred, val = self._split_slot(fact.get("content", ""))
                if not subj:
                    continue
                fact["_slot_value"] = val
                groups.setdefault((subj, pred), []).append(fact)
            except Exception:
                continue
        out = []
        for _slot, members in groups.items():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if members[i]["_slot_value"] == members[j]["_slot_value"]:
                        continue  # same value: agreement, not conflict
                    a = {k: v for k, v in members[i].items() if not k.startswith("_")}
                    b = {k: v for k, v in members[j].items() if not k.startswith("_")}
                    out.append({"fact_a": a, "fact_b": b, "entity_overlap": 1.0,
                                "content_similarity": 0.0, "contradiction_score": 1.0,
                                "shared_entities": [], "method": "slot",
                                "slot": list(_slot)})
                    if len(out) >= limit:
                        return out
        return out

    def _vector_rows(self, category: str | None, columns: str = _FACT_COLUMNS + ", hrr_vector", limit: int = 2000) -> list:
        """All facts that carry an HRR vector, optionally filtered by category.

        Capped (default 2000, trust-ordered) so vector scans stay bounded on
        large DBs; callers needing exhaustive comparison pass limit=None."""
        where = "WHERE hrr_vector IS NOT NULL" + (" AND category = ?" if category else "")
        sql = f"SELECT {columns} FROM facts {where} ORDER BY trust_score DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params = ([category] if category else []) + [limit]
        else:
            params = [category] if category else []
        return self.store._conn.execute(sql, params).fetchall()

    def _rank_by_vector(self, rows: list, sim_fn: Callable[[dict, object], float], limit: int) -> list[dict]:
        """Score each row as (sim + 1) / 2 * trust_score (sim shifted to [0, 1]),
        demoted by lifecycle, verified/current partitioned above history."""
        scored = [dict(row) for row in rows]
        for fact in scored:
            demote = self._LIFECYCLE_DEMOTE.get(self._lifecycle_of(fact), 1.0)
            fact["score"] = _shift(sim_fn(fact, self._phases(fact.pop("hrr_vector")))) * fact["trust_score"] * demote
        scored = sorted(scored, key=lambda x: x["score"], reverse=True)
        return self._apply_lifecycle_partition(scored)[:limit]

    def _with_aliases(self, query: str) -> str:
        """Append entity-alias tokens to the query (cheap lexical recall widening)."""
        extra = self._alias_tokens(query)
        return query + (" " + " ".join(sorted(extra)) if extra else "")

    def _alias_tokens(self, query: str) -> set[str]:
        """Tokens that are known aliases/names of stored entities (and v.v.).

        Expansion members are stopword-filtered (a multi-word alias like
        "New York" must not inject high-frequency tokens into MATCH)."""
        try:
            rows = self.store._conn.execute("SELECT name, aliases FROM entities").fetchall()
        except Exception:
            return set()
        if not rows:
            return set()
        qtokens = {t.lower() for t in self._tokenize(query)}
        extra: set[str] = set()
        for row in rows:
            try:
                name = (row["name"] or "").strip()
                aliases = [a.strip() for a in (row["aliases"] or "").split(",") if a.strip()]
            except Exception:
                continue
            group = {name.lower()} | {a.lower() for a in aliases}
            hit = qtokens & group
            if hit:
                for token in group - hit:
                    for sub in self._tokenize(token):
                        if len(sub) >= 2 and sub not in _FTS_STOPWORDS:
                            extra.add(sub)
        return extra

    def _hrr_fallback(self, query: str, category: str | None, min_trust: float, limit: int) -> list[dict]:
        """Bounded HRR vector scan when FTS5 yields nothing. Returns [] without
        numpy/vectors; rows are tagged ``_fallback_hrr`` and carry fts_rank 0."""
        if not hrr._HAS_NUMPY or not (query or "").strip():
            return []
        try:
            rows = self._vector_rows(category)
        except Exception:
            return []
        if not rows:
            return []
        try:
            query_vec = hrr.encode_text(query, self.hrr_dim)
        except Exception:
            return []
        scored = []
        for row in rows:
            try:
                fact = dict(row)
                blob = fact.pop("hrr_vector", None)
                if blob is None:
                    continue
                if float(fact.get("trust_score", 0.0) or 0.0) < min_trust:
                    continue
                fact["score"] = _shift(hrr.similarity(query_vec, self._phases(blob)))
                fact["fts_rank"] = 0.0
                fact["_fallback_hrr"] = True
                scored.append(fact)
            except Exception:
                continue
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
        # FTS5 rank is negative (lower = better); normalize |rank| / max to [0, 1] (1e-6 floor avoids div by zero).
        # The ``or 0.0`` guard is belt-and-braces: rank must never crash ranking.
        max_rank = max([abs(f["fts_rank_raw"] or 0.0) for f in results] + [1e-6])
        for fact in results:
            fact["fts_rank"] = abs(fact.pop("fts_rank_raw") or 0.0) / max_rank
        return results

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Lowercase whitespace tokens with surrounding punctuation stripped (no stemming).

        Hardened: NFKC normalization first (fullwidth/compat forms), hyphens
        and underscores treated as separators so ``blue-green`` matches
        ``blue green`` lexically."""
        if not text:
            return set()
        norm = unicodedata.normalize("NFKC", text.lower()).replace("-", " ").replace("_", " ")
        return {c for c in (w.strip(_PUNCT) for w in norm.split()) if c}

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

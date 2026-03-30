#!/usr/bin/env python3
"""
Integrated Retriever — Two-phase search with zero-token fast path.

Phase 1: Keyword bitmap lookup (O(1), 0 tokens, ~50μs)
Phase 2: BM25 fallback (0 tokens, ~5ms) — only if Phase 1 < 3 results
Phase 3: Semantic (placeholder, future embeddings)

Scoring merge:
  keyword_match: 0.5 weight
  bm25_score:    0.3 weight
  temporal:      0.2 weight
"""

import json
import math
import sqlite3
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))
from quantum_index import QuantumIndex, ProceduralExtractor
from semantic_index import SemanticIndex

DB_PATH = Path.home() / ".hermes/memory-engine/db/memory.db"


@dataclass
class SearchResult:
    fact_id: str
    summary: str
    score: float
    keywords: Dict
    status: str
    tier: str
    source_phase: int  # 1=keyword, 2=bm25, 3=semantic
    matched_dimensions: List[str] = field(default_factory=list)
    raw_content: str = ""


class IntegratedRetriever:
    """Two-phase retriever: keyword fast-path → BM25 fallback."""

    KEYWORD_WEIGHT = 0.5
    BM25_WEIGHT = 0.3
    TEMPORAL_WEIGHT = 0.2
    DECAY_LAMBDA = 0.05

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self.extractor = ProceduralExtractor()
        self.idx = None
        self.sem = None
        self.conn = None

    def connect(self):
        self.idx = QuantumIndex(self.db_path)
        self.idx.connect()
        self.conn = self.idx.conn
        self.sem = SemanticIndex(self.db_path)
        self.sem.connect()

    def close(self):
        if self.idx:
            self.idx.close()
        if self.sem:
            self.sem.close()

    # ── Phase 1: Keyword Fast Path ──────────────────────────────

    def _phase1_keyword(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        O(1) keyword lookup. Zero tokens.
        Decomposes query into dimensions, looks up bitmap index.
        """
        # Extract keywords from query
        keywords = self.extractor.extract(query)

        # Build lookup filters from strongest dimensions
        filters = {}
        query_dimensions = []

        for dim in ("domain", "action", "status", "entities"):
            values = keywords.get(dim, [])
            if values and dim != "emotion":  # skip emotion for search
                # Try each value independently and merge results
                for val in values[:2]:  # max 2 per dimension
                    filters[dim] = val
                    query_dimensions.append(dim)

        if not filters:
            return []

        # Try progressively relaxed lookups
        results = []
        seen_ids = set()

        # First: try all filters combined (most specific)
        if len(filters) > 1:
            matches = self.idx.lookup(top_k=top_k, **filters)
            for m in matches:
                if m.fact_id not in seen_ids:
                    seen_ids.add(m.fact_id)
                    # Score based on dimensions matched
                    dim_score = len(filters) / max(len(query_dimensions), 1)
                    results.append(SearchResult(
                        fact_id=m.fact_id,
                        summary=m.summary,
                        score=dim_score * self.KEYWORD_WEIGHT,
                        keywords=m.keywords,
                        status=m.status,
                        tier=m.storage_tier,
                        source_phase=1,
                        matched_dimensions=list(filters.keys()),
                    ))

        # Then: try individual dimensions (broader)
        for dim, val in filters.items():
            matches = self.idx.lookup(top_k=top_k, **{dim: val})
            for m in matches:
                if m.fact_id not in seen_ids:
                    seen_ids.add(m.fact_id)
                    dim_score = 1.0 / max(len(query_dimensions), 1)
                    results.append(SearchResult(
                        fact_id=m.fact_id,
                        summary=m.summary,
                        score=dim_score * self.KEYWORD_WEIGHT * 0.7,  # slightly lower for partial match
                        keywords=m.keywords,
                        status=m.status,
                        tier=m.storage_tier,
                        source_phase=1,
                        matched_dimensions=[dim],
                    ))

        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

    # ── Phase 2: BM25 Fallback ──────────────────────────────────

    def _phase2_bm25(self, query: str, top_k: int = 10, exclude_ids: Set = None) -> List[SearchResult]:
        """
        BM25 scoring against memory_facts + quantum_facts. Zero tokens.
        Only runs if Phase 1 returned < 3 results.
        """
        exclude_ids = exclude_ids or set()
        query_terms = self._tokenize(query)

        if not query_terms:
            return []

        # Get total document count
        total_docs = self.conn.execute("SELECT COUNT(*) FROM quantum_facts").fetchone()[0]
        if total_docs == 0:
            return []

        # BM25 parameters
        k1 = 1.5
        b = 0.75

        # Get average document length
        avg_row = self.conn.execute(
            "SELECT AVG(LENGTH(summary)) FROM quantum_facts"
        ).fetchone()
        avg_dl = avg_row[0] or 50.0

        # Score each document
        rows = self.conn.execute("""
            SELECT id, summary, keywords_readable, status, priority_weight,
                   storage_tier, created_at, LENGTH(summary) as doc_len
            FROM quantum_facts
        """).fetchall()

        results = []
        for row in rows:
            fid, summary, kw_json, status, pw, tier, created, doc_len = row
            if fid in exclude_ids:
                continue

            # BM25 scoring
            score = 0.0
            summary_lower = summary.lower()
            kw_text = (kw_json or "").lower()
            full_text = summary_lower + " " + kw_text

            for term in query_terms:
                # Term frequency in document
                tf = full_text.count(term)
                if tf == 0:
                    continue

                # Document frequency (how many docs contain this term)
                df = sum(1 for r in rows if term in (r[1] or "").lower() + " " + (r[2] or "").lower())

                # IDF
                idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)

                # BM25 term score
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_dl)))
                score += idf * tf_norm

            if score > 0:
                # Normalize to 0-1 range
                normalized_score = min(1.0, score / (len(query_terms) * 3))

                # Temporal decay
                temporal_score = self._temporal_score(created)

                # Combined score
                final_score = (
                    normalized_score * self.BM25_WEIGHT +
                    temporal_score * self.TEMPORAL_WEIGHT
                )

                keywords = json.loads(kw_json) if kw_json else {}

                results.append(SearchResult(
                    fact_id=fid,
                    summary=summary,
                    score=final_score,
                    keywords=keywords,
                    status=status or "unknown",
                    tier=tier or "warm",
                    source_phase=2,
                    matched_dimensions=["bm25"],
                ))

        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

    # ── Phase 3: Semantic Search ───────────────────────────────

    def _phase3_semantic(self, query: str, top_k: int = 10, exclude_ids: set = None) -> List[SearchResult]:
        """
        Semantic similarity search via local embeddings.
        0 tokens, ~10ms. Only runs if Phase 1+2 < 2 results.
        """
        exclude_ids = exclude_ids or set()

        try:
            sem_results = self.sem.search(query, top_k=top_k, min_similarity=0.3)
        except Exception:
            return []

        results = []
        for r in sem_results:
            if r.fact_id in exclude_ids:
                continue

            # Temporal decay
            temporal = 0.5  # default

            # Weighted score: semantic gets SEMANTIC_WEIGHT (0.2 base, but here it's the primary signal)
            score = r.similarity * 0.4 + temporal * self.TEMPORAL_WEIGHT

            results.append(SearchResult(
                fact_id=r.fact_id,
                summary=r.summary,
                score=score,
                keywords=r.keywords,
                status=r.status,
                tier=r.tier,
                source_phase=3,
                matched_dimensions=["semantic"],
            ))

        return results

    # ── Unified Search ──────────────────────────────────────────

    def search(self, query: str, top_k: int = 5, status_filter: str = None) -> List[SearchResult]:
        """
        Three-phase search:
        1. Keyword fast-path (0 tokens, ~50μs)
        2. BM25 fallback if needed (0 tokens, ~5ms)
        3. Semantic if still insufficient (0 tokens, ~10ms)
        """
        # Phase 1: Keyword lookup
        results = self._phase1_keyword(query, top_k=top_k * 2)

        # Phase 2: BM25 if Phase 1 insufficient
        if len(results) < 3:
            seen_ids = {r.fact_id for r in results}
            bm25_results = self._phase2_bm25(query, top_k=top_k, exclude_ids=seen_ids)
            results.extend(bm25_results)

        # Phase 3: Semantic if still insufficient
        if len(results) < 2:
            seen_ids = {r.fact_id for r in results}
            sem_results = self._phase3_semantic(query, top_k=top_k, exclude_ids=seen_ids)
            results.extend(sem_results)

        # Apply status filter
        if status_filter:
            results = [r for r in results if r.status == status_filter]

        # Sort by score and return top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def search_pending(self, top_k: int = 10) -> List[SearchResult]:
        """Shortcut: all pending/in_progress/blocked work."""
        pending = self.idx.lookup_pending(top_k=top_k)
        return [
            SearchResult(
                fact_id=p.fact_id,
                summary=p.summary,
                score=p.priority_weight,
                keywords=p.keywords,
                status=p.status,
                tier=p.storage_tier,
                source_phase=1,
                matched_dimensions=["status"],
            )
            for p in pending
        ]

    def get_context_window(self, fact_id: str, window: int = 2) -> List[SearchResult]:
        """Get temporal neighbors of a fact."""
        row = self.conn.execute(
            "SELECT created_at FROM quantum_facts WHERE id=?", (fact_id,)
        ).fetchone()

        if not row or not row[0]:
            return []

        # Get facts within window of time
        neighbors = self.conn.execute("""
            SELECT id, summary, keywords_readable, status, priority_weight,
                   storage_tier, created_at
            FROM quantum_facts
            WHERE id != ?
            ORDER BY ABS(JULIANDAY(created_at) - JULIANDAY(?))
            LIMIT ?
        """, (fact_id, row[0], window * 2)).fetchall()

        return [
            SearchResult(
                fact_id=n[0], summary=n[1],
                score=0.5,
                keywords=json.loads(n[2]) if n[2] else {},
                status=n[3] or "unknown",
                tier=n[5] or "warm",
                source_phase=1,
                matched_dimensions=["temporal_window"],
            )
            for n in neighbors
        ]

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenization for BM25."""
        text = text.lower()
        # Remove punctuation, split
        tokens = re.findall(r'[a-záéíóúñü0-9]+', text)
        # Remove stopwords
        stops = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                 "to", "for", "of", "and", "or", "but", "not", "with", "from",
                 "el", "la", "los", "las", "de", "en", "un", "una", "que", "y",
                 "por", "con", "es", "del", "al", "se", "no", "lo"}
        return [t for t in tokens if t not in stops and len(t) > 1]

    @staticmethod
    def _temporal_score(created_at: str) -> float:
        """Exponential temporal decay."""
        if not created_at:
            return 0.5

        try:
            created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            days = (datetime.now() - created.replace(tzinfo=None)).days
            return math.exp(-0.05 * max(days, 0))
        except (ValueError, TypeError):
            return 0.5


# ── CLI ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Integrated Two-Phase Retriever")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--pending", action="store_true", help="Show pending work")
    parser.add_argument("--context", type=str, help="Get context window for fact ID")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--status", type=str, help="Filter by status")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    ret = IntegratedRetriever()
    ret.connect()

    try:
        if args.pending:
            results = ret.search_pending(top_k=args.top_k)
            title = "PENDING WORK"
        elif args.context:
            results = ret.get_context_window(args.context)
            title = f"CONTEXT WINDOW: {args.context}"
        elif args.query:
            results = ret.search(args.query, top_k=args.top_k, status_filter=args.status)
            title = f"SEARCH: {args.query}"
        else:
            parser.print_help()
            return

        if args.json:
            print(json.dumps([{
                "id": r.fact_id, "summary": r.summary, "score": round(r.score, 4),
                "status": r.status, "tier": r.tier, "phase": r.source_phase,
                "matched": r.matched_dimensions, "keywords": r.keywords,
            } for r in results], indent=2))
        else:
            phase_label = {1: "KW", 2: "BM25", 3: "SEM"}
            tier_icon = {"hot": "●", "warm": "◐", "cold": "○"}

            print(f"\n  {title}")
            print(f"  {'─' * 55}\n")

            if not results:
                print("  No results found.\n")
                return

            for i, r in enumerate(results, 1):
                icon = tier_icon.get(r.tier, "?")
                phase = phase_label.get(r.source_phase, "?")
                print(f"  [{i}] {icon} {r.summary}")
                print(f"      Score: {r.score:.3f} | Phase: {phase} | Status: {r.status}")
                domains = r.keywords.get("domain", [])
                if domains:
                    print(f"      Domain: {', '.join(domains)}")
                if r.matched_dimensions:
                    print(f"      Matched: {', '.join(r.matched_dimensions)}")
                print()

    finally:
        ret.close()


if __name__ == "__main__":
    main()

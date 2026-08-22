#!/usr/bin/env python3
"""
Dynamic Context Selector — Phase 4
Smart token-budgeted context injection for LLM prompts.

Given a query and a token budget, selects the optimal subset of facts
to inject into context. Uses the three-phase retriever as the scoring
backbone, then applies selection strategies and formatting.

Design principles:
  - Token budget awareness (never exceed, prefer headroom)
  - Quality > quantity (3 ultra-relevant facts > 10 mediocre ones)
  - Status priority (pending/blocked facts surface first)
  - Source diversity (max N facts per domain)
  - Adaptive strategy (auto-detects query type)
  - Zero LLM tokens for selection (all local computation)

Strategies:
  greedy    → highest scored until budget full
  diverse   → source diversity cap + score
  windowed  → temporal neighbors for episodic recall
  pending   → prioritize unfinished work
  adaptive  → auto-select based on query signals

Usage:
    selector = ContextSelector(token_budget=2000)
    ctx = selector.select("Stripe payment error")
    print(ctx.formatted_context)  # inject this into prompt
    print(ctx.token_cost)         # tokens used
"""

import json
import math
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))
from quantum_index import QuantumIndex
from integrated_retriever import IntegratedRetriever, SearchResult
from semantic_index import SemanticIndex
from deep_layer import DeepLayer, ContextMonitor, ActivatedFact

DB_PATH = Path.home() / ".hermes/memory-engine/db/memory.db"


# ── Data Models ──────────────────────────────────────────────────

@dataclass
class ContextFact:
    """A fact selected for context injection."""
    fact_id: str
    summary: str
    content: str  # full text if available, else summary
    score: float
    token_cost: int
    status: str
    tier: str
    source_phase: int
    domains: List[str] = field(default_factory=list)
    matched_dimensions: List[str] = field(default_factory=list)


@dataclass
class ContextInjection:
    """The final context package ready for prompt injection."""
    facts: List[ContextFact]
    total_tokens: int
    budget: int
    budget_used_pct: float
    query: str
    strategy: str
    domain_diversity: Dict[str, int]
    formatted_context: str = ""
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "facts_selected": len(self.facts),
            "total_tokens": self.total_tokens,
            "budget": self.budget,
            "budget_used": f"{self.budget_used_pct:.0f}%",
            "domain_diversity": self.domain_diversity,
            "facts": [
                {
                    "id": f.fact_id,
                    "summary": f.summary[:80],
                    "score": round(f.score, 3),
                    "tokens": f.token_cost,
                    "status": f.status,
                    "phase": f.source_phase,
                    "domains": f.domains,
                }
                for f in self.facts
            ],
        }


# ── Token estimation ─────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Estimate token count. ~4 chars per token for English,
    ~3.5 for mixed English/Spanish.
    """
    return max(1, len(text) // 4)


FACT_OVERHEAD_TOKENS = 8  # [N] (domain | score%) \n


# ── Context Selector ─────────────────────────────────────────────

class ContextSelector:
    """
    Selects optimal facts for context injection within a token budget.
    Uses the three-phase retriever for scoring, then applies selection
    strategies and token-aware packing.
    """

    def __init__(
        self,
        token_budget: int = 2000,
        db_path: Path = None,
        max_per_domain: int = 3,
        min_score: float = 0.05,
        pending_boost: float = 1.5,
        surface_ratio: float = 0.6,    # Phase 8B: % of budget for surface facts
        auto_activate: bool = True,     # Phase 8B: auto-trigger Deep Layer
    ):
        self.token_budget = token_budget
        self.db_path = db_path or DB_PATH
        self.max_per_domain = max_per_domain
        self.min_score = min_score
        self.pending_boost = pending_boost
        self.surface_ratio = surface_ratio
        self.auto_activate = auto_activate
        self.retriever = None
        self.idx = None
        self.deep_layer = None

    def connect(self):
        self.retriever = IntegratedRetriever(self.db_path)
        self.retriever.connect()
        self.idx = QuantumIndex(self.db_path)
        self.idx.connect()
        # Phase 8B: connect Deep Layer
        self.deep_layer = DeepLayer(self.db_path)
        self.deep_layer.connect()

    def close(self):
        if self.retriever:
            self.retriever.close()
        if self.idx:
            self.idx.close()
        if self.deep_layer:
            self.deep_layer.close()

    # ── Scoring ──────────────────────────────────────────────────

    def _score_candidates(self, query: str, top_k: int = 30) -> List[ContextFact]:
        """
        Get scored candidates from the three-phase retriever.
        Enriches with full content and token costs.
        """
        results = self.retriever.search(query, top_k=top_k)

        # Also get pending work (always relevant)
        pending = self.retriever.search_pending(top_k=10)

        # Merge, deduplicate
        seen = set()
        all_results = []

        for r in results:
            if r.fact_id not in seen:
                seen.add(r.fact_id)
                all_results.append(r)

        for r in pending:
            if r.fact_id not in seen:
                seen.add(r.fact_id)
                # Boost pending score
                r.score *= self.pending_boost
                all_results.append(r)

        # Convert to ContextFact with full content
        candidates = []
        for r in all_results:
            # Try to get full content
            content = self.idx.reconstruct(r.fact_id) or r.summary

            # Apply status boost
            score = r.score
            if r.status in ("pending", "in_progress", "blocked"):
                score *= self.pending_boost

            token_cost = estimate_tokens(content) + FACT_OVERHEAD_TOKENS
            domains = r.keywords.get("domain", [])

            candidates.append(ContextFact(
                fact_id=r.fact_id,
                summary=r.summary,
                content=content,
                score=score,
                token_cost=token_cost,
                status=r.status,
                tier=r.tier,
                source_phase=r.source_phase,
                domains=domains,
                matched_dimensions=r.matched_dimensions,
            ))

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    # ── Selection Strategies ─────────────────────────────────────

    def _select_greedy(self, candidates: List[ContextFact]) -> List[ContextFact]:
        """Highest score until budget full."""
        selected = []
        tokens_used = 0

        for c in candidates:
            if c.score < self.min_score:
                continue
            if tokens_used + c.token_cost > self.token_budget:
                continue
            selected.append(c)
            tokens_used += c.token_cost

        return selected

    def _select_diverse(self, candidates: List[ContextFact]) -> List[ContextFact]:
        """Score + domain diversity cap."""
        selected = []
        tokens_used = 0
        domain_counts: Dict[str, int] = {}

        for c in candidates:
            if c.score < self.min_score:
                continue
            if tokens_used + c.token_cost > self.token_budget:
                continue

            # Check domain cap
            capped = False
            for d in c.domains:
                if domain_counts.get(d, 0) >= self.max_per_domain:
                    capped = True
                    break
            if capped:
                continue

            selected.append(c)
            tokens_used += c.token_cost
            for d in c.domains:
                domain_counts[d] = domain_counts.get(d, 0) + 1

        return selected

    def _select_pending_first(self, candidates: List[ContextFact]) -> List[ContextFact]:
        """Pending/blocked facts first, then by score."""
        # Separate pending from completed
        pending = [c for c in candidates if c.status in ("pending", "in_progress", "blocked")]
        others = [c for c in candidates if c.status not in ("pending", "in_progress", "blocked")]

        # Pack pending first (up to 60% budget)
        pending_budget = int(self.token_budget * 0.6)
        selected = []
        tokens_used = 0

        for c in pending:
            if tokens_used + c.token_cost > pending_budget:
                continue
            selected.append(c)
            tokens_used += c.token_cost

        # Fill remainder with scored facts
        for c in others:
            if c.score < self.min_score:
                continue
            if tokens_used + c.token_cost > self.token_budget:
                continue
            selected.append(c)
            tokens_used += c.token_cost

        return selected

    def _select_windowed(self, candidates: List[ContextFact]) -> List[ContextFact]:
        """Top facts + temporal neighbors."""
        # Use 70% budget for primary selection
        primary_budget = int(self.token_budget * 0.7)
        old_budget = self.token_budget
        self.token_budget = primary_budget
        primary = self._select_diverse(candidates)
        self.token_budget = old_budget

        # Get temporal neighbors for top facts
        tokens_used = sum(c.token_cost for c in primary)
        selected_ids = {c.fact_id for c in primary}
        neighbors = []

        for c in primary[:3]:  # only top 3
            window = self.retriever.get_context_window(c.fact_id, window=2)
            for w in window:
                if w.fact_id in selected_ids:
                    continue
                content = self.idx.reconstruct(w.fact_id) or w.summary
                token_cost = estimate_tokens(content) + FACT_OVERHEAD_TOKENS
                if tokens_used + token_cost > self.token_budget:
                    continue

                neighbors.append(ContextFact(
                    fact_id=w.fact_id,
                    summary=w.summary,
                    content=content,
                    score=w.score * 0.8,  # slightly lower for neighbors
                    token_cost=token_cost,
                    status=w.status,
                    tier=w.tier,
                    source_phase=w.source_phase,
                    domains=w.keywords.get("domain", []),
                    matched_dimensions=["temporal_window"],
                ))
                tokens_used += token_cost
                selected_ids.add(w.fact_id)

        return primary + neighbors

    # ── Strategy Detection ───────────────────────────────────────

    def _detect_strategy(self, query: str) -> str:
        """Auto-detect best strategy from query signals."""
        q = query.lower()

        # Episodic queries → windowed
        episodic = ["when", "what happened", "history", "timeline", "before",
                     "after", "last time", "remember", "cuándo", "recuerdas",
                     "qué pasó", "historial"]
        if any(s in q for s in episodic):
            return "windowed"

        # Status queries → pending_first
        status = ["pending", "todo", "unfinished", "pendiente", "falta",
                   "qué queda", "what's left", "blocked", "retomar"]
        if any(s in q for s in status):
            return "pending_first"

        # Short lookups → greedy
        if len(query.split()) <= 3:
            return "greedy"

        # Default → diverse
        return "diverse"

    # ── Formatting ───────────────────────────────────────────────

    @staticmethod
    def format_context(facts: List[ContextFact], compact: bool = False) -> str:
        """
        Format selected facts into injectable context block.

        Two modes:
          compact:  one-liners (for tight budgets)
          full:     structured with metadata
        """
        if not facts:
            return ""

        lines = ["[MEMORY CONTEXT]"]

        if compact:
            for i, f in enumerate(facts, 1):
                status_icon = {
                    "pending": "☐", "in_progress": "◐", "committed": "◆",
                    "completed": "✔", "abandoned": "✖", "blocked": "✖",
                }.get(f.status, "·")
                lines.append(f"{status_icon} {f.content}")
        else:
            # Group by status for readability
            pending = [f for f in facts if f.status in ("pending", "in_progress", "blocked")]
            active = [f for f in facts if f.status in ("committed",)]
            done = [f for f in facts if f.status in ("completed",)]
            other = [f for f in facts if f.status not in ("pending", "in_progress", "blocked", "committed", "completed")]

            if pending:
                lines.append("  UNFINISHED:")
                for f in pending:
                    domains = ", ".join(f.domains[:2]) if f.domains else ""
                    lines.append(f"    ☐ [{domains}] {f.content}")

            if active:
                lines.append("  ACTIVE:")
                for f in active:
                    domains = ", ".join(f.domains[:2]) if f.domains else ""
                    lines.append(f"    ◆ [{domains}] {f.content}")

            if done:
                lines.append("  CONTEXT:")
                for f in done:
                    domains = ", ".join(f.domains[:2]) if f.domains else ""
                    lines.append(f"    · [{domains}] {f.content}")

            if other:
                for f in other:
                    lines.append(f"    · {f.content}")

        lines.append("[/MEMORY CONTEXT]")
        return "\n".join(lines)

    # ── Phase 8B: Surface-First Retrieval ─────────────────────────

    def _get_surface_facts(self, query: str) -> List[ContextFact]:
        """
        Get pre-activated facts from the Surface Buffer (Phase 8A).
        If buffer is empty and auto_activate is on, trigger the Deep Layer.
        Returns ContextFact list (same format as retriever candidates).
        """
        if not self.deep_layer:
            return []

        # Check if surface buffer has active facts
        status = self.deep_layer.surface.get_surface_status()

        if status["active"] == 0 and self.auto_activate:
            # Auto-activate: run Deep Layer with the query as signal
            self.deep_layer.process(query)

        # Read from surface_buffer (unconsumed, not expired)
        conn = self.deep_layer.surface.conn
        rows = conn.execute("""
            SELECT sb.fact_id, sb.activation_score, sb.domain, sb.trigger_type,
                   sb.token_estimate,
                   qf.summary, qf.raw_content, qf.status, qf.storage_tier
            FROM surface_buffer sb
            JOIN quantum_facts qf ON sb.fact_id = qf.id
            WHERE sb.consumed = 0
            AND (sb.expires_at IS NULL OR sb.expires_at > CURRENT_TIMESTAMP)
            ORDER BY sb.activation_score DESC
        """).fetchall()

        facts = []
        for r in rows:
            content = r["raw_content"] or r["summary"] or ""
            facts.append(ContextFact(
                fact_id=r["fact_id"],
                summary=r["summary"] or "",
                content=content,
                score=r["activation_score"] * 1.2,  # boost: surface facts are pre-vetted
                token_cost=r["token_estimate"] or estimate_tokens(content) + FACT_OVERHEAD_TOKENS,
                status=r["status"] or "committed",
                tier=r["storage_tier"] or "hot",
                source_phase=8,  # Phase 8 origin
                domains=[r["domain"]] if r["domain"] else [],
                matched_dimensions=[r["trigger_type"] or "surface"],
            ))

        return facts

    def _merge_surface_and_retriever(
        self, surface_facts: List[ContextFact], retriever_facts: List[ContextFact]
    ) -> List[ContextFact]:
        """
        Merge surface (push) and retriever (pull) facts.
        Deduplicates by fact_id, preferring the higher score.
        """
        merged = {}

        # Surface facts first (priority)
        for f in surface_facts:
            merged[f.fact_id] = f

        # Retriever facts fill gaps
        for f in retriever_facts:
            if f.fact_id not in merged:
                merged[f.fact_id] = f
            elif f.score > merged[f.fact_id].score:
                merged[f.fact_id] = f

        # Sort by score
        result = sorted(merged.values(), key=lambda x: x.score, reverse=True)
        return result

    # ── Main Entry Point ─────────────────────────────────────────

    def select(self, query: str, strategy: str = "adaptive") -> ContextInjection:
        """
        Select optimal facts for context injection.
        Phase 8B: Surface-first, pull-fallback.

        Flow:
          1. Check Surface Buffer (pre-activated by Deep Layer)
          2. If surface covers budget → use it (zero-query path)
          3. If not → fallback to retriever for remaining budget
          4. Merge and deduplicate

        Args:
            query: The user's query or task description
            strategy: "greedy", "diverse", "windowed", "pending_first", "adaptive"

        Returns:
            ContextInjection with selected facts and formatted context
        """
        # Phase 8B: Surface-first path
        surface_budget = int(self.token_budget * self.surface_ratio)
        surface_facts = self._get_surface_facts(query)

        # Calculate surface token usage
        surface_tokens = 0
        surface_selected = []
        for f in surface_facts:
            if surface_tokens + f.token_cost > surface_budget:
                break
            surface_selected.append(f)
            surface_tokens += f.token_cost

        # Determine if we need retriever fallback
        remaining_budget = self.token_budget - surface_tokens
        retriever_facts = []
        source_mode = "surface_only"

        if remaining_budget > 200 and len(surface_selected) < 5:
            # Fallback: use retriever for remaining budget
            old_budget = self.token_budget
            self.token_budget = remaining_budget
            retriever_candidates = self._score_candidates(query, top_k=20)

            if strategy == "adaptive":
                detected_strategy = self._detect_strategy(query)
            else:
                detected_strategy = strategy

            if detected_strategy == "windowed":
                retriever_facts = self._select_windowed(retriever_candidates)
            elif detected_strategy == "pending_first":
                retriever_facts = self._select_pending_first(retriever_candidates)
            elif detected_strategy == "diverse":
                retriever_facts = self._select_diverse(retriever_candidates)
            else:
                retriever_facts = self._select_greedy(retriever_candidates)

            self.token_budget = old_budget
            source_mode = "hybrid"
        elif surface_selected:
            source_mode = "surface_only"
        else:
            # No surface data at all — pure retriever
            retriever_candidates = self._score_candidates(query, top_k=30)
            if strategy == "adaptive":
                strategy = self._detect_strategy(query)
            if strategy == "windowed":
                retriever_facts = self._select_windowed(retriever_candidates)
            elif strategy == "pending_first":
                retriever_facts = self._select_pending_first(retriever_candidates)
            elif strategy == "diverse":
                retriever_facts = self._select_diverse(retriever_candidates)
            else:
                retriever_facts = self._select_greedy(retriever_candidates)
            source_mode = "retriever_only"

        # Merge
        if source_mode == "hybrid":
            selected = self._merge_surface_and_retriever(surface_selected, retriever_facts)
        elif source_mode == "surface_only":
            selected = surface_selected
        else:
            selected = retriever_facts

        # Calculate totals
        total_tokens = sum(f.token_cost for f in selected)

        domain_counts = {}
        for f in selected:
            for d in f.domains:
                domain_counts[d] = domain_counts.get(d, 0) + 1

        # Detect final strategy name
        if strategy == "adaptive":
            strategy = self._detect_strategy(query)

        compact = self.token_budget < 1500

        injection = ContextInjection(
            facts=selected,
            total_tokens=total_tokens,
            budget=self.token_budget,
            budget_used_pct=(total_tokens / self.token_budget * 100) if self.token_budget > 0 else 0,
            query=query,
            strategy=f"{strategy}+{source_mode}",
            domain_diversity=domain_counts,
            metadata={
                "total_candidates": len(surface_facts) + len(retriever_facts),
                "surface_facts": len(surface_selected),
                "retriever_facts": len(retriever_facts),
                "source_mode": source_mode,
                "surface_tokens": surface_tokens,
                "min_score": self.min_score,
                "compact_mode": compact,
                "timestamp": datetime.now().isoformat(),
            },
        )
        injection.formatted_context = self.format_context(selected, compact=compact)
        return injection


# ── CLI ──────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dynamic Context Selector (Phase 4)")
    parser.add_argument("query", nargs="+", help="Query to select context for")
    parser.add_argument("--budget", type=int, default=2000, help="Token budget (default: 2000)")
    parser.add_argument("--strategy", choices=["adaptive", "greedy", "diverse", "windowed", "pending_first"],
                        default="adaptive")
    parser.add_argument("--compact", action="store_true", help="Force compact format")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    query = " ".join(args.query)

    selector = ContextSelector(token_budget=args.budget)
    selector.connect()

    try:
        injection = selector.select(query, strategy=args.strategy)

        if args.json:
            print(json.dumps(injection.to_dict(), indent=2))
        else:
            print(f"\n  CONTEXT SELECTION")
            print(f"  {'─' * 55}")
            print(f"  Query:       {injection.query}")
            print(f"  Strategy:    {injection.strategy}")
            print(f"  Budget:      {injection.budget} tokens")
            print(f"  Used:        {injection.total_tokens} tokens ({injection.budget_used_pct:.0f}%)")
            print(f"  Facts:       {len(injection.facts)} selected / {injection.metadata['total_candidates']} candidates")
            # Phase 8B source breakdown
            meta = injection.metadata
            if "source_mode" in meta:
                sf = meta.get("surface_facts", 0)
                rf = meta.get("retriever_facts", 0)
                mode = meta["source_mode"]
                print(f"  Source:      {mode} (surface={sf}, retriever={rf}, surface_tokens={meta.get('surface_tokens',0)})")
            if injection.domain_diversity:
                print(f"  Domains:     {injection.domain_diversity}")
            print()
            print(injection.formatted_context)
            print()

            excluded = injection.metadata['total_candidates'] - len(injection.facts)
            if excluded > 0:
                print(f"  ({excluded} candidates excluded by budget/diversity/score)")
            print()
    finally:
        selector.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Executor Router — keyword-based semi-automatic executor selection.

Input:  TaskCreateContext (title, goal, project, available executors)
Output: RouterRecommendation (recommended executor, confidence, reason, alternatives)

The router uses keyword matching against the task title and goal to produce
a recommendation.  The user MUST confirm before execution — no automatic dispatch.

Rules (v0.5):
  Architecture / ADR / review tasks → claude-code
  Complex implementation / large refactor → codex-cli
  Local open-source agent validation / backup / review → opencode
  Quick bug scan / small fix → deepseek-tui
  Hermes internal processes / adapter / scheduler → hermes-local

Confidence is reduced for unavailable executors; the router will fall back
to the next best available option.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from executors.types import (
    ExecutorId,
    TaskCreateContext,
    RouterRecommendation,
    ExecutorHealthStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

@dataclass
class RouteRule:
    """A single keyword-based routing rule."""
    executor: ExecutorId
    keywords: List[str]         # case-insensitive substring matches
    reason_template: str        # format string for the recommendation reason
    priority: int = 0           # higher priority = stronger match
    confidence: float = 0.85    # base confidence for this rule


# Ordered by priority (higher = checked first)
_DEFAULT_RULES: List[RouteRule] = [
    RouteRule(
        executor="claude-code",
        keywords=[
            "architecture", "architectural", "ADR", "architect",
            "design review", "design doc", "code review", "review",
            "blueprint", "system design", "technical spec", "tech spec",
            "proposal", "RFC", "design pattern",
        ],
        reason_template="Task involves architecture/design/review work — Claude Code excels at design reasoning",
        priority=10,
        confidence=0.90,
    ),
    RouteRule(
        executor="codex-cli",
        keywords=[
            "implement", "implementation", "refactor", "refactoring",
            "rewrite", "build", "feature", "create", "migrate",
            "complex", "large", "multi-file", "multi-module",
            "production", "API", "endpoint", "service",
        ],
        reason_template="Task involves complex implementation/refactoring — Codex CLI handles large codebases well",
        priority=9,
        confidence=0.85,
    ),
    RouteRule(
        executor="opencode",
        keywords=[
            "opencode", "open source", "open-source", "oss",
            "validate", "validation", "backup", "alternative",
            "compare", "comparison", "experiment", "prototype",
            "local", "offline", "self-hosted",
        ],
        reason_template="Task involves open-source agent validation/alternative implementation — OpenCode is a good fit",
        priority=8,
        confidence=0.80,
    ),
    RouteRule(
        executor="deepseek-tui",
        keywords=[
            "bug", "fix", "quick", "small", "simple",
            "patch", "hotfix", "typo", "lint",
            "scan", "audit", "find", "locate",
            "trivial", "minor", "cosmetic",
        ],
        reason_template="Task is a quick/small fix or scan — DeepSeek TUI is fast and low-cost",
        priority=7,
        confidence=0.88,
    ),
    RouteRule(
        executor="hermes-local",
        keywords=[
            "hermes", "gateway", "adapter",
            "orchestrator", "dispatcher", "agent router",
            "internal", "admin", "config", "configuration",
            "cron", "batch", "pipeline", "workflow",
            "telegram", "feishu", "lark", "webhook",
        ],
        reason_template="Task involves Hermes internal infrastructure — Hermes Local is the native executor",
        priority=5,
        confidence=0.82,
    ),
]

# Fallback: if nothing matches, use the first available executor from this list
_FALLBACK_ORDER: List[ExecutorId] = [
    "codex-cli", "claude-code", "opencode", "hermes-local", "deepseek-tui",
]


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_text(text: str, keywords: List[str]) -> float:
    """Return a match score (0.0–1.0) based on keyword overlap."""
    if not text or not keywords:
        return 0.0

    normalized = _normalize(text)
    hits = 0
    for kw in keywords:
        if _normalize(kw) in normalized:
            hits += 1

    if hits == 0:
        return 0.0

    # More hits = higher score, but diminishing returns
    ratio = hits / len(keywords)
    return min(0.3 + ratio * 0.7, 1.0)  # floor at 0.3 for any match


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class ExecutorRouter:
    """Keyword-based executor router.

    Usage::

        router = ExecutorRouter()
        ctx = TaskCreateContext(
            title="Refactor auth module",
            goal="Implement OAuth2 flow",
            available_executors=["claude-code", "codex-cli", "hermes-local"],
        )
        rec = router.route(ctx)
        print(rec.recommended_executor, rec.confidence, rec.reason)
    """

    def __init__(
        self,
        rules: Optional[List[RouteRule]] = None,
        fallback_order: Optional[List[ExecutorId]] = None,
    ):
        self._rules = sorted(
            rules or _DEFAULT_RULES,
            key=lambda r: -r.priority,
        )
        self._fallback_order = fallback_order or _FALLBACK_ORDER

    def route(
        self,
        ctx: TaskCreateContext,
        available_set: Optional[Set[ExecutorId]] = None,
    ) -> RouterRecommendation:
        """Produce a router recommendation for the given task context.

        Args:
            ctx: Task creation context (title, goal, available executors, etc.).
            available_set: Optional pre-filtered set of available executor IDs.
                           If None, uses ``ctx.available_executors``.

        Returns:
            RouterRecommendation with the best-match executor, confidence,
            reason, and alternatives.
        """
        available = available_set or set(ctx.available_executors)

        # Combine title and goal for keyword matching
        search_text = f"{ctx.title} {ctx.goal}"

        # Score each rule
        scored: List[tuple[float, RouteRule, float]] = []
        for rule in self._rules:
            text_score = _score_text(search_text, rule.keywords)
            if text_score > 0.0:
                combined = text_score * rule.confidence
                scored.append((combined, rule, text_score))

        # Sort by combined score (highest first), then priority
        scored.sort(key=lambda x: (-x[0], -x[1].priority))

        # Build alternatives list from scored rules (excluding top pick)
        alternatives: List[ExecutorId] = []
        for _, rule, _ in scored[1:]:
            if rule.executor not in alternatives:
                alternatives.append(rule.executor)

        # If no keyword match, use fallback
        if not scored:
            return self._fallback_recommendation(available, search_text)

        best_score, best_rule, best_text_score = scored[0]

        # Check if recommended executor is available
        recommended = best_rule.executor
        reason = best_rule.reason_template
        confidence = best_score

        if recommended not in available:
            # Downgrade: try alternatives or fallback
            confidence = min(confidence, 0.3)
            reason = (
                f"{best_rule.reason_template} "
                f"(but {recommended} is unavailable — "
                f"consider installing it or using an alternative)"
            )

            # Try to find the best available alternative from the scored list
            for alt_score, alt_rule, _ in scored[1:]:
                if alt_rule.executor in available:
                    recommended = alt_rule.executor
                    reason = (
                        f"Best match was {best_rule.executor} (unavailable). "
                        f"Falling back to {recommended}: {alt_rule.reason_template}"
                    )
                    confidence = min(alt_score, 0.70)
                    break
            else:
                # No scored alternative available — use fallback
                return self._fallback_recommendation(available, search_text)

        return RouterRecommendation(
            recommended_executor=recommended,
            confidence=round(confidence, 2),
            reason=reason,
            alternatives=[a for a in alternatives if a != recommended],
            source="keyword",
        )

    def _fallback_recommendation(
        self, available: Set[ExecutorId], search_text: str
    ) -> RouterRecommendation:
        """Return a fallback recommendation based on the priority order."""
        reason_parts = [f"No keyword match for: '{search_text[:80]}'. "]

        for executor in self._fallback_order:
            if executor in available:
                reason_parts.append(
                    f"Defaulting to {executor} (first available in fallback order)."
                )
                return RouterRecommendation(
                    recommended_executor=executor,
                    confidence=0.40,
                    reason="".join(reason_parts),
                    alternatives=[],
                    source="health_fallback",
                )

        # Nothing available at all
        return RouterRecommendation(
            recommended_executor="hermes-local",
            confidence=0.10,
            reason="No executors are available. Install at least one executor CLI.",
            alternatives=[],
            source="health_fallback",
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def create_default_router() -> ExecutorRouter:
    """Create a router with the default v0.5 rules."""
    return ExecutorRouter()

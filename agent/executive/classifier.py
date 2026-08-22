"""Objective classifier — pure heuristic, no LLM.

Classifies tokens into (goal_class, risk_profile, complexity).
Pure function. No side effects. No provider calls.
"""

from __future__ import annotations

from .types import ClassifiedObjective, Complexity, GoalClass, RiskProfile

# ──────────────────────────────────────────────────────────────────────
# Goal-class keyword sets (lower-case)
# ──────────────────────────────────────────────────────────────────────

GOAL_CLASS_KEYWORDS: dict[GoalClass, frozenset[str]] = {
    GoalClass.RESEARCH: frozenset({
        "investiga", "research", "estudia", "explora",
        "what", "why", "how",
        "documentation", "literature", "papers",
    }),
    GoalClass.BUILD: frozenset({
        "build", "implement", "create", "develop",
        "construye", "implementa",
        "code", "function", "module", "feature",
    }),
    GoalClass.ANALYZE: frozenset({
        "analyze", "analiza", "examine", "review", "study", "understand",
        "metrics", "kpi", "data",
    }),
    GoalClass.AUTOMATE: frozenset({
        "automate", "automatiza", "script", "cron", "schedule",
        "workflow", "pipeline",
    }),
    GoalClass.INTEGRATE: frozenset({
        "integrate", "integra", "connect", "conecta", "merge", "unify",
        "api", "webhook", "sync",
    }),
    GoalClass.OPTIMIZE: frozenset({
        "optimize", "optimiza", "improve", "performance", "speed", "latency",
        "cost", "throughput",
    }),
    GoalClass.DOCUMENT: frozenset({
        "document", "documenta", "write", "readme", "guide", "tutorial",
        "spec", "design",
    }),
    GoalClass.VERIFY: frozenset({
        "verify", "verifica", "audit", "test", "validate", "check",
        "compliance",
    }),
    GoalClass.MAINTAIN: frozenset({
        "maintain", "fix", "clean", "refactor", "migrate", "update",
        "cleanup",
    }),
    GoalClass.STRATEGIC: frozenset({
        "achieve", "consigue", "deliver", "ship", "launch",
        "obtain", "ensure", "make", "become", "transform", "convert",
    }),
}

HIGH_RISK_TOKENS = frozenset({
    "delete", "destroy", "production", "prod", "live",
    "customer", "client", "user-data", "pii", "personal",
    "payment", "banking", "financial", "fintech", "money",
    "compliance", "regulatory", "legal", "gdpr", "kyc", "aml",
    "password", "secret", "credential", "token",
    "merge", "deploy", "release",
})

MEDIUM_RISK_TOKENS = frozenset({
    "test", "staging", "internal", "experiment",
    "refactor", "migrate", "upgrade",
})

# Token count -> complexity bucket.
COMPLEXITY_TOKEN_COUNT: dict[Complexity, tuple[int, int | None]] = {
    Complexity.XS: (0, 3),
    Complexity.S: (4, 10),
    Complexity.M: (11, 30),
    Complexity.L: (31, 100),
    Complexity.XL: (101, None),
}


def classify_goal_class(tokens: list[str]) -> GoalClass:
    """Return the goal_class with the highest keyword match score.

    Tie-breaker: STRATEGIC wins over BUILD/OTHER when score is non-zero.
    """
    scores: dict[GoalClass, int] = {gc: 0 for gc in GoalClass}
    for tok in tokens:
        for gc, keywords in GOAL_CLASS_KEYWORDS.items():
            if tok in keywords:
                scores[gc] += 1
    best_score = max(scores.values())
    if best_score == 0:
        return GoalClass.OTHER
    # Tie-breaker: if STRATEGIC has any score, prefer it.
    if scores.get(GoalClass.STRATEGIC, 0) > 0:
        return GoalClass.STRATEGIC
    # Else return the first non-zero goal_class in the original order.
    for gc in GoalClass:
        if scores[gc] > 0:
            return gc
    return GoalClass.OTHER


def compute_risk_profile(
    goal_class: GoalClass,
    tokens: list[str],
    constraints: list[str] | None = None,
) -> RiskProfile:
    """Compute risk_profile from goal_class, tokens, and constraints."""
    has_high = any(tok in HIGH_RISK_TOKENS for tok in tokens)
    has_medium = any(tok in MEDIUM_RISK_TOKENS for tok in tokens)
    has_constraint = any(
        isinstance(c, str) and c.startswith("forbidden:") for c in (constraints or [])
    )
    if goal_class == GoalClass.STRATEGIC:
        if has_high:
            return RiskProfile.HIGH
        return RiskProfile.MEDIUM
    if has_high:
        return RiskProfile.HIGH
    if has_medium or has_constraint:
        return RiskProfile.MEDIUM
    return RiskProfile.LOW


def estimate_complexity(tokens: list[str]) -> Complexity:
    """Heuristic: bucket the token count."""
    n = len(tokens)
    for complexity, (lo, hi) in COMPLEXITY_TOKEN_COUNT.items():
        if n >= lo and (hi is None or n <= hi):
            return complexity
    return Complexity.XL  # fallback (shouldn't trigger)


def classify_objective(tokens: list[str]) -> ClassifiedObjective:
    """Classify tokens into (goal_class, risk_profile, complexity)."""
    goal_class = classify_goal_class(tokens)
    risk_profile = compute_risk_profile(goal_class, tokens)
    complexity = estimate_complexity(tokens)
    signal_tokens = tuple(
        t for t in tokens
        if any(t in kws for kws in GOAL_CLASS_KEYWORDS.values())
    )
    rationale = (
        f"goal_class={goal_class.value} (matched {len(signal_tokens)} signal tokens), "
        f"risk={risk_profile.value}, complexity={complexity.value}"
    )
    return ClassifiedObjective(
        goal_class=goal_class,
        risk_profile=risk_profile,
        estimated_complexity=complexity,
        rationale=rationale,
        signal_tokens=signal_tokens,
    )

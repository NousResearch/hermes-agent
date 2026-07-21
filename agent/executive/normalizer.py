"""Objective normalizer — pure heuristic, no LLM.

Converts a free-text objective into a NormalizedObjective using
keyword tokenization, stopword removal, classifier delegation, and
constraint extraction. Pure function with no side effects.
"""

from __future__ import annotations

import re
from typing import Iterable

from .classifier import classify_objective
from .types import (
    Complexity,
    GoalClass,
    NormalizedObjective,
    RiskProfile,
    compute_fingerprint,
    new_uuid,
    now_iso8601,
)

STOPWORDS = frozenset({
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or",
    "with", "by", "is", "are", "be", "this", "that", "it", "as",
})

# Default per-goal-class success criteria templates.
SUCCESS_CRITERIA_TEMPLATES: dict[GoalClass, list[str]] = {
    GoalClass.RESEARCH: ["Information about {topic} is documented"],
    GoalClass.BUILD: [
        "Implementation of {topic} is functional",
        "Tests for {topic} pass",
    ],
    GoalClass.ANALYZE: ["Analysis of {topic} is complete"],
    GoalClass.AUTOMATE: ["Process {topic} is automated"],
    GoalClass.INTEGRATE: ["Integration of {topic} is end-to-end tested"],
    GoalClass.OPTIMIZE: ["Optimization of {topic} is measured"],
    GoalClass.DOCUMENT: ["Documentation for {topic} is complete"],
    GoalClass.VERIFY: ["Verification of {topic} is documented"],
    GoalClass.MAINTAIN: ["Maintenance of {topic} is complete"],
    GoalClass.STRATEGIC: [
        "Strategic objective {topic} is broken into sub-goals",
        "All sub-goals of {topic} are completed",
        "Final result of {topic} is delivered",
    ],
    GoalClass.OTHER: ["Objective {topic} is addressed"],
}

# Domain knowledge tokens that trigger knowledge_requirements entries.
DOMAIN_KNOWLEDGE_TRIGGERS: dict[str, str] = {
    "customer": "kb:user_requirements",
    "client": "kb:user_requirements",
    "user": "kb:user_requirements",
    "compliance": "kb:regulatory",
    "regulation": "kb:regulatory",
    "legal": "kb:regulatory",
    "kyc": "kb:regulatory",
    "aml": "kb:regulatory",
    "gdpr": "kb:regulatory",
    "payment": "kb:financial",
    "banking": "kb:financial",
    "financial": "kb:financial",
    "fintech": "kb:financial",
    "money": "kb:financial",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split, drop stopwords/short tokens."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t and t not in STOPWORDS and len(t) > 1]


def extract_constraints(tokens: list[str]) -> list[str]:
    """Extract typed constraints from tokens (heuristic)."""
    constraints: list[str] = []
    for i, tok in enumerate(tokens):
        if tok == "no" and i + 1 < len(tokens):
            constraints.append(f"forbidden:{tokens[i + 1]}")
        elif tok == "max" and i + 1 < len(tokens):
            constraints.append(f"limit:{tokens[i + 1]}")
        elif tok in {"in", "with"} and i + 1 < len(tokens):
            constraints.append(f"context:{tokens[i + 1]}")
    return constraints


def generate_success_criteria(goal_class: GoalClass, tokens: list[str]) -> tuple[str, ...]:
    """Generate verifiable success criteria based on goal_class."""
    topic = " ".join(tokens[:3]) if tokens else "objective"
    templates = SUCCESS_CRITERIA_TEMPLATES.get(
        goal_class, SUCCESS_CRITERIA_TEMPLATES[GoalClass.OTHER]
    )
    return tuple(
        t.format(topic=topic) for t in templates
    )


def identify_knowledge_requirements(
    goal_class: GoalClass, tokens: list[str]
) -> list[str]:
    """Identify what knowledge the objective needs."""
    requirements: set[str] = {"memory:global"}
    if goal_class == GoalClass.STRATEGIC:
        requirements.update({"kb:domain", "kb:constraints"})
    elif goal_class == GoalClass.BUILD:
        requirements.add("kb:best_practices")
    elif goal_class == GoalClass.RESEARCH:
        requirements.add("kb:literature")
    for tok in tokens:
        if tok in DOMAIN_KNOWLEDGE_TRIGGERS:
            requirements.add(DOMAIN_KNOWLEDGE_TRIGGERS[tok])
    return tuple(sorted(requirements))


def build_execution_requirements(
    complexity: Complexity,
) -> dict:
    """Build execution_requirements sub-schema from complexity."""
    estimates = {
        Complexity.XS: (3, 30, 1.0),
        Complexity.S: (10, 90, 5.0),
        Complexity.M: (30, 240, 20.0),
        Complexity.L: (100, 1440, 100.0),
        Complexity.XL: (300, 4320, 500.0),
    }
    iterations, duration_min, cost_usd = estimates[complexity]
    return {
        "required_capabilities": [],
        "required_tools": [],
        "required_skills": [],
        "required_roles": [],
        "required_workflows": [],
        "required_providers": [],
        "estimated_iterations": iterations,
        "estimated_duration_minutes": duration_min,
        "estimated_cost_usd": cost_usd,
        "budget_policy": "standard",
    }


def derive_approval_requirements(
    risk_profile: RiskProfile, complexity: Complexity
) -> list[dict]:
    """Derive approval gates from risk and complexity."""
    approvals: list[dict] = []
    if risk_profile == RiskProfile.HIGH or complexity in {Complexity.L, Complexity.XL}:
        approvals.append({
            "gate": "HIGH_RISK_DRAFT",
            "approver": "user",
            "ttl_hours": 24,
        })
    return approvals


def normalize_objective(
    objective_text: str,
    *,
    constraints: list[str] | None = None,
    human_constraints: list[str] | None = None,
    user_id: str,
) -> NormalizedObjective:
    """Normalize objective_text into a NormalizedObjective.

    Pure function. No LLM. No provider calls. No side effects.
    """
    if not objective_text or not objective_text.strip():
        raise ValueError("objective_text must be non-empty")
    if not user_id:
        raise ValueError("user_id must be non-empty")
    if len(objective_text) > 10_000:
        objective_text = objective_text[:10_000]

    tokens = tokenize(objective_text)
    user_constraints = list(constraints or [])
    extracted = extract_constraints(tokens)
    merged_constraints = tuple(dict.fromkeys(user_constraints + extracted))

    classified = classify_objective(tokens)
    created_at = now_iso8601()
    obj_id = new_uuid()
    fingerprint = compute_fingerprint(
        objective_text, merged_constraints, user_id, created_at
    )

    success = tuple(generate_success_criteria(classified.goal_class, tokens))
    knowledge = identify_knowledge_requirements(classified.goal_class, tokens)
    exec_req = build_execution_requirements(classified.estimated_complexity)
    approvals = derive_approval_requirements(
        classified.risk_profile, classified.estimated_complexity
    )

    return NormalizedObjective(
        objective_id=obj_id,
        goal_class=classified.goal_class,
        constraints=merged_constraints,
        success_criteria=success,
        human_constraints=tuple(human_constraints or []),
        approval_requirements=tuple(approvals),
        risk_profile=classified.risk_profile,
        estimated_complexity=classified.estimated_complexity,
        knowledge_requirements=knowledge,
        execution_requirements=exec_req,
        created_at=created_at,
        created_by=user_id,
        parent_objective_id=None,
        session_id=None,
        fingerprint=fingerprint,
        schema_version="1.0",
    )

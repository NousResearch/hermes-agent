"""RiskClassification v1 — Phase 4A.

Pure heuristic that assigns one of 7 risk levels (R0-R6) to a
combination of:

* ``execution_contract`` (Phase 1 dict, optional)
* ``goal_linkage`` (Phase 2 dataclass, optional)
* ``objective_plan`` (Phase 3 dataclass, optional)
* ``orchestrator_preview`` (Phase 3 dataclass, optional)

Levels (highest first):

* **R6** — external / network / provider / API.
* **R5** — runtime / workers.
* **R4** — Kanban apply (creates tasks, no workers).
* **R3** — state_meta / local state writes.
* **R2** — sandbox (in-memory + temp).
* **R1** — report-dir only.
* **R0** — read-only (default; lowest risk).

The algorithm evaluates from **highest** risk (R6) to **lowest**
(R0) and returns the **first** matching level. This means R6
triggers R6; if absent, the algorithm descends to R5, R4, etc.

This module is **pure**: no side effects, no LLM calls, no network
calls, no subprocess, no DB writes. Reads only.

See ``.hermes/reports/hermes_executive_v2_phase4a_policy_approval_gates_design/risk_classification_v1.md``
for the design rationale and the full test plan.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .types import (
    GoalLinkage,
    ObjectivePlan,
    OrchestratorPlanPreview,
    RiskLevel,
)


# ──────────────────────────────────────────────────────────────────────
# Risk thresholds (canonical constants)
# ──────────────────────────────────────────────────────────────────────

# Risk score thresholds for level promotion.
_RISK_THRESHOLD_R6 = 0.95
_RISK_THRESHOLD_R5 = 0.80
_RISK_THRESHOLD_R4 = 0.60
_RISK_THRESHOLD_R3 = 0.30

# Keywords that flag external/network intent in subgoal titles or intents.
_EXTERNAL_KEYWORDS = (
    "external",
    "network",
    "api",
    "provider",
    "webhook",
)

# Worker-profile identifiers that flag worker-spawn intent.
_WORKER_PROFILES = (
    "worker",
    "background",
    "async",
    "ci",
)

# Subgoal intents that imply Kanban-like apply.
_KANBAN_INTENTS = (
    "BUILD",
    "AUTOMATE",
)

# Intent values that map to R2 (sandbox).
_R2_INTENTS = ("RESEARCH",)

# Intent values that map to R1 (report-dir).
_R1_INTENTS = ("ANALYZE", "DOCUMENT")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _clamp_score(value: Any) -> float:
    """Clamp a numeric value to [0.0, 1.0]. Non-numeric → 0.0."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _has_component(components: Mapping[str, Any], name: str) -> bool:
    """Return True if the named risk component is > 0.

    Missing or non-numeric entries are treated as 0.
    """
    try:
        return float(components.get(name, 0) or 0) > 0.0
    except (TypeError, ValueError):
        return False


def _intent_from_goal_linkage(
    goal_linkage: Optional[GoalLinkage],
) -> Optional[str]:
    """Extract a coarse intent label from a GoalLinkage's goal_text.

    Heuristic: returns the goal_text itself (caller compares keywords),
    or ``None`` if the linkage is absent.
    """
    if goal_linkage is None:
        return None
    text = getattr(goal_linkage, "goal_text", "") or ""
    return text.strip().upper() or None


def _has_external_intent(
    subgoals: tuple,
    goal_linkage: Optional[GoalLinkage],
) -> bool:
    """Return True if any subgoal or linkage hint implies external."""
    for sg in subgoals:
        title = (getattr(sg, "title", "") or "").lower()
        for kw in _EXTERNAL_KEYWORDS:
            if kw in title:
                return True
        intent = (getattr(sg, "intent", "") or "").lower()
        if intent in ("external", "network", "provider", "api"):
            return True
    if goal_linkage is not None:
        text = (getattr(goal_linkage, "goal_text", "") or "").lower()
        for kw in _EXTERNAL_KEYWORDS:
            if kw in text:
                return True
    return False


def _has_worker_intent(subgoals: tuple) -> bool:
    """Return True if any subgoal has a worker-like assigned profile.

    ``assigned_profile`` is a non-Phase-3 attribute. We accept it
    defensively (return False if absent on every subgoal).
    """
    for sg in subgoals:
        profile = getattr(sg, "assigned_profile", None)
        if profile is None:
            profile = getattr(sg, "profile", None)
        if profile is None:
            continue
        if str(profile).lower() in _WORKER_PROFILES:
            return True
    return False


def _has_kanban_intent(subgoals: tuple) -> bool:
    """Return True if any subgoal has Kanban-like intent."""
    for sg in subgoals:
        intent = getattr(sg, "intent", "") or ""
        if intent.upper() in _KANBAN_INTENTS:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def classify_risk(
    execution_contract: Optional[Mapping[str, Any]] = None,
    goal_linkage: Optional[GoalLinkage] = None,
    objective_plan: Optional[ObjectivePlan] = None,
    orchestrator_preview: Optional[OrchestratorPlanPreview] = None,
) -> RiskLevel:
    """Compute the RiskLevel for a Phase 3 plan.

    Pure: no side effects, no IO, no LLM. ``execution_contract`` is a
    dict (Phase 1 ``ExecutionContract.v1.to_dict()``); ``goal_linkage``,
    ``objective_plan``, ``orchestrator_preview`` are Phase 2/3 dataclasses.

    The algorithm is deterministic: same inputs → same RiskLevel.
    """
    contract = execution_contract or {}
    if not isinstance(contract, Mapping):
        contract = {}

    risk_score = _clamp_score(contract.get("risk_score", 0.0))
    risk_components = contract.get("risk_components", {}) or {}
    if not isinstance(risk_components, Mapping):
        risk_components = {}

    has_financial = _has_component(risk_components, "financial")
    has_regulatory = _has_component(risk_components, "regulatory")
    has_customer_facing = _has_component(risk_components, "customer_facing")
    has_irreversibility = _has_component(risk_components, "irreversibility")
    has_data_sensitivity = _has_component(risk_components, "data_sensitivity")

    subgoals = (
        tuple(objective_plan.subgoals) if objective_plan is not None else ()
    )

    has_external_intent = _has_external_intent(subgoals, goal_linkage)
    has_worker_intent = _has_worker_intent(subgoals)
    has_kanban_intent = _has_kanban_intent(subgoals)

    # R6: external / network / provider / API. Also triggered by
    # risk_score >= 0.95 or any financial or data-sensitivity component.
    if (
        has_external_intent
        or risk_score >= _RISK_THRESHOLD_R6
        or has_financial
        or has_data_sensitivity
    ):
        return RiskLevel.R6

    # R5: workers. Also triggered by risk_score >= 0.8 or irreversibility.
    if (
        has_worker_intent
        or risk_score >= _RISK_THRESHOLD_R5
        or has_irreversibility
    ):
        return RiskLevel.R5

    # R4: Kanban apply. Also triggered by risk_score >= 0.6,
    # customer_facing, or regulatory components.
    if (
        has_kanban_intent
        or risk_score >= _RISK_THRESHOLD_R4
        or has_customer_facing
        or has_regulatory
    ):
        return RiskLevel.R4

    # R3: state_meta / local state. Any non-zero risk component or
    # risk_score >= 0.3 implies local writes.
    if risk_score >= _RISK_THRESHOLD_R3 or risk_components:
        return RiskLevel.R3

    # R2: sandbox (in-memory + temp). RESEARCH intent falls here.
    intent = _intent_from_goal_linkage(goal_linkage)
    if intent is not None and intent in _R2_INTENTS:
        return RiskLevel.R2

    # R1: report-dir only. ANALYZE / DOCUMENT intents fall here.
    if intent is not None and intent in _R1_INTENTS:
        return RiskLevel.R1

    # R0: read-only (default; lowest risk).
    return RiskLevel.R0


__all__ = [
    "classify_risk",
]
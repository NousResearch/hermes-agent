"""Phase 3 minimal planner.

Pure heuristic module that takes a goal_text + ExecutionContract.v1
and produces an ordered list of ``PlannerSubgoal``. No LLM, no
Orchestrator, no Kanban, no DAG. The output is consumed by
``orchestrator_preview.py`` which produces a
``OrchestratorPlanPreview``.

Public surface:
- ``decompose_goal_to_subgoals(...)`` — pure mapping.
- ``map_subgoals_to_task_specs(...)`` — reuses ``TaskSpec`` from
  ``agent.orchestrator_interface`` (byte-intact).
- ``compute_plan_fingerprint(...)`` — stable sha256 of canonical plan.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Optional

from .types import PlannerSubgoal, now_iso8601

# Optional import of TaskSpec from existing orchestrator_interface.
# This is the only "real" Orchestrator symbol we touch (read-only).
try:
    from agent.orchestrator_interface import TaskSpec
    _TASK_SPEC_AVAILABLE = True
except Exception:
    TaskSpec = None  # type: ignore[assignment]
    _TASK_SPEC_AVAILABLE = False

# Intent keywords for classification.
INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "RESEARCH": (
        "investiga", "search", "find", "look up", "analyze",
        "study", "research", "examina", "evalúa", "documenta",
    ),
    "BUILD": (
        "implementa", "build", "create", "develop", "code",
        "write", "ship", "crea", "desarrolla", "programa",
    ),
    "AUTOMATE": (
        "automate", "script", "workflow", "schedule", "trigger",
        "automatiza", "orquesta", "lanza", "programa",
    ),
    "STRATEGIC": (
        "consigue", "achieve", "deliver", "accomplish",
        "bancarizar", "lanza", "ship", "go to market",
    ),
}

HIGH_RISK_KEYWORDS: tuple[str, ...] = (
    "payment", "banking", "production", "delete", "pii",
    "personal", "credential", "secret", "deploy", "lanzar",
    "pagar", "eliminar", "destruir",
)

MEDIUM_RISK_KEYWORDS: tuple[str, ...] = (
    "staging", "test", "demo", "internal", "preview",
    "pruebas", "demo",
)


# ── Heuristics ──────────────────────────────────────────────────────

def _classify_intent(criterion_text: str) -> str:
    text = (criterion_text or "").lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return intent
    return "OTHER"


def _classify_risk(criterion_text: str, base_risk: float) -> str:
    if base_risk >= 0.7:
        return "high"
    if base_risk >= 0.4:
        return "medium"
    text = (criterion_text or "").lower()
    if any(kw in text for kw in HIGH_RISK_KEYWORDS):
        return "high"
    if any(kw in text for kw in MEDIUM_RISK_KEYWORDS):
        return "medium"
    return "low"


def _approval_required(
    criterion_text: str,
    base_risk: float,
    approval_requirements: Iterable[dict],
) -> bool:
    if base_risk >= 0.7:
        return True
    for ar in approval_requirements or ():
        if "STRATEGIC" in str(ar.get("gate", "")):
            return True
    text = (criterion_text or "").lower()
    if any(
        kw in text
        for kw in ("payment", "production", "delete", "deploy", "lanzar")
    ):
        return True
    return False


def _derive_title(criterion: str) -> str:
    text = (criterion or "").strip()
    if text.lower().startswith("objective:"):
        text = text[len("objective:"):].strip()
    if len(text) > 80:
        text = text[:77] + "..."
    return text or "(untitled subgoal)"


def _derive_iterations(budget_max_iterations: Any, count: int) -> int:
    base = (
        budget_max_iterations
        if isinstance(budget_max_iterations, int) and budget_max_iterations > 0
        else 10
    )
    return max(1, base // max(1, count))


def _derive_timeout(budget_max_duration_minutes: Any, count: int) -> int:
    base_minutes = (
        budget_max_duration_minutes
        if isinstance(budget_max_duration_minutes, int)
        and budget_max_duration_minutes > 0
        else 30
    )
    return max(60, (base_minutes * 60) // max(1, count))


# ── Public functions ───────────────────────────────────────────────

def decompose_goal_to_subgoals(
    goal_text: str,
    goal_contract: Any,  # GoalContract from hermes_cli.goals (or stub)
    execution_contract: dict,
    *,
    risk_score: float = 0.0,
    max_subgoals: int = 8,
) -> list[PlannerSubgoal]:
    """Pure heuristic: produce a linear list of PlannerSubgoal.

    The number of subgoals equals ``len(success_criteria)`` (capped at
    ``max_subgoals``). One subgoal per success_criterion.
    """
    if max_subgoals <= 0:
        max_subgoals = 8
    success_criteria: list[str] = list(
        execution_contract.get("success_criteria") or ()
    )
    approval_requirements: list[dict] = list(
        execution_contract.get("approval_requirements") or ()
    )
    hard: tuple[str, ...] = tuple(execution_contract.get("hard_constraints") or ())
    soft: tuple[str, ...] = tuple(execution_contract.get("soft_constraints") or ())
    budget: dict = execution_contract.get("budget", {}) or {}
    base_risk = max(0.0, min(1.0, risk_score))

    constraints: tuple[str, ...] = hard + soft

    if not success_criteria:
        return []

    count = min(len(success_criteria), max_subgoals)
    iterations_per = _derive_iterations(budget.get("max_iterations"), count)
    timeout_per = _derive_timeout(budget.get("max_duration_minutes"), count)

    subgoals: list[PlannerSubgoal] = []
    # ── B1 EvidencePack gate (Gate C; default OFF; no-op when absent) ──
    # When the execution contract carries an evidence_pack_summary whose
    # prefix signals human/expert review is required, every subgoal
    # gets a [GATED: ...] marker appended to expected_output. The
    # marker is deterministic and contributes to the plan fingerprint.
    ep_summary = (execution_contract.get("evidence_pack_summary") or "").strip()
    ep_gated = ep_summary.startswith(
        ("[REQUIRES_HUMAN]", "[NEEDS_EXPERT_REVIEW]")
    )
    gated_suffix = " [GATED: evidence pack requires human review]"
    for i, criterion in enumerate(success_criteria[:count]):
        # Classify intent from the full criterion text (not the
        # truncated title), so the classifier sees the full context.
        expected_output_base = (
            (criterion or "").strip()[:200] or "(no output defined)"
        )
        if ep_gated:
            expected_output_base = expected_output_base + gated_suffix
        sg = PlannerSubgoal(
            id=f"sg-{i}",
            title=_derive_title(criterion),
            intent=_classify_intent(criterion),
            constraints=constraints,
            expected_output=expected_output_base,
            risk_level=_classify_risk(criterion, base_risk),
            approval_required=_approval_required(
                criterion, base_risk, approval_requirements
            ),
            estimated_iterations=iterations_per,
            timeout_seconds=timeout_per,
            source_criterion_index=i,
        )
        subgoals.append(sg)
    return subgoals


def map_subgoals_to_task_specs(
    subgoals: list[PlannerSubgoal],
) -> list[Any]:
    """Map PlannerSubgoal -> TaskSpec (from agent.orchestrator_interface).

    Reuses ``TaskSpec`` (byte-intact). ``task_id`` is empty (Phase 3 =
    preview only). Dependencies are empty (linear, no DAG).
    """
    if not _TASK_SPEC_AVAILABLE:
        return []
    specs: list[Any] = []
    for sg in subgoals:
        spec = TaskSpec(
            task_id="",
            description=sg.title,
            assigned_profile=sg.intent.lower(),
            inputs={
                "criterion_index": sg.source_criterion_index,
                "constraints": list(sg.constraints),
            },
            expected_outputs=[sg.expected_output],
            dependencies=[],
            timeout_s=sg.timeout_seconds,
            requires_user_input=sg.approval_required,
            approval_id=None,
        )
        specs.append(spec)
    return specs


def compute_plan_fingerprint(
    objective_id: str,
    subgoals: list[PlannerSubgoal],
    task_specs: list[Any],
) -> str:
    """Stable sha256 of canonical plan inputs (idempotent)."""
    canonical = json.dumps(
        {
            "objective_id": objective_id,
            "subgoals": [s.to_dict() for s in subgoals],
            "task_specs": [
                s.to_dict() if hasattr(s, "to_dict") else s.__dict__
                for s in task_specs
            ],
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

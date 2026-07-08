"""ExecutionContract.v1 builder.

Pure function that takes a NormalizedObjective, ClassifiedObjective,
and CapabilityDiscovery, and produces an ExecutionContractV1 DRAFT.

Phase 1 produces DRAFT only. The contract is an output artifact, not
an input to runtime.
"""

from __future__ import annotations

from typing import Iterable

from .types import (
    ApprovalRequirement,
    BudgetPolicy,
    CapabilityDiscovery,
    ClassifiedObjective,
    Complexity,
    ExecutionContractV1,
    GoalClass,
    NormalizedObjective,
    RiskComponents,
    RiskProfile,
    compute_contract_fingerprint,
    new_uuid,
    now_iso8601,
)


# Map complexity to default budget policy.
COMPLEXITY_BUDGET: dict[Complexity, BudgetPolicy] = {
    Complexity.XS: BudgetPolicy("standard", 10, 30, 5.0),
    Complexity.S: BudgetPolicy("standard", 30, 90, 20.0),
    Complexity.M: BudgetPolicy("standard", 100, 240, 100.0),
    Complexity.L: BudgetPolicy("strict", 300, 1440, 500.0),
    Complexity.XL: BudgetPolicy("strict", 1000, 4320, 2000.0),
}


def _extract_tokens(normalized: NormalizedObjective) -> set[str]:
    tokens: set[str] = set()
    for c in normalized.constraints:
        tokens.update(c.lower().split())
    for sc in normalized.success_criteria:
        tokens.update(sc.lower().split())
    for hc in normalized.human_constraints:
        tokens.update(hc.lower().split())
    return tokens


def compute_risk_components(
    classified: ClassifiedObjective,
    normalized: NormalizedObjective,
) -> RiskComponents:
    """Compute risk components from classified + normalized tokens."""
    tokens = _extract_tokens(normalized)
    # STRATEGIC goal class itself bumps customer_facing to 1.0
    # (multi-step objectives inherently cross boundaries).
    cf = 1.0 if (tokens & {
        "customer", "client", "user", "production", "live", "deploy"
    } or classified.goal_class == GoalClass.STRATEGIC) else 0.0
    return RiskComponents(
        financial=1.0 if tokens & {
            "payment", "banking", "financial", "money", "fintech"
        } else 0.0,
        regulatory=1.0 if tokens & {
            "compliance", "regulatory", "legal", "gdpr", "kyc", "aml"
        } else 0.0,
        customer_facing=cf,
        irreversibility=1.0 if tokens & {
            "delete", "destroy", "remove", "drop"
        } else 0.0,
        data_sensitivity=1.0 if tokens & {
            "pii", "personal", "secret", "password", "credential", "token"
        } else 0.0,
    )


def build_knowledge_summary_text(discovered: CapabilityDiscovery) -> str:
    """Build a human-readable knowledge summary from the discovery."""
    if not discovered.candidates:
        return "No P0/P1 sources returned relevant candidates."
    top = sorted(discovered.candidates, key=lambda c: c.match_score, reverse=True)[:3]
    lines = [f"Top matches ({len(discovered.candidates)} total):"]
    for c in top:
        lines.append(f"  - {c.id} (score {c.match_score:.2f})")
    return "\n".join(lines)


def build_execution_contract_v1(
    normalized: NormalizedObjective,
    classified: ClassifiedObjective,
    discovered: CapabilityDiscovery,
    *,
    user_id: str,
) -> ExecutionContractV1:
    """Build the ExecutionContract.v1 DRAFT from Phase 1 inputs.

    Pure function. No side effects. No LLM.
    """
    risk_components = compute_risk_components(classified, normalized)
    budget = COMPLEXITY_BUDGET[classified.estimated_complexity]
    # Inherit approval_requirements from normalized. If empty, derive from
    # classified.risk_profile and classified.estimated_complexity.
    if normalized.approval_requirements:
        approvals = tuple(
            ApprovalRequirement(**ar) for ar in normalized.approval_requirements
        )
    else:
        from .normalizer import derive_approval_requirements as _derive
        approvals = tuple(
            ApprovalRequirement(**ar)
            for ar in _derive(classified.risk_profile, classified.estimated_complexity)
        )
    required_capabilities = tuple(
        c.id for c in discovered.candidates
        if c.kind in ("skill", "tool", "module")
    )
    required_tools = tuple(
        c.id for c in discovered.candidates if c.kind == "tool"
    )
    required_skills = tuple(
        c.id for c in discovered.candidates if c.kind == "skill"
    )
    knowledge_keys = tuple(c.source_path for c in discovered.candidates)

    hard_constraints = tuple(
        c for c in normalized.constraints
        if c.startswith("forbidden:") or c.startswith("limit:")
    )
    soft_constraints = tuple(
        c for c in normalized.constraints
        if c not in hard_constraints
    )

    return ExecutionContractV1(
        contract_version="1.0",
        contract_id=new_uuid(),
        objective_id=normalized.objective_id,
        goal_id=None,  # Phase 1: not wired
        fingerprint=compute_contract_fingerprint(
            normalized.objective_id, normalized.fingerprint
        ),
        required_capabilities=required_capabilities,
        required_tools=required_tools,
        required_skills=required_skills,
        required_roles=(),
        required_workflows=(),
        required_providers=(),
        knowledge_summary_keys=knowledge_keys,
        knowledge_summary_text=build_knowledge_summary_text(discovered),
        hard_constraints=hard_constraints,
        soft_constraints=soft_constraints,
        approval_requirements=tuple(ar.__dict__ for ar in approvals),
        risk_components=risk_components.__dict__,
        risk_score=risk_components.total,
        budget=budget.__dict__,
        execution_strategy="sequential",
        rollback_strategy="manual",
        planner_inputs_sub_goals=(),
        planner_inputs_success_criteria=normalized.success_criteria,
        planner_inputs_hard_constraints=hard_constraints,
        planner_inputs_soft_constraints=soft_constraints,
        planner_inputs_preferred_workflow=None,
        planner_inputs_preferred_role=None,
        scheduler_hints_priority="medium",
        scheduler_hints_deadline=None,
        scheduler_hints_blocking_objectives=(),
        scheduler_hints_parallelism_allowed=False,
        success_criteria=normalized.success_criteria,
        verification_method="judge",
        verification_timeout_minutes=60,
        judge_model=None,
        evidence_required=True,
        created_at=now_iso8601(),
        created_by=user_id,
    )

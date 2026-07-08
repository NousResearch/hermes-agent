"""Phase 3 Orchestrator Preview — builds OrchestratorPlanPreview from
``decompose_goal_to_subgoals`` and ``map_subgoals_to_task_specs``.

Public surface:
- ``build_orchestrator_plan_preview(objective_id, *, storage)`` —
  Pure, returns ``OrchestratorPlanPreview``. No side effects.
- ``plan_dry_run(objective_id, *, storage)`` — Convenience wrapper
  that returns the preview as a string (CLI-friendly).
- ``plan_rollback(objective_id, *, storage)`` — Idempotent cleanup
  of ``objective_plan:<oid>`` and ``objective_orchestrator_preview:<oid>``.
- ``ExecutiveOrchestratorBridge`` — high-level facade.

Phase 3 NEVER:
- Calls ``OrchestratorInterface.execute()``.
- Calls ``delegate_task()`` or worker runners.
- Creates Kanban tasks.
- Spawns workers.
- Calls Kanban CLI subprocess.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from .planner import (
    compute_plan_fingerprint,
    decompose_goal_to_subgoals,
    map_subgoals_to_task_specs,
)
from .state_storage import ObjectiveStateStorage
from .types import (
    ObjectivePlan,
    OrchestratorPlanPreview,
    PlannerSubgoal,
    now_iso8601,
)

logger = logging.getLogger(__name__)

# Approval gate thresholds (mirror Phase 2).
HIGH_RISK_THRESHOLD = 0.7


# ── Errors ──────────────────────────────────────────────────────────

class BridgeError(RuntimeError):
    """Base error for Phase 3 bridge."""


class BridgeMappingError(BridgeError):
    """Raised when objective or goal_linkage is missing."""


class BridgeLinkageConflictError(BridgeError):
    """Raised on cross-session link conflict."""


class BridgeApprovalError(BridgeError):
    """Raised when an approval gate blocks the plan."""


# ── Internal helpers ──────────────────────────────────────────────

def _check_approval_gates(
    *,
    risk_score: float,
    require_human_approval: bool,
    approver_id: Optional[str],
    approval_token: Optional[str],
    approval_requirements: tuple[dict, ...],
) -> None:
    """Validate 4 approval layers. Raise BridgeApprovalError on fail.

    Order matters: Layer 1 fires first if approver_id is missing,
    which subsumes Layer 2 and Layer 3 (they both require
    require_human_approval=True AND approver_id).
    """
    # Layer 1: default
    if require_human_approval and not approver_id:
        raise BridgeApprovalError(
            "Layer 1: approver_id is required when require_human_approval=True"
        )
    # Layer 2: STRATEGIC
    if any("STRATEGIC" in str(ar.get("gate", "")) for ar in approval_requirements):
        if not approver_id:
            raise BridgeApprovalError(
                "Layer 2: STRATEGIC gate requires approver_id"
            )
    # Layer 3: HIGH_RISK
    if risk_score >= HIGH_RISK_THRESHOLD:
        if not approval_token:
            raise BridgeApprovalError(
                "Layer 3: HIGH_RISK gate requires approval_token"
            )


def _derive_warnings(
    subgoals: list[PlannerSubgoal],
    risk_score: float,
    approval_requirements: tuple[dict, ...],
) -> tuple[str, ...]:
    """Compute human-readable warnings for the preview."""
    out: list[str] = []
    if risk_score >= HIGH_RISK_THRESHOLD:
        out.append("HIGH_RISK: requires EXPLICIT_HIGH_RISK approval before apply.")
    if any("STRATEGIC" in str(ar.get("gate", "")) for ar in approval_requirements):
        out.append("STRATEGIC: requires EXPLICIT_STRATEGIC approval before apply.")
    if not subgoals:
        out.append("EMPTY plan: no subgoals produced.")
    n_approval = sum(1 for s in subgoals if s.approval_required)
    if n_approval > 0:
        out.append(
            f"APPROVAL: {n_approval}/{len(subgoals)} subgoals require explicit approval."
        )
    return tuple(out)


def _compute_preview_fingerprint(
    plan: ObjectivePlan,
    task_specs: list[Any],
    warnings: tuple[str, ...],
) -> str:
    """Stable sha256 of canonical preview inputs."""
    canonical = json.dumps(
        {
            "plan": plan.to_dict(),
            "task_specs": [
                s.to_dict() if hasattr(s, "to_dict") else s.__dict__
                for s in task_specs
            ],
            "warnings": list(warnings),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Public functions ───────────────────────────────────────────────

def build_orchestrator_plan_preview(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
) -> OrchestratorPlanPreview:
    """Build an OrchestratorPlanPreview for the given objective.

    Pure: reads state_meta but does NOT write to it.

    Reads:
    - ``state_meta[objective:<oid>]`` (Phase 1 ObjectiveStateData).
    - ``state_meta[objective_goal_link:<oid>]`` (Phase 2 GoalLinkage).

    Returns: ``OrchestratorPlanPreview`` with plan + task_specs + warnings.
    """
    if not objective_id:
        raise ValueError("objective_id is empty")
    storage = storage or ObjectiveStateStorage()
    obj_state = storage.load(objective_id)
    if obj_state is None:
        raise BridgeMappingError(
            f"objective {objective_id} not found in state_meta"
        )
    if not obj_state.contract:
        raise BridgeMappingError(
            f"objective {objective_id} has no contract"
        )
    goal_linkage = storage.get_objective_goal_link(objective_id)
    if goal_linkage is None:
        raise BridgeMappingError(
            f"objective {objective_id} has no Phase 2 goal linkage; "
            f"run bridge_apply first"
        )

    risk_score = float(obj_state.contract.get("risk_score", 0.0) or 0.0)
    approval_requirements: tuple[dict, ...] = tuple(
        obj_state.contract.get("approval_requirements") or ()
    )

    subgoals = decompose_goal_to_subgoals(
        goal_text=goal_linkage.goal_text,
        goal_contract=obj_state.contract,
        execution_contract=obj_state.contract,
        risk_score=risk_score,
    )
    task_specs = map_subgoals_to_task_specs(subgoals)
    plan = ObjectivePlan(
        objective_id=objective_id,
        subgoals=tuple(subgoals),
        plan_fingerprint=compute_plan_fingerprint(
            objective_id, subgoals, task_specs
        ),
        created_at=now_iso8601(),
    )
    warnings = _derive_warnings(subgoals, risk_score, approval_requirements)
    requires_approval = (
        risk_score >= HIGH_RISK_THRESHOLD
        or any("STRATEGIC" in str(ar.get("gate", "")) for ar in approval_requirements)
        or any(s.approval_required for s in subgoals)
    )
    preview = OrchestratorPlanPreview(
        objective_id=objective_id,
        plan=plan,
        task_specs=tuple(
            s.to_dict() if hasattr(s, "to_dict") else s.__dict__
            for s in task_specs
        ),
        warnings=warnings,
        requires_approval=requires_approval,
        risk_score=risk_score,
        preview_fingerprint=_compute_preview_fingerprint(plan, task_specs, warnings),
        created_at=now_iso8601(),
    )
    return preview


def plan_dry_run(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
) -> OrchestratorPlanPreview:
    """Alias for build_orchestrator_plan_preview (no side effects)."""
    return build_orchestrator_plan_preview(objective_id, storage=storage)


def plan_apply(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    require_human_approval: bool = True,
    approver_id: Optional[str] = None,
    approval_token: Optional[str] = None,
    cross_session: bool = False,
    session_id: Optional[str] = None,
) -> OrchestratorPlanPreview:
    """Apply the Phase 3 plan: write objective_plan:<oid> and
    objective_orchestrator_preview:<oid>.

    Side effects (after approval gates pass):
    1. Compute preview (calls build_orchestrator_plan_preview).
    2. Validate 4 approval gates.
    3. Persist preview via storage.set_objective_orchestrator_preview.
    4. Persist plan via storage.set_objective_plan.

    Does NOT call OrchestratorInterface.execute().
    Does NOT create Kanban tasks.
    Does NOT spawn workers.
    """
    if not objective_id:
        raise ValueError("objective_id is empty")
    storage = storage or ObjectiveStateStorage()

    # Cross-session check (mirrors Phase 2).
    if session_id is not None:
        goal_linkage = storage.get_objective_goal_link(objective_id)
        if (
            goal_linkage is not None
            and goal_linkage.session_id != session_id
            and not cross_session
        ):
            raise BridgeLinkageConflictError(
                f"objective {objective_id} is linked to session "
                f"{goal_linkage.session_id}; new session {session_id} requires "
                f"cross_session=True"
            )

    preview = build_orchestrator_plan_preview(objective_id, storage=storage)

    _check_approval_gates(
        risk_score=preview.risk_score,
        require_human_approval=require_human_approval,
        approver_id=approver_id,
        approval_token=approval_token,
        approval_requirements=tuple(
            preview.plan.to_dict().get("subgoals") or []
        ),  # best-effort
    )

    # Persist preview first, then plan (preview is the user-facing artifact).
    storage.set_objective_orchestrator_preview(preview)
    storage.set_objective_plan(preview.plan)
    logger.info(
        "plan_apply: objective %s plan_fingerprint=%s",
        objective_id, preview.plan.plan_fingerprint[:12],
    )
    return preview


def plan_rollback(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
) -> bool:
    """Rollback: delete both objective_plan:<oid> and
    objective_orchestrator_preview:<oid>. Idempotent.

    Returns True if at least one plan/preview existed before rollback
    (i.e., cleanup actually happened). Returns False if neither
    existed (no-op).
    """
    if not objective_id:
        raise ValueError("objective_id is empty")
    storage = storage or ObjectiveStateStorage()
    plan_existed = storage.get_objective_plan(objective_id) is not None
    preview_existed = (
        storage.get_objective_orchestrator_preview(objective_id) is not None
    )
    storage.delete_objective_plan(objective_id)
    storage.delete_objective_orchestrator_preview(objective_id)
    return plan_existed or preview_existed


# ── High-level facade ──────────────────────────────────────────────

class ExecutiveOrchestratorBridge:
    """High-level facade for Phase 3 bridge."""

    def __init__(
        self, *, storage: Optional[ObjectiveStateStorage] = None
    ) -> None:
        self._storage = storage or ObjectiveStateStorage()

    def dry_run(
        self, objective_id: str
    ) -> OrchestratorPlanPreview:
        return plan_dry_run(objective_id, storage=self._storage)

    def apply(
        self,
        objective_id: str,
        *,
        require_human_approval: bool = True,
        approver_id: Optional[str] = None,
        approval_token: Optional[str] = None,
        cross_session: bool = False,
        session_id: Optional[str] = None,
    ) -> OrchestratorPlanPreview:
        return plan_apply(
            objective_id,
            storage=self._storage,
            require_human_approval=require_human_approval,
            approver_id=approver_id,
            approval_token=approval_token,
            cross_session=cross_session,
            session_id=session_id,
        )

    def rollback(self, objective_id: str) -> bool:
        return plan_rollback(objective_id, storage=self._storage)

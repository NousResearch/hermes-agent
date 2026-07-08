"""Phase 4A Policy / Approval Gates — engine facade.

Phase 4A centralizes the policy/approval layer for Executive v2:

* ``classify_risk_level(...)`` — pure function: assigns a
  ``RiskLevel`` (R0-R6) to a Phase 3 plan.
* ``build_policy_decision(...)`` — pure function: builds a
  ``PolicyDecision`` (allowed/forbidden actions, warnings,
  fingerprints).
* ``policy_dry_run(...)`` — pure orchestrator: returns a
  ``PolicyDecision`` and an ``ApprovalGateResult`` without touching
  storage.
* ``policy_persist(...)`` — runs 8-layer approval and writes
  ``state_meta[objective_policy_decision:<oid>]`` and
  ``state_meta[objective_approval_request:<oid>]``.
* ``policy_rollback(...)`` — best-effort, idempotent cleanup of
  both new keys.

The 8-layer gate evaluation lives in
``agent.executive.approval_gates``. This module is the **storage
facade** that wires together risk classification, policy building,
gate evaluation, and state_meta persistence.

No Kanban, no Orchestrator execution, no workers, no providers,
no network.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from .approval_gates import (
    ApprovalGateEvaluator,
    ApprovalGateResult,
    evaluate_approval_gates,
)
from .goalmanager_bridge import BridgeApprovalError, BridgeMappingError
from .risk import classify_risk
from .state_storage import ObjectiveStateStorage
from .types import (
    ALL_ACTIONS,
    ACTION_ASSIGN_KANBAN_TASK,
    ACTION_CREATE_KANBAN_TASK,
    ACTION_EXECUTE_BACKGROUND_PROCESS,
    ACTION_EXTERNAL_API_CALL,
    ACTION_NETWORK_CALL,
    ACTION_PROVIDER_API_CALL,
    ACTION_READ_APPROVAL_REQUEST,
    ACTION_READ_ORCHESTRATOR_PREVIEW,
    ACTION_READ_POLICY_DECISION,
    ACTION_READ_STATE_META,
    ACTION_SPAWN_WORKER,
    ACTION_WRITE_APPROVAL_REQUEST,
    ACTION_WRITE_KANBAN_METADATA,
    ACTION_WRITE_OBJECTIVE_LINK,
    ACTION_WRITE_OBJECTIVE_PLAN,
    ACTION_WRITE_OBJECTIVE_PREVIEW,
    ACTION_WRITE_POLICY_DECISION,
    ACTION_WRITE_REPORTS_DIR,
    ACTION_WRITE_STATE_META,
    ACTION_WRITE_TEMP_DIR,
    ApprovalRequest,
    GoalLinkage,
    ObjectivePlan,
    OrchestratorPlanPreview,
    PolicyDecision,
    RiskLevel,
    compute_decision_fingerprint,
    new_uuid,
    now_iso8601,
)


# ──────────────────────────────────────────────────────────────────────
# Allowed actions per risk level (matrix)
# ──────────────────────────────────────────────────────────────────────

_R0_ACTIONS: tuple[str, ...] = (
    ACTION_READ_STATE_META,
    ACTION_READ_ORCHESTRATOR_PREVIEW,
    ACTION_READ_POLICY_DECISION,
    ACTION_READ_APPROVAL_REQUEST,
)
_R1_ACTIONS: tuple[str, ...] = _R0_ACTIONS + (ACTION_WRITE_REPORTS_DIR,)
_R2_ACTIONS: tuple[str, ...] = _R1_ACTIONS + (ACTION_WRITE_TEMP_DIR,)
_R3_ACTIONS: tuple[str, ...] = _R2_ACTIONS + (
    ACTION_WRITE_STATE_META,
    ACTION_WRITE_OBJECTIVE_LINK,
    ACTION_WRITE_OBJECTIVE_PLAN,
    ACTION_WRITE_OBJECTIVE_PREVIEW,
    ACTION_WRITE_POLICY_DECISION,
    ACTION_WRITE_APPROVAL_REQUEST,
)
_R4_ACTIONS: tuple[str, ...] = _R3_ACTIONS + (
    ACTION_WRITE_KANBAN_METADATA,
    ACTION_CREATE_KANBAN_TASK,
    ACTION_ASSIGN_KANBAN_TASK,
)
_R5_ACTIONS: tuple[str, ...] = _R4_ACTIONS + (
    ACTION_SPAWN_WORKER,
    ACTION_EXECUTE_BACKGROUND_PROCESS,
)
_R6_ACTIONS: tuple[str, ...] = _R5_ACTIONS + (
    ACTION_NETWORK_CALL,
    ACTION_PROVIDER_API_CALL,
    ACTION_EXTERNAL_API_CALL,
)

_ALLOWED_BY_LEVEL: dict[RiskLevel, tuple[str, ...]] = {
    RiskLevel.R0: _R0_ACTIONS,
    RiskLevel.R1: _R1_ACTIONS,
    RiskLevel.R2: _R2_ACTIONS,
    RiskLevel.R3: _R3_ACTIONS,
    RiskLevel.R4: _R4_ACTIONS,
    RiskLevel.R5: _R5_ACTIONS,
    RiskLevel.R6: _R6_ACTIONS,
}


def classify_risk_level(
    execution_contract: Optional[Mapping[str, Any]] = None,
    goal_linkage: Optional[GoalLinkage] = None,
    objective_plan: Optional[ObjectivePlan] = None,
    orchestrator_preview: Optional[OrchestratorPlanPreview] = None,
) -> RiskLevel:
    """Canonical alias for ``classify_risk`` (spec naming)."""
    return classify_risk(
        execution_contract=execution_contract,
        goal_linkage=goal_linkage,
        objective_plan=objective_plan,
        orchestrator_preview=orchestrator_preview,
    )


def allowed_actions_for(risk_level: RiskLevel) -> tuple[str, ...]:
    """Return the canonical allowed-actions tuple for a RiskLevel."""
    return _ALLOWED_BY_LEVEL.get(risk_level, _R0_ACTIONS)


def forbidden_actions_for(risk_level: RiskLevel) -> tuple[str, ...]:
    """Return the canonical forbidden-actions tuple for a RiskLevel."""
    allowed = set(allowed_actions_for(risk_level))
    return tuple(a for a in ALL_ACTIONS if a not in allowed)


# ──────────────────────────────────────────────────────────────────────
# Pure: build_policy_decision
# ──────────────────────────────────────────────────────────────────────

def build_policy_decision(
    objective_id: str,
    execution_contract: Optional[Mapping[str, Any]] = None,
    goal_linkage: Optional[GoalLinkage] = None,
    objective_plan: Optional[ObjectivePlan] = None,
    orchestrator_preview: Optional[OrchestratorPlanPreview] = None,
) -> PolicyDecision:
    """Build a PolicyDecision (pure).

    Loads nothing from storage; consumes caller-provided dataclasses
    and produces a deterministic ``PolicyDecision``.

    Raises ``BridgeMappingError`` when required Phase 2/3 artifacts
    are missing.
    """
    if not goal_linkage:
        raise BridgeMappingError(
            f"policy build: goal_linkage is required (objective_id={objective_id})"
        )
    if objective_plan is None:
        raise BridgeMappingError(
            f"policy build: objective_plan is required (objective_id={objective_id})"
        )

    contract: dict = dict(execution_contract or {})
    risk_score = float(contract.get("risk_score", 0.0) or 0.0)
    risk_components = dict(contract.get("risk_components", {}) or {})
    approval_requirements_raw = contract.get("approval_requirements", []) or []
    approval_requirements = tuple(
        dict(a) if isinstance(a, Mapping) else {"gate": str(a)}
        for a in approval_requirements_raw
    )

    risk_level = classify_risk_level(
        execution_contract=contract,
        goal_linkage=goal_linkage,
        objective_plan=objective_plan,
        orchestrator_preview=orchestrator_preview,
    )

    allowed = allowed_actions_for(risk_level)
    forbidden = forbidden_actions_for(risk_level)
    approval_required = int(risk_level) >= int(RiskLevel.R3)

    warnings: list[str] = []
    if any(
        "STRATEGIC" in str(ar.get("gate", "")).upper()
        for ar in approval_requirements
    ):
        warnings.append(
            "STRATEGIC intent: requires explicit STRATEGIC approval."
        )
    if risk_score >= 0.7:
        warnings.append(
            f"HIGH_RISK: risk_score={risk_score:.3f} >= 0.7; requires explicit HIGH_RISK approval."
        )
    if any(
        getattr(sg, "approval_required", False) for sg in objective_plan.subgoals
    ):
        for sg in objective_plan.subgoals:
            if getattr(sg, "approval_required", False):
                warnings.append(
                    f"Subgoal {sg.id!r} requires approval (per PlannerSubgoal.approval_required)."
                )
                break
    if (
        goal_linkage is not None
        and getattr(goal_linkage, "session_id", None)
        and orchestrator_preview is not None
    ):
        warnings.append(
            "Cross-session: requires cross_session=True if session_id differs from linkage."
        )

    fingerprint = compute_decision_fingerprint(
        objective_id=objective_id,
        risk_level=risk_level,
        allowed_actions=allowed,
        forbidden_actions=forbidden,
        approval_required=approval_required,
        risk_score=risk_score,
        risk_components=risk_components,
    )

    return PolicyDecision(
        objective_id=objective_id,
        risk_level=risk_level,
        allowed_actions=allowed,
        forbidden_actions=forbidden,
        approval_required=approval_required,
        warnings=tuple(warnings),
        approval_requirements=approval_requirements,
        risk_score=risk_score,
        risk_components=risk_components,
        created_at=now_iso8601(),
        decision_fingerprint=fingerprint,
    )


# ──────────────────────────────────────────────────────────────────────
# Storage helpers
# ──────────────────────────────────────────────────────────────────────

def _serialize(decision: PolicyDecision) -> str:
    return json.dumps(decision.to_dict(), default=str, sort_keys=True)


def _serialize_request(request: ApprovalRequest) -> str:
    return json.dumps(request.to_dict(), default=str, sort_keys=True)


# ──────────────────────────────────────────────────────────────────────
# 3-mode facade: policy_dry_run / policy_persist / policy_rollback
# ──────────────────────────────────────────────────────────────────────

class ExecutivePolicyEngine:
    """High-level facade for Phase 4A. Three modes:

    * ``dry_run(objective_id)`` — pure compute; no state_meta writes.
    * ``persist(objective_id, **kwargs)`` — runs 8-layer approval,
      writes two new keys.
    * ``rollback(objective_id)`` — best-effort, idempotent cleanup of
      both new keys.

    No Kanban, no Orchestrator, no workers, no providers, no
    network, no subprocess.
    """

    SCHEMA_VERSION = "phase4a.v1"

    def __init__(
        self,
        *,
        storage: Optional[ObjectiveStateStorage] = None,
    ) -> None:
        self._storage = storage or ObjectiveStateStorage()

    # ── load helpers ─────────────────────────────────────────────

    def _load_goal_linkage(
        self, objective_id: str
    ) -> Optional[GoalLinkage]:
        return self._storage.get_objective_goal_link(objective_id)

    def _load_objective_plan(
        self, objective_id: str
    ) -> Optional[ObjectivePlan]:
        return self._storage.get_objective_plan(objective_id)

    def _load_orchestrator_preview(
        self, objective_id: str
    ) -> Optional[OrchestratorPlanPreview]:
        return self._storage.get_objective_orchestrator_preview(objective_id)

    def _load_execution_contract(
        self, objective_id: str
    ) -> dict:
        state = self._storage.load(objective_id)
        if state is None:
            return {}
        contract = state.contract or {}
        return contract if isinstance(contract, dict) else {}

    # ── mode 1: dry_run ──────────────────────────────────────────

    def dry_run(self, objective_id: str) -> PolicyDecision:
        """Compute PolicyDecision. NO state_meta writes. NO side effects."""
        return policy_dry_run(
            objective_id,
            storage=self._storage,
        )

    # ── mode 2: persist ──────────────────────────────────────────

    def persist(
        self,
        objective_id: str,
        *,
        approver_id: Optional[str] = None,
        approval_token: Optional[str] = None,
        kanban_approver_id: Optional[str] = None,
        worker_approver_id: Optional[str] = None,
        external_approver_id: Optional[str] = None,
        cross_session: bool = False,
        session_id: Optional[str] = None,
        expiry: Optional[str] = None,
        renewal: bool = False,
        approval_reason: str = "",
        scope: Optional[tuple[str, ...]] = None,
    ) -> tuple[PolicyDecision, ApprovalRequest]:
        """Compute, run 8-layer approval, and persist both keys."""
        return policy_persist(
            objective_id,
            storage=self._storage,
            approver_id=approver_id,
            approval_token=approval_token,
            kanban_approver_id=kanban_approver_id,
            worker_approver_id=worker_approver_id,
            external_approver_id=external_approver_id,
            cross_session=cross_session,
            session_id=session_id,
            expiry=expiry,
            renewal=renewal,
            approval_reason=approval_reason,
            scope=scope,
        )

    # ── mode 3: rollback ─────────────────────────────────────────

    def rollback(self, objective_id: str) -> bool:
        """Delete both new keys. Idempotent. Best-effort."""
        return policy_rollback(objective_id, storage=self._storage)


# ──────────────────────────────────────────────────────────────────────
# Module-level functions (the spec's three primary entry points)
# ──────────────────────────────────────────────────────────────────────

def policy_dry_run(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
) -> PolicyDecision:
    """Pure: compute PolicyDecision. NO state_meta writes.

    Loads Phase 1+2+3 artifacts from storage when ``storage`` is
    provided. When ``storage`` is ``None``, uses an auto-instantiated
    ``ObjectiveStateStorage``.
    """
    store = storage or ObjectiveStateStorage()

    goal_linkage = store.get_objective_goal_link(objective_id)
    if goal_linkage is None:
        raise BridgeMappingError(
            f"policy_dry_run: goal_linkage missing (objective_id={objective_id})"
        )
    objective_plan = store.get_objective_plan(objective_id)
    if objective_plan is None:
        raise BridgeMappingError(
            f"policy_dry_run: objective_plan missing (objective_id={objective_id})"
        )
    orchestrator_preview = store.get_objective_orchestrator_preview(objective_id)

    state = store.load(objective_id)
    execution_contract: dict = {}
    if state is not None and isinstance(state.contract, dict):
        execution_contract = state.contract

    return build_policy_decision(
        objective_id=objective_id,
        execution_contract=execution_contract,
        goal_linkage=goal_linkage,
        objective_plan=objective_plan,
        orchestrator_preview=orchestrator_preview,
    )


def policy_persist(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    approver_id: Optional[str] = None,
    approval_token: Optional[str] = None,
    kanban_approver_id: Optional[str] = None,
    worker_approver_id: Optional[str] = None,
    external_approver_id: Optional[str] = None,
    cross_session: bool = False,
    session_id: Optional[str] = None,
    expiry: Optional[str] = None,
    renewal: bool = False,
    approval_reason: str = "",
    scope: Optional[tuple[str, ...]] = None,
) -> tuple[PolicyDecision, ApprovalRequest]:
    """Compute, run 8-layer approval, and persist both keys.

    Side effects (only after all 8 gates pass):

    1. ``state_meta[objective_policy_decision:<oid>]``.
    2. ``state_meta[objective_approval_request:<oid>]``.

    Raises ``BridgeApprovalError`` on the first failing gate.
    Raises ``BridgeMappingError`` if Phase 2/3 inputs are missing.
    """
    decision = policy_dry_run(objective_id, storage=storage)

    gate_result: ApprovalGateResult = evaluate_approval_gates(
        decision,
        approver_id=approver_id,
        approval_token=approval_token,
        kanban_approver_id=kanban_approver_id,
        worker_approver_id=worker_approver_id,
        external_approver_id=external_approver_id,
        cross_session=cross_session,
        session_id=session_id,
        expiry=expiry,
        renewal=renewal,
        approval_reason=approval_reason,
        scope=scope,
    )
    if not gate_result.approved or gate_result.approval_request is None:
        raise BridgeApprovalError(
            f"approval not granted (failure_layer={gate_result.failure_layer}, "
            f"reason={gate_result.failure_reason})"
        )
    request = gate_result.approval_request

    store = storage or ObjectiveStateStorage()
    store.set_objective_policy_decision(decision)
    store.set_objective_approval_request(request)
    return decision, request


def policy_rollback(
    objective_id: str,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
) -> bool:
    """Delete both new keys. Idempotent. Best-effort.

    Returns ``True`` if at least one row existed before the call;
    ``False`` otherwise.
    """
    store = storage or ObjectiveStateStorage()
    decision_existed = store.delete_objective_policy_decision(objective_id)
    request_existed = store.delete_objective_approval_request(objective_id)
    return bool(decision_existed or request_existed)


# ──────────────────────────────────────────────────────────────────────
# Read helpers
# ──────────────────────────────────────────────────────────────────────

def get_decision_for_objective(
    objective_id: str,
    storage: Optional[ObjectiveStateStorage] = None,
) -> Optional[PolicyDecision]:
    store = storage or ObjectiveStateStorage()
    return store.get_objective_policy_decision(objective_id)


def get_approval_request_for_objective(
    objective_id: str,
    storage: Optional[ObjectiveStateStorage] = None,
) -> Optional[ApprovalRequest]:
    store = storage or ObjectiveStateStorage()
    return store.get_objective_approval_request(objective_id)


__all__ = [
    "classify_risk_level",
    "build_policy_decision",
    "allowed_actions_for",
    "forbidden_actions_for",
    "policy_dry_run",
    "policy_persist",
    "policy_rollback",
    "ExecutivePolicyEngine",
    "get_decision_for_objective",
    "get_approval_request_for_objective",
]
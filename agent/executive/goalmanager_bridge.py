"""Executive v2 Phase 2 — GoalManager Bridge.

Translates ``ExecutionContract.v1`` (Phase 1 output) into a
``GoalContract`` and applies it to an existing ``GoalManager``
(``hermes_cli/goals.py``). Phase 2 is the bridge between cross-session
Phase 1 objectives and per-session GoalManager goals.

Key constraints:

- Does NOT modify ``hermes_cli/goals.py``.
- Does NOT call ``run_kanban_goal_loop`` or ``evaluate_after_turn``.
- Does NOT import Orchestrator, Planner, Scheduler, GBrain, Obsidian,
  NotebookLM, Kanban, Gateway, Workers, or any LLM provider.
- Default-off: ``bridge_apply`` requires explicit human approval.

Public surface:

- ``map_contract_to_goal`` — pure function: contract → (goal_text, GoalContract, max_turns, fingerprint, warnings).
- ``bridge_dry_run`` — pure: returns a ``BridgePreview``. No side effects.
- ``bridge_apply`` — creates a goal + writes a link. Requires approval.
- ``bridge_rollback`` — best-effort cleanup. Idempotent.
- ``ExecutiveGoalBridge`` — high-level facade that wires the above.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from typing import Any, Optional

from .state_storage import ObjectiveStateStorage
from .types import (
    BridgePreview,
    GoalLinkage,
    now_iso8601,
    objective_goal_link_key,
)
from hermes_cli.goals import (
    DEFAULT_MAX_TURNS,
    GoalContract,
    GoalManager,
)

logger = logging.getLogger(__name__)

# Approval gate thresholds.
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4


# ── Errors ──────────────────────────────────────────────────────────

class BridgeError(RuntimeError):
    """Base error for Phase 2 bridge."""


class BridgeApprovalError(BridgeError):
    """Raised when an approval gate blocks a bridge_apply."""


class BridgeLinkageConflictError(BridgeError):
    """Raised when an existing link is for a different session."""


class BridgeMappingError(BridgeError):
    """Raised when contract → goal mapping fails irrecoverably."""


# ── Mapping ────────────────────────────────────────────────────────

def _derive_goal_text(success_criteria: tuple[str, ...]) -> str:
    """Derive goal_text from success_criteria.

    Take the first 3 criteria, join with "; ", prefix with "OBJECTIVE:".
    Empty criteria → degenerate "(no success_criteria)" + warning.
    """
    criteria = [c.strip() for c in success_criteria if c and c.strip()]
    if not criteria:
        return "(no success_criteria)"
    primary = "; ".join(criteria[:3])
    return f"OBJECTIVE: {primary}"


def _derive_constraints_text(approval_requirements: tuple[dict, ...]) -> str:
    """Map approval_requirements → GoalContract.constraints prose."""
    gates = [
        f"Gate: {ar.get('gate', '?')} "
        f"(approver={ar.get('approver', '?')}, "
        f"ttl={ar.get('ttl_hours', '?')}h)"
        for ar in approval_requirements
    ]
    return "\n".join(gates) if gates else ""


def _derive_boundaries_text(
    hard: tuple[str, ...], soft: tuple[str, ...]
) -> str:
    """Map hard+soft constraints → GoalContract.boundaries prose."""
    parts: list[str] = []
    if hard:
        parts.append("HARD:")
        parts.extend(f"  - {c}" for c in hard)
    if soft:
        parts.append("SOFT:")
        parts.extend(f"  - {c}" for c in soft)
    return "\n".join(parts) if parts else ""


def _derive_verification_text(
    method: str,
    judge_model: Optional[str],
    timeout: int,
    evidence_required: bool,
) -> str:
    """Map verification_method + judge_model + evidence_required → verification prose."""
    parts: list[str] = []
    if evidence_required:
        parts.append("Evidence REQUIRED.")
    parts.append(f"Verification method: {method}.")
    if judge_model:
        parts.append(f"Judge model: {judge_model}.")
    parts.append(f"Verification timeout: {timeout} min.")
    return " ".join(parts)


def _derive_stop_when_text(success_criteria: tuple[str, ...]) -> str:
    """Map success_criteria → GoalContract.stop_when prose."""
    criteria = [c.strip() for c in success_criteria if c and c.strip()]
    if not criteria:
        return ""
    return "STOP WHEN ALL OF: " + "; ".join(criteria)


def _derive_outcome_text(objective_id: str, fingerprint: str) -> str:
    """Synthesize GoalContract.outcome from objective_id + fingerprint."""
    return (
        f"OUTCOME: complete objective {objective_id} "
        f"(fingerprint {fingerprint[:12]})"
    )


def _derive_max_turns(
    budget_max_iterations: Any,
    risk_score: float,
    default_max: int,
) -> int:
    """Derive max_turns from budget + risk adjustment."""
    base = budget_max_iterations if isinstance(budget_max_iterations, int) and budget_max_iterations > 0 else default_max
    if risk_score >= HIGH_RISK_THRESHOLD:
        return max(1, int(base * 0.5))
    if risk_score >= MEDIUM_RISK_THRESHOLD:
        return max(1, int(base * 0.75))
    return int(base)


def _compute_bridge_fingerprint(
    goal_text: str,
    goal_contract: GoalContract,
    max_turns: int,
    objective_id: str,
) -> str:
    """Stable sha256 of canonical bridge inputs."""
    canonical = json.dumps(
        {
            "goal_text": goal_text,
            "contract": goal_contract.to_dict() if hasattr(goal_contract, "to_dict") else asdict(goal_contract),
            "max_turns": max_turns,
            "objective_id": objective_id,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _derive_warnings(
    risk_score: float,
    approval_requirements: tuple[dict, ...],
    success_criteria: tuple[str, ...],
    budget_max_iterations: Any,
) -> tuple[str, ...]:
    """Compute human-readable warnings."""
    out: list[str] = []
    if risk_score >= HIGH_RISK_THRESHOLD:
        out.append("HIGH_RISK: requires EXPLICIT_HIGH_RISK approval before apply.")
    if any("STRATEGIC" in str(ar.get("gate", "")) for ar in approval_requirements):
        out.append("STRATEGIC: requires EXPLICIT_STRATEGIC approval before apply.")
    if not success_criteria:
        out.append("EMPTY success_criteria: dry-run output is degenerate.")
    if not isinstance(budget_max_iterations, int) or budget_max_iterations <= 0:
        out.append("INVALID max_iterations: bridge will use default.")
    return tuple(out)


def map_contract_to_goal(contract: dict) -> tuple[str, GoalContract, int, str, tuple[str, ...]]:
    """Pure mapping: ExecutionContract.v1 dict → (goal_text, GoalContract, max_turns, fingerprint, warnings).

    The contract is a dict (not a dataclass) so this function is
    decoupled from Phase 1's types module. The caller is responsible
    for converting the stored ``ObjectiveStateData.contract`` dict
    into the input format expected here.

    Returns a 5-tuple: (goal_text, GoalContract, max_turns, fingerprint, warnings).
    """
    success_criteria: tuple[str, ...] = tuple(contract.get("success_criteria") or ())
    approval_requirements: tuple[dict, ...] = tuple(contract.get("approval_requirements") or ())
    hard: tuple[str, ...] = tuple(contract.get("hard_constraints") or ())
    soft: tuple[str, ...] = tuple(contract.get("soft_constraints") or ())
    risk_score = float(contract.get("risk_score", 0.0) or 0.0)
    budget: dict = contract.get("budget", {}) or {}
    budget_max = budget.get("max_iterations")
    verification_method = str(contract.get("verification_method", "judge"))
    judge_model = contract.get("judge_model")
    timeout = int(contract.get("verification_timeout_minutes", 60) or 60)
    evidence_required = bool(contract.get("evidence_required", True))
    objective_id = str(contract.get("objective_id", ""))
    fingerprint_seed = str(contract.get("fingerprint", ""))

    goal_text = _derive_goal_text(success_criteria)
    constraints_text = _derive_constraints_text(approval_requirements)
    boundaries_text = _derive_boundaries_text(hard, soft)
    verification_text = _derive_verification_text(
        verification_method, judge_model, timeout, evidence_required
    )
    stop_when_text = _derive_stop_when_text(success_criteria)
    outcome_text = _derive_outcome_text(objective_id, fingerprint_seed)
    max_turns = _derive_max_turns(budget_max, risk_score, DEFAULT_MAX_TURNS)
    warnings = _derive_warnings(
        risk_score, approval_requirements, success_criteria, budget_max
    )

    goal_contract = GoalContract(
        outcome=outcome_text,
        verification=verification_text,
        constraints=constraints_text,
        boundaries=boundaries_text,
        stop_when=stop_when_text,
    )
    bridge_fp = _compute_bridge_fingerprint(
        goal_text, goal_contract, max_turns, objective_id
    )
    return goal_text, goal_contract, max_turns, bridge_fp, warnings


# ── BridgePreview construction ─────────────────────────────────────

def _build_preview(
    objective_id: str,
    session_id: str,
    contract: dict,
    warnings: tuple[str, ...],
    existing_link: Optional[GoalLinkage],
) -> BridgePreview:
    """Build a BridgePreview from a contract + warnings + existing link."""
    goal_text, goal_contract, max_turns, bridge_fp, _ = map_contract_to_goal(contract)
    risk_score = float(contract.get("risk_score", 0.0) or 0.0)
    approval_requirements: tuple[dict, ...] = tuple(
        contract.get("approval_requirements") or ()
    )
    would_apply_to_existing_goal = existing_link is not None
    cross_session_conflict = (
        existing_link is not None and existing_link.session_id != session_id
    )
    return BridgePreview(
        objective_id=objective_id,
        session_id=session_id,
        goal_text=goal_text,
        goal_contract_outcome=goal_contract.outcome,
        goal_contract_verification=goal_contract.verification,
        goal_contract_constraints=goal_contract.constraints,
        goal_contract_boundaries=goal_contract.boundaries,
        goal_contract_stop_when=goal_contract.stop_when,
        max_turns=max_turns,
        bridge_fingerprint=bridge_fp,
        risk_score=risk_score,
        approval_requirements=approval_requirements,
        warnings=warnings,
        would_apply_to_existing_goal=would_apply_to_existing_goal,
        cross_session_conflict=cross_session_conflict,
    )


# ── Public functions ───────────────────────────────────────────────

def bridge_dry_run(
    objective_id: str,
    goal_manager: GoalManager,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
) -> BridgePreview:
    """Compute a BridgePreview for the given objective.

    **Zero side effects**: no state_meta writes, no GoalManager mutation.

    Raises ``BridgeMappingError`` if the objective is not in state_meta,
    or if the stored state is malformed.
    """
    if not objective_id:
        raise ValueError("objective_id is empty")
    if goal_manager is None:
        raise ValueError("goal_manager is None")

    storage = storage or ObjectiveStateStorage()
    state = storage.load(objective_id)
    if state is None:
        raise BridgeMappingError(
            f"objective {objective_id} not found in state_meta"
        )
    if not state.contract:
        raise BridgeMappingError(
            f"objective {objective_id} has no contract (state={state.state.value})"
        )

    existing_link = storage.get_objective_goal_link(objective_id)
    _, _, _, _, warnings = map_contract_to_goal(state.contract)
    return _build_preview(
        objective_id=objective_id,
        session_id=goal_manager.session_id,
        contract=state.contract,
        warnings=warnings,
        existing_link=existing_link,
    )


def _check_approval_gates(
    preview: BridgePreview,
    *,
    require_human_approval: bool,
    approver_id: Optional[str],
    approval_token: Optional[str],
) -> None:
    """Validate the 4 approval layers. Raise BridgeApprovalError on fail."""
    # Layer 1: default
    if require_human_approval and not approver_id:
        raise BridgeApprovalError(
            "Layer 1: approver_id is required when require_human_approval=True"
        )
    # Layer 2: STRATEGIC
    if any(
        "STRATEGIC" in str(ar.get("gate", ""))
        for ar in preview.approval_requirements
    ):
        if not require_human_approval or not approver_id:
            raise BridgeApprovalError(
                "Layer 2: STRATEGIC gate requires require_human_approval=True AND approver_id"
            )
    # Layer 3: HIGH_RISK
    if preview.risk_score >= HIGH_RISK_THRESHOLD:
        if not require_human_approval or not approver_id or not approval_token:
            raise BridgeApprovalError(
                "Layer 3: HIGH_RISK gate requires require_human_approval=True, "
                "approver_id, AND approval_token"
            )


def bridge_apply(
    objective_id: str,
    goal_manager: GoalManager,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
    require_human_approval: bool = True,
    approver_id: Optional[str] = None,
    approval_token: Optional[str] = None,
    cross_session: bool = False,
) -> GoalLinkage:
    """Apply a Phase 2 bridge.

    Side effects (in order):
    1. Load objective from state_meta.
    2. Validate approval gates.
    3. Call ``goal_manager.set(goal_text, max_turns=, contract=)``.
    4. Save ``GoalLinkage`` to ``state_meta[objective_goal_link:<oid>]``.

    Raises:
    - ``BridgeMappingError``: if objective not found or contract missing.
    - ``BridgeLinkageConflictError``: if existing link for different session.
    - ``BridgeApprovalError``: if any approval gate fails.
    - ``ValueError``: on invalid input.
    """
    if not objective_id:
        raise ValueError("objective_id is empty")
    if goal_manager is None:
        raise ValueError("goal_manager is None")

    storage = storage or ObjectiveStateStorage()
    state = storage.load(objective_id)
    if state is None:
        raise BridgeMappingError(
            f"objective {objective_id} not found in state_meta"
        )
    if not state.contract:
        raise BridgeMappingError(
            f"objective {objective_id} has no contract"
        )

    # Check existing link BEFORE any mutation.
    existing_link = storage.get_objective_goal_link(objective_id)
    if (
        existing_link is not None
        and existing_link.session_id != goal_manager.session_id
        and not cross_session
    ):
        raise BridgeLinkageConflictError(
            f"objective {objective_id} is already linked to session "
            f"{existing_link.session_id}; new session "
            f"{goal_manager.session_id} requires cross_session=True"
        )

    # Compute preview and check gates.
    _, _, _, _, warnings = map_contract_to_goal(state.contract)
    preview = _build_preview(
        objective_id=objective_id,
        session_id=goal_manager.session_id,
        contract=state.contract,
        warnings=warnings,
        existing_link=existing_link,
    )
    _check_approval_gates(
        preview,
        require_human_approval=require_human_approval,
        approver_id=approver_id,
        approval_token=approval_token,
    )

    # Apply to GoalManager. Note: we use ONLY .set() per the design.
    # The contract is passed via the contract= parameter of .set().
    try:
        goal_manager.set(
            preview.goal_text,
            max_turns=preview.max_turns,
            contract=GoalContract(
                outcome=preview.goal_contract_outcome,
                verification=preview.goal_contract_verification,
                constraints=preview.goal_contract_constraints,
                boundaries=preview.goal_contract_boundaries,
                stop_when=preview.goal_contract_stop_when,
            ),
        )
    except Exception as exc:
        logger.warning("bridge_apply: gm.set failed: %s", exc)
        raise

    # Save the link.
    link = GoalLinkage(
        objective_id=objective_id,
        session_id=goal_manager.session_id,
        goal_text=preview.goal_text,
        bridge_applied_at=now_iso8601(),
        bridge_fingerprint=preview.bridge_fingerprint,
        bridge_applied_by=approver_id or "unknown",
        bridge_version="phase2.v1",
        bridge_objective_fingerprint=state.fingerprint or "",
    )
    storage.set_objective_goal_link(link)
    logger.info(
        "bridge_apply: objective %s -> session %s (fingerprint %s)",
        objective_id,
        goal_manager.session_id,
        preview.bridge_fingerprint[:12],
    )
    return link


def bridge_rollback(
    objective_id: str,
    goal_manager: GoalManager,
    *,
    storage: Optional[ObjectiveStateStorage] = None,
) -> bool:
    """Rollback a Phase 2 bridge apply.

    Best-effort cleanup:
    1. If the link's session_id matches goal_manager.session_id AND the
       current goal's text matches the link's goal_text, call
       ``goal_manager.clear()`` (mark cleared, audit preserved).
    2. Delete ``state_meta[objective_goal_link:<oid>]``.

    Idempotent: returns False if no link exists, True if cleanup happened.
    """
    if not objective_id:
        raise ValueError("objective_id is empty")
    if goal_manager is None:
        raise ValueError("goal_manager is None")

    storage = storage or ObjectiveStateStorage()
    link = storage.get_objective_goal_link(objective_id)
    if link is None:
        return False

    cleared_goal = False
    if (
        link.session_id == goal_manager.session_id
        and goal_manager.state is not None
        and goal_manager.state.goal == link.goal_text
    ):
        try:
            goal_manager.clear()
            cleared_goal = True
        except Exception as exc:
            logger.warning("bridge_rollback: gm.clear failed: %s", exc)

    deleted = storage.delete_objective_goal_link(objective_id)
    logger.info(
        "bridge_rollback: objective %s cleared_goal=%s deleted_link=%s",
        objective_id, cleared_goal, deleted,
    )
    return True


# ── High-level facade ──────────────────────────────────────────────

class ExecutiveGoalBridge:
    """High-level facade for the Phase 2 bridge.

    Holds a ``ObjectiveStateStorage`` instance. The caller provides a
    fresh ``GoalManager`` per session.
    """

    def __init__(self, *, storage: Optional[ObjectiveStateStorage] = None) -> None:
        self._storage = storage or ObjectiveStateStorage()

    def dry_run(
        self, objective_id: str, goal_manager: GoalManager
    ) -> BridgePreview:
        return bridge_dry_run(
            objective_id, goal_manager, storage=self._storage
        )

    def apply(
        self,
        objective_id: str,
        goal_manager: GoalManager,
        *,
        require_human_approval: bool = True,
        approver_id: Optional[str] = None,
        approval_token: Optional[str] = None,
        cross_session: bool = False,
    ) -> GoalLinkage:
        return bridge_apply(
            objective_id,
            goal_manager,
            storage=self._storage,
            require_human_approval=require_human_approval,
            approver_id=approver_id,
            approval_token=approval_token,
            cross_session=cross_session,
        )

    def rollback(
        self, objective_id: str, goal_manager: GoalManager
    ) -> bool:
        return bridge_rollback(
            objective_id, goal_manager, storage=self._storage
        )

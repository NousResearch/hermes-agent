"""Core dataclasses for Executive v2.

Frozen dataclasses used by the engine, normalizer, classifier,
capability discovery, and contract builder. Immutable: a contract
or a normalized objective is a value object.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────

class GoalClass(str, Enum):
    RESEARCH = "RESEARCH"
    BUILD = "BUILD"
    ANALYZE = "ANALYZE"
    AUTOMATE = "AUTOMATE"
    INTEGRATE = "INTEGRATE"
    OPTIMIZE = "OPTIMIZE"
    DOCUMENT = "DOCUMENT"
    VERIFY = "VERIFY"
    MAINTAIN = "MAINTAIN"
    STRATEGIC = "STRATEGIC"
    OTHER = "OTHER"


class RiskProfile(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Complexity(str, Enum):
    XS = "XS"
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"


class ObjectiveState(str, Enum):
    """Lifecycle states for an objective in Phase 1.

    Phase 1 implements DRAFT, NORMALIZED, CLASSIFIED, DISCOVERED,
    CONTRACT_DRAFT, PERSISTED, FAILED. EXECUTING/VERIFYING/DONE/ABORTED
    are Phase 2+ and require Planner/Orchestrator.
    """
    DRAFT = "DRAFT"
    NORMALIZED = "NORMALIZED"
    CLASSIFIED = "CLASSIFIED"
    DISCOVERED = "DISCOVERED"
    CONTRACT_DRAFT = "CONTRACT_DRAFT"
    PERSISTED = "PERSISTED"
    FAILED = "FAILED"


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def now_iso8601() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def new_uuid() -> str:
    return str(uuid.uuid4())


def compute_fingerprint(
    objective_text: str,
    constraints: list[str] | tuple[str, ...] | None,
    user_id: str,
    created_at: str,
) -> str:
    """Compute the stable fingerprint of the objective.

    fingerprint = sha256(objective_text || sorted_constraints || user_id || created_at)[:64]
    """
    payload = "|".join([
        objective_text.strip(),
        "|".join(sorted(constraints or [])),
        user_id,
        created_at,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:64]


def compute_contract_fingerprint(objective_id: str, fingerprint_seed: str) -> str:
    """Derive the contract fingerprint from the objective."""
    payload = f"{objective_id}|{fingerprint_seed}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:64]


# ──────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NormalizedObjective:
    objective_id: str
    goal_class: GoalClass
    constraints: tuple[str, ...]
    success_criteria: tuple[str, ...]
    human_constraints: tuple[str, ...]
    approval_requirements: tuple[dict, ...]
    risk_profile: RiskProfile
    estimated_complexity: Complexity
    knowledge_requirements: tuple[str, ...]
    execution_requirements: dict
    created_at: str
    created_by: str
    parent_objective_id: str | None = None
    session_id: str | None = None
    fingerprint: str = ""
    schema_version: str = "1.0"


@dataclass(frozen=True)
class ClassifiedObjective:
    goal_class: GoalClass
    risk_profile: RiskProfile
    estimated_complexity: Complexity
    rationale: str
    signal_tokens: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityCandidate:
    kind: str
    id: str
    name: str
    source_path: str
    description: str
    keywords: tuple[str, ...]
    match_score: float
    match_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityDiscovery:
    objective_id: str
    discovered_at: str
    candidates: tuple[CapabilityCandidate, ...]
    reuse_decision: str
    rationale: str
    gaps: tuple[str, ...]
    p0_query_duration_ms: int
    p1_query_duration_ms: int


@dataclass(frozen=True)
class ApprovalRequirement:
    gate: str
    approver: str
    ttl_hours: int = 24


@dataclass(frozen=True)
class RiskComponents:
    financial: float = 0.0
    regulatory: float = 0.0
    customer_facing: float = 0.0
    irreversibility: float = 0.0
    data_sensitivity: float = 0.0

    @property
    def total(self) -> float:
        return min(
            1.0,
            self.financial * 0.3
            + self.regulatory * 0.3
            + self.customer_facing * 0.2
            + self.irreversibility * 0.1
            + self.data_sensitivity * 0.1,
        )


@dataclass(frozen=True)
class BudgetPolicy:
    policy: str
    max_iterations: int
    max_duration_minutes: int
    max_cost_usd: float


@dataclass(frozen=True)
class ExecutionContractV1:
    contract_version: str
    contract_id: str
    objective_id: str
    goal_id: str | None
    fingerprint: str
    required_capabilities: tuple[str, ...]
    required_tools: tuple[str, ...]
    required_skills: tuple[str, ...]
    required_roles: tuple[str, ...]
    required_workflows: tuple[str, ...]
    required_providers: tuple[str, ...]
    knowledge_summary_keys: tuple[str, ...]
    knowledge_summary_text: str
    hard_constraints: tuple[str, ...]
    soft_constraints: tuple[str, ...]
    approval_requirements: tuple[dict, ...]
    risk_components: dict
    risk_score: float
    budget: dict
    execution_strategy: str
    rollback_strategy: str
    planner_inputs_sub_goals: tuple[str, ...]
    planner_inputs_success_criteria: tuple[str, ...]
    planner_inputs_hard_constraints: tuple[str, ...]
    planner_inputs_soft_constraints: tuple[str, ...]
    planner_inputs_preferred_workflow: str | None
    planner_inputs_preferred_role: str | None
    scheduler_hints_priority: str
    scheduler_hints_deadline: str | None
    scheduler_hints_blocking_objectives: tuple[str, ...]
    scheduler_hints_parallelism_allowed: bool
    success_criteria: tuple[str, ...]
    verification_method: str
    verification_timeout_minutes: int
    judge_model: str | None
    evidence_required: bool
    created_at: str
    created_by: str


@dataclass
class ObjectiveStateData:
    """Mutable in-memory snapshot of an objective's state at a point in time."""
    objective_id: str
    state: ObjectiveState
    objective_text: str
    constraints: list[str]
    user_id: str
    created_at: str
    normalized: dict | None = None
    classified: dict | None = None
    discovered: dict | None = None
    contract: dict | None = None
    fingerprint: str | None = None
    last_error: str | None = None
    last_transition_at: str | None = None
    last_transition_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectiveStateData":
        state_value = data.get("state")
        if isinstance(state_value, str):
            try:
                data = dict(data)
                data["state"] = ObjectiveState(state_value)
            except ValueError:
                pass
        return cls(**data)


# ──────────────────────────────────────────────────────────────────────
# State storage keys
# ──────────────────────────────────────────────────────────────────────

def objective_key(objective_id: str) -> str:
    """Build the state_meta key for an active objective.

    Pattern: 'objective:<objective_id>'. Parallel to GoalManager's
    'goal:<session_id>' namespace. No collision.
    """
    return f"objective:{objective_id}"


def objective_archive_key(objective_id: str) -> str:
    """Build the state_meta key for an archived objective."""
    return f"objective_archive:{objective_id}"


# ── Phase 2 GoalManager Bridge types ────────────────────────────────

# Linkage key for cross-system linkage between Phase 1 objective and
# GoalManager goal. Phase 2 writes here; Phase 1 does NOT touch it.
OBJECTIVE_GOAL_LINK_PREFIX = "objective_goal_link:"


def objective_goal_link_key(objective_id: str) -> str:
    """Build the state_meta key for the objective↔goal linkage."""
    return f"{OBJECTIVE_GOAL_LINK_PREFIX}{objective_id}"


# ── Phase 3 Planner / Orchestrator Bridge types ──────────────────

# Storage prefixes for Phase 3 plan and orchestrator preview.
OBJECTIVE_PLAN_PREFIX = "objective_plan:"
OBJECTIVE_ORCHESTRATOR_PREVIEW_PREFIX = "objective_orchestrator_preview:"


def objective_plan_key(objective_id: str) -> str:
    """Build the state_meta key for the Phase 3 plan."""
    return f"{OBJECTIVE_PLAN_PREFIX}{objective_id}"


def objective_orchestrator_preview_key(objective_id: str) -> str:
    """Build the state_meta key for the Phase 3 orchestrator preview."""
    return f"{OBJECTIVE_ORCHESTRATOR_PREVIEW_PREFIX}{objective_id}"


@dataclass(frozen=True)
class BridgePreview:
    """Result of ``bridge_dry_run``. Pure data; no side effects.

    Represents what ``bridge_apply`` would do, computed deterministically
    from the Phase 1 ``ExecutionContract.v1`` and the caller's
    ``GoalManager`` session.
    """

    objective_id: str
    session_id: str
    goal_text: str
    goal_contract_outcome: str
    goal_contract_verification: str
    goal_contract_constraints: str
    goal_contract_boundaries: str
    goal_contract_stop_when: str
    max_turns: int
    bridge_fingerprint: str
    risk_score: float
    approval_requirements: tuple[dict, ...]
    warnings: tuple[str, ...]
    would_apply_to_existing_goal: bool
    cross_session_conflict: bool

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "session_id": self.session_id,
            "goal_text": self.goal_text,
            "goal_contract_outcome": self.goal_contract_outcome,
            "goal_contract_verification": self.goal_contract_verification,
            "goal_contract_constraints": self.goal_contract_constraints,
            "goal_contract_boundaries": self.goal_contract_boundaries,
            "goal_contract_stop_when": self.goal_contract_stop_when,
            "max_turns": self.max_turns,
            "bridge_fingerprint": self.bridge_fingerprint,
            "risk_score": self.risk_score,
            "approval_requirements": list(self.approval_requirements),
            "warnings": list(self.warnings),
            "would_apply_to_existing_goal": self.would_apply_to_existing_goal,
            "cross_session_conflict": self.cross_session_conflict,
        }


@dataclass(frozen=True)
class GoalLinkage:
    """Audit record of a Phase 2 bridge apply.

    Stored in state_meta[objective_goal_link:<objective_id>] as JSON.
    Links a Phase 1 objective_id with the GoalManager session_id and
    the goal_text that was applied.
    """

    objective_id: str
    session_id: str
    goal_text: str
    bridge_applied_at: str
    bridge_fingerprint: str
    bridge_applied_by: str
    bridge_version: str
    bridge_objective_fingerprint: str

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "session_id": self.session_id,
            "goal_text": self.goal_text,
            "bridge_applied_at": self.bridge_applied_at,
            "bridge_fingerprint": self.bridge_fingerprint,
            "bridge_applied_by": self.bridge_applied_by,
            "bridge_version": self.bridge_version,
            "bridge_objective_fingerprint": self.bridge_objective_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoalLinkage":
        return cls(
            objective_id=str(data.get("objective_id", "")),
            session_id=str(data.get("session_id", "")),
            goal_text=str(data.get("goal_text", "")),
            bridge_applied_at=str(data.get("bridge_applied_at", "")),
            bridge_fingerprint=str(data.get("bridge_fingerprint", "")),
            bridge_applied_by=str(data.get("bridge_applied_by", "")),
            bridge_version=str(data.get("bridge_version", "phase2.v1")),
            bridge_objective_fingerprint=str(
                data.get("bridge_objective_fingerprint", "")
            ),
        )


# ── Phase 3 Planner / Orchestrator Bridge dataclasses ────────

@dataclass(frozen=True)
class PlannerSubgoal:
    """A single sub-goal produced by the Phase 3 minimal planner.

    One ``PlannerSubgoal`` per ``success_criterion`` in the parent
    ``ExecutionContract.v1``. Phase 3 produces a linear ordered list
    (no DAG). The ``source_criterion_index`` ties the subgoal back to
    its origin criterion.
    """

    id: str
    title: str
    intent: str  # "RESEARCH" | "BUILD" | "AUTOMATE" | "STRATEGIC" | "OTHER"
    constraints: tuple[str, ...]
    expected_output: str
    risk_level: str  # "low" | "medium" | "high"
    approval_required: bool
    estimated_iterations: int
    timeout_seconds: int
    source_criterion_index: int

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "intent": self.intent,
            "constraints": list(self.constraints),
            "expected_output": self.expected_output,
            "risk_level": self.risk_level,
            "approval_required": self.approval_required,
            "estimated_iterations": self.estimated_iterations,
            "timeout_seconds": self.timeout_seconds,
            "source_criterion_index": self.source_criterion_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlannerSubgoal":
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            intent=str(data.get("intent", "OTHER")),
            constraints=tuple(data.get("constraints") or ()),
            expected_output=str(data.get("expected_output", "")),
            risk_level=str(data.get("risk_level", "low")),
            approval_required=bool(data.get("approval_required", False)),
            estimated_iterations=int(data.get("estimated_iterations", 1) or 1),
            timeout_seconds=int(data.get("timeout_seconds", 60) or 60),
            source_criterion_index=int(
                data.get("source_criterion_index", 0) or 0
            ),
        )


@dataclass(frozen=True)
class ObjectivePlan:
    """A list of subgoals for a given objective.

    Stored in ``state_meta[objective_plan:<oid>]`` as JSON.
    """

    objective_id: str
    subgoals: tuple[PlannerSubgoal, ...]
    plan_fingerprint: str
    created_at: str

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "subgoals": [s.to_dict() for s in self.subgoals],
            "plan_fingerprint": self.plan_fingerprint,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectivePlan":
        subgoals_data = data.get("subgoals") or []
        subgoals = tuple(
            PlannerSubgoal.from_dict(s) for s in subgoals_data
        )
        return cls(
            objective_id=str(data.get("objective_id", "")),
            subgoals=subgoals,
            plan_fingerprint=str(data.get("plan_fingerprint", "")),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(frozen=True)
class OrchestratorPlanPreview:
    """Phase 3 preview: a plan + serialized TaskSpec + validation metadata.

    Phase 3 NEVER creates real Kanban tasks. This preview is the
    output artifact that humans see and approve before any real
    execution (which is Phase 4).
    """

    objective_id: str
    plan: ObjectivePlan
    task_specs: tuple[dict, ...]  # serialized TaskSpec dicts
    warnings: tuple[str, ...]
    requires_approval: bool
    risk_score: float
    preview_fingerprint: str
    created_at: str

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "plan": self.plan.to_dict(),
            "task_specs": list(self.task_specs),
            "warnings": list(self.warnings),
            "requires_approval": self.requires_approval,
            "risk_score": self.risk_score,
            "preview_fingerprint": self.preview_fingerprint,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OrchestratorPlanPreview":
        plan_data = data.get("plan") or {}
        return cls(
            objective_id=str(data.get("objective_id", "")),
            plan=ObjectivePlan.from_dict(plan_data),
            task_specs=tuple(data.get("task_specs") or ()),
            warnings=tuple(data.get("warnings") or ()),
            requires_approval=bool(data.get("requires_approval", False)),
            risk_score=float(data.get("risk_score", 0.0) or 0.0),
            preview_fingerprint=str(data.get("preview_fingerprint", "")),
            created_at=str(data.get("created_at", "")),
        )


# ── Phase 4A Policy / Approval Gates types ──────────────────────

# Phase 4A is the centralized policy/approval layer that consumes
# Phase 1+2+3 artifacts and produces a PolicyDecision and (when
# required) an ApprovalRequest. Two new state_meta namespaces are
# added; no new tables are created.

import json  # noqa: E402  (Phase 4A additions)
from typing import Optional  # noqa: E402  (Phase 4A additions)
from enum import IntEnum  # noqa: E402  (kept near RiskLevel for clarity)


# ── Phase 4B Kanban Apply types ─────────────────────────────

# Phase 4B is the "apply" layer that consumes a Phase 3
# OrchestratorPlanPreview + a Phase 4A ApprovalRequest and produces
# real Kanban tasks via the existing kb.create_task API. Two new
# state_meta keys are added; no new tables are created.

# Storage prefixes for Phase 4B kanban apply and kanban task list.
OBJECTIVE_KANBAN_APPLY_PREFIX = "objective_kanban_apply:"
OBJECTIVE_KANBAN_TASKS_PREFIX = "objective_kanban_tasks:"


def objective_kanban_apply_key(objective_id: str) -> str:
    """Build the state_meta key for a Phase 4B apply record."""
    return f"{OBJECTIVE_KANBAN_APPLY_PREFIX}{objective_id}"


def objective_kanban_tasks_key(objective_id: str) -> str:
    """Build the state_meta key for a Phase 4B task list."""
    return f"{OBJECTIVE_KANBAN_TASKS_PREFIX}{objective_id}"


def compute_kanban_apply_fingerprint(
    objective_id: str,
    task_ids: tuple,
    decision_fingerprint: str,
    request_fingerprint: str,
    kanban_apply_fingerprint: str = "",
) -> str:
    """Stable fingerprint of a KanbanApply's canonical inputs.

    Two applies with the same canonical inputs (objective_id, task_ids,
    decision_fingerprint, request_fingerprint, apply_fingerprint) yield
    the same fingerprint, regardless of ``created_at``.
    """
    canonical = json.dumps(
        {
            "objective_id": objective_id,
            "task_ids": sorted(task_ids),
            "decision_fingerprint": decision_fingerprint,
            "request_fingerprint": request_fingerprint,
            "kanban_apply_fingerprint": kanban_apply_fingerprint,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_kanban_result_fingerprint(
    objective_id: str,
    task_ids: tuple,
    preview_fingerprint: str,
    decision_fingerprint: str,
    request_fingerprint: str,
) -> str:
    """Stable fingerprint of a KanbanApplyResult's canonical inputs."""
    canonical = json.dumps(
        {
            "objective_id": objective_id,
            "task_ids": sorted(task_ids),
            "preview_fingerprint": preview_fingerprint,
            "decision_fingerprint": decision_fingerprint,
            "request_fingerprint": request_fingerprint,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class KanbanApplyPreview:
    """Phase 4B output: a pure preview of an apply before any write.

    ``task_kwargs_list`` is a tuple of kwargs dicts suitable for
    ``kb.create_task(conn, **kwargs)``. The ``parents`` field in each
    dict is an empty tuple (placeholder); the actual parent linkage
    is resolved sequentially at apply time.

    No state_meta writes. No Kanban writes.
    """

    objective_id: str
    task_specs: tuple  # tuple[dict, ...] raw Phase 3 task_specs
    task_kwargs_list: tuple  # tuple[dict, ...] kwargs for kb.create_task
    kanban_apply_fingerprint: str
    warnings: tuple  # tuple[str, ...]
    created_at: str

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "task_specs": list(self.task_specs),
            "task_kwargs_list": list(self.task_kwargs_list),
            "kanban_apply_fingerprint": self.kanban_apply_fingerprint,
            "warnings": list(self.warnings),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KanbanApplyPreview":
        return cls(
            objective_id=str(data.get("objective_id", "")),
            task_specs=tuple(data.get("task_specs") or ()),
            task_kwargs_list=tuple(data.get("task_kwargs_list") or ()),
            kanban_apply_fingerprint=str(data.get("kanban_apply_fingerprint", "")),
            warnings=tuple(data.get("warnings") or ()),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(frozen=True)
class KanbanTaskLink:
    """A single task link: (objective_id, spec_index) -> task_id.

    Stored implicitly in the apply record's ``task_ids`` tuple (ordered
    by spec_index) and used by the rollback plan to delete tasks in
    reverse order. This dataclass is provided for callers that want a
    structured view of one link.
    """

    objective_id: str
    spec_index: int
    task_id: str
    idempotency_key: str

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "spec_index": int(self.spec_index),
            "task_id": self.task_id,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KanbanTaskLink":
        return cls(
            objective_id=str(data.get("objective_id", "")),
            spec_index=int(data.get("spec_index", 0) or 0),
            task_id=str(data.get("task_id", "")),
            idempotency_key=str(data.get("idempotency_key", "")),
        )


@dataclass(frozen=True)
class KanbanApplyResult:
    """Phase 4B output: the result of a successful (or duplicate) apply.

    Stored in ``state_meta[objective_kanban_apply:<oid>]`` as JSON.
    """

    objective_id: str
    task_ids: tuple  # tuple[str, ...]
    preview_fingerprint: str
    decision_fingerprint: str
    request_fingerprint: str
    result_fingerprint: str
    duplicate: bool  # True if returned from a dedup hit
    created_at: str
    created_by: str
    board: object  # Optional[str]; object for forward-compat

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "task_ids": list(self.task_ids),
            "preview_fingerprint": self.preview_fingerprint,
            "decision_fingerprint": self.decision_fingerprint,
            "request_fingerprint": self.request_fingerprint,
            "result_fingerprint": self.result_fingerprint,
            "duplicate": bool(self.duplicate),
            "created_at": self.created_at,
            "created_by": self.created_by,
            "board": self.board,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KanbanApplyResult":
        return cls(
            objective_id=str(data.get("objective_id", "")),
            task_ids=tuple(data.get("task_ids") or ()),
            preview_fingerprint=str(data.get("preview_fingerprint", "")),
            decision_fingerprint=str(data.get("decision_fingerprint", "")),
            request_fingerprint=str(data.get("request_fingerprint", "")),
            result_fingerprint=str(data.get("result_fingerprint", "")),
            duplicate=bool(data.get("duplicate", False)),
            created_at=str(data.get("created_at", "")),
            created_by=str(data.get("created_by", "")),
            board=data.get("board"),
        )


@dataclass(frozen=True)
class KanbanRollbackPlan:
    """Phase 4B output: an idempotent rollback plan for a KanbanApplyResult.

    Stores the list of task_ids to delete (in reverse order so parent
    tasks are removed after children). The ``mode`` field chooses
    hard-delete (kb.delete_task) or soft-archive (kb.archive_task).
    """

    objective_id: str
    task_ids: tuple  # tuple[str, ...] in reverse-creation order
    kanban_apply_fingerprint: str
    mode: str  # "hard_delete" | "soft_archive"

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "task_ids": list(self.task_ids),
            "kanban_apply_fingerprint": self.kanban_apply_fingerprint,
            "mode": str(self.mode),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KanbanRollbackPlan":
        return cls(
            objective_id=str(data.get("objective_id", "")),
            task_ids=tuple(data.get("task_ids") or ()),
            kanban_apply_fingerprint=str(
                data.get("kanban_apply_fingerprint", "")
            ),
            mode=str(data.get("mode", "hard_delete")),
        )

    @classmethod
    def from_apply_result(
        cls,
        result: "KanbanApplyResult",
        *,
        mode: str = "hard_delete",
    ) -> "KanbanRollbackPlan":
        """Build a rollback plan from a KanbanApplyResult.

        task_ids are returned in **reverse** creation order so that
        parent tasks (created later) are removed after their children.
        """
        return cls(
            objective_id=result.objective_id,
            task_ids=tuple(reversed(result.task_ids)),
            kanban_apply_fingerprint=result.preview_fingerprint,
            mode=str(mode),
        )


class RiskLevel(IntEnum):
    """7 risk levels (R0-R6) used by Phase 4A.

    IntEnum allows direct comparison: ``risk >= RiskLevel.R3``.
    """

    R0 = 0  # read-only.
    R1 = 1  # report-dir only.
    R2 = 2  # sandbox (in-memory + temp).
    R3 = 3  # state_meta / local state.
    R4 = 4  # Kanban apply.
    R5 = 5  # runtime / workers.
    R6 = 6  # external / network / provider / API.


# Storage prefixes for Phase 4A policy decision and approval request.
OBJECTIVE_POLICY_DECISION_PREFIX = "objective_policy_decision:"
OBJECTIVE_APPROVAL_REQUEST_PREFIX = "objective_approval_request:"


def objective_policy_decision_key(objective_id: str) -> str:
    """Build the state_meta key for a Phase 4A policy decision."""
    return f"{OBJECTIVE_POLICY_DECISION_PREFIX}{objective_id}"


def objective_approval_request_key(objective_id: str) -> str:
    """Build the state_meta key for a Phase 4A approval request."""
    return f"{OBJECTIVE_APPROVAL_REQUEST_PREFIX}{objective_id}"


# Canonical action catalogue (used by allowed/forbidden matrices).

ACTION_READ_STATE_META = "read_state_meta"
ACTION_READ_ORCHESTRATOR_PREVIEW = "read_orchestrator_preview"
ACTION_READ_POLICY_DECISION = "read_policy_decision"
ACTION_READ_APPROVAL_REQUEST = "read_approval_request"
ACTION_WRITE_REPORTS_DIR = "write_reports_dir"
ACTION_WRITE_TEMP_DIR = "write_temp_dir"
ACTION_WRITE_STATE_META = "write_state_meta"
ACTION_WRITE_OBJECTIVE_LINK = "write_objective_link"
ACTION_WRITE_OBJECTIVE_PLAN = "write_objective_plan"
ACTION_WRITE_OBJECTIVE_PREVIEW = "write_objective_preview"
ACTION_WRITE_POLICY_DECISION = "write_policy_decision"
ACTION_WRITE_APPROVAL_REQUEST = "write_approval_request"
ACTION_WRITE_KANBAN_METADATA = "write_kanban_metadata"
ACTION_CREATE_KANBAN_TASK = "create_kanban_task"
ACTION_ASSIGN_KANBAN_TASK = "assign_kanban_task"
ACTION_SPAWN_WORKER = "spawn_worker"
ACTION_EXECUTE_BACKGROUND_PROCESS = "execute_background_process"
ACTION_NETWORK_CALL = "network_call"
ACTION_PROVIDER_API_CALL = "provider_api_call"
ACTION_EXTERNAL_API_CALL = "external_api_call"

ALL_ACTIONS: tuple[str, ...] = (
    ACTION_READ_STATE_META,
    ACTION_READ_ORCHESTRATOR_PREVIEW,
    ACTION_READ_POLICY_DECISION,
    ACTION_READ_APPROVAL_REQUEST,
    ACTION_WRITE_REPORTS_DIR,
    ACTION_WRITE_TEMP_DIR,
    ACTION_WRITE_STATE_META,
    ACTION_WRITE_OBJECTIVE_LINK,
    ACTION_WRITE_OBJECTIVE_PLAN,
    ACTION_WRITE_OBJECTIVE_PREVIEW,
    ACTION_WRITE_POLICY_DECISION,
    ACTION_WRITE_APPROVAL_REQUEST,
    ACTION_WRITE_KANBAN_METADATA,
    ACTION_CREATE_KANBAN_TASK,
    ACTION_ASSIGN_KANBAN_TASK,
    ACTION_SPAWN_WORKER,
    ACTION_EXECUTE_BACKGROUND_PROCESS,
    ACTION_NETWORK_CALL,
    ACTION_PROVIDER_API_CALL,
    ACTION_EXTERNAL_API_CALL,
)


def compute_decision_fingerprint(
    objective_id: str,
    risk_level: RiskLevel,
    allowed_actions: tuple[str, ...],
    forbidden_actions: tuple[str, ...],
    approval_required: bool,
    risk_score: float,
    risk_components: dict,
) -> str:
    """Stable fingerprint of a PolicyDecision's canonical inputs.

    Two PolicyDecisions with the same canonical inputs (objective_id,
    risk_level, allowed/forbidden actions, approval_required, risk_score,
    risk_components) yield the same fingerprint, regardless of
    ``created_at``.
    """
    canonical = json.dumps(
        {
            "objective_id": objective_id,
            "risk_level": int(risk_level),
            "allowed_actions": sorted(allowed_actions),
            "forbidden_actions": sorted(forbidden_actions),
            "approval_required": approval_required,
            "risk_score": round(float(risk_score), 6),
            "risk_components": dict(
                sorted(
                    (str(k), float(v))
                    for k, v in (risk_components or {}).items()
                )
            ),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_request_fingerprint(
    objective_id: str,
    risk_level: RiskLevel,
    approver_id: Optional[str],
    approval_token: Optional[str],
    kanban_approver_id: Optional[str],
    worker_approver_id: Optional[str],
    external_approver_id: Optional[str],
    scope: tuple[str, ...],
) -> str:
    """Stable fingerprint of an ApprovalRequest's canonical inputs."""
    canonical = json.dumps(
        {
            "objective_id": objective_id,
            "risk_level": int(risk_level),
            "approver_id": approver_id,
            "approval_token": approval_token,
            "kanban_approver_id": kanban_approver_id,
            "worker_approver_id": worker_approver_id,
            "external_approver_id": external_approver_id,
            "scope": sorted(scope),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PolicyDecision:
    """Phase 4A output: a centralized policy decision for an objective.

    Tells the caller (Phase 4B or human reviewer) what risk level
    was computed, what actions are allowed/forbidden, and whether
    human approval is required.

    Stored in ``state_meta[objective_policy_decision:<oid>]`` as JSON.
    """

    objective_id: str
    risk_level: RiskLevel
    allowed_actions: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    approval_required: bool
    warnings: tuple[str, ...]
    approval_requirements: tuple[dict, ...]
    risk_score: float
    risk_components: dict
    created_at: str  # ISO 8601
    decision_fingerprint: str  # sha256 of canonical inputs

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "risk_level": int(self.risk_level),
            "risk_level_name": f"R{int(self.risk_level)}",
            "allowed_actions": list(self.allowed_actions),
            "forbidden_actions": list(self.forbidden_actions),
            "approval_required": self.approval_required,
            "warnings": list(self.warnings),
            "approval_requirements": list(self.approval_requirements),
            "risk_score": self.risk_score,
            "risk_components": dict(self.risk_components),
            "created_at": self.created_at,
            "decision_fingerprint": self.decision_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyDecision":
        risk_value = data.get("risk_level", 0)
        try:
            risk_level = RiskLevel(int(risk_value))
        except (TypeError, ValueError):
            risk_level = RiskLevel.R0
        return cls(
            objective_id=str(data.get("objective_id", "")),
            risk_level=risk_level,
            allowed_actions=tuple(data.get("allowed_actions") or ()),
            forbidden_actions=tuple(data.get("forbidden_actions") or ()),
            approval_required=bool(data.get("approval_required", False)),
            warnings=tuple(data.get("warnings") or ()),
            approval_requirements=tuple(data.get("approval_requirements") or ()),
            risk_score=float(data.get("risk_score", 0.0) or 0.0),
            risk_components=dict(data.get("risk_components") or {}),
            created_at=str(data.get("created_at", "")),
            decision_fingerprint=str(data.get("decision_fingerprint", "")),
        )


@dataclass(frozen=True)
class ApprovalRequest:
    """Phase 4A output: a human approval request for a high-risk
    policy decision.

    Built by ``ApprovalGateEvaluator.evaluate(...)`` after all
    applicable gates pass. Stored in
    ``state_meta[objective_approval_request:<oid>]`` as JSON.
    """

    objective_id: str
    risk_level: RiskLevel
    approver_id: Optional[str]
    approval_token: Optional[str]
    kanban_approver_id: Optional[str]
    worker_approver_id: Optional[str]
    external_approver_id: Optional[str]
    approval_reason: str
    scope: tuple[str, ...]
    expiry: Optional[str]  # ISO 8601, or None for no expiry.
    created_at: str
    request_fingerprint: str
    policy_decision_fingerprint: str

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "risk_level": int(self.risk_level),
            "risk_level_name": f"R{int(self.risk_level)}",
            "approver_id": self.approver_id,
            "approval_token": self.approval_token,
            "kanban_approver_id": self.kanban_approver_id,
            "worker_approver_id": self.worker_approver_id,
            "external_approver_id": self.external_approver_id,
            "approval_reason": self.approval_reason,
            "scope": list(self.scope),
            "expiry": self.expiry,
            "created_at": self.created_at,
            "request_fingerprint": self.request_fingerprint,
            "policy_decision_fingerprint": self.policy_decision_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ApprovalRequest":
        risk_value = data.get("risk_level", 0)
        try:
            risk_level = RiskLevel(int(risk_value))
        except (TypeError, ValueError):
            risk_level = RiskLevel.R0
        return cls(
            objective_id=str(data.get("objective_id", "")),
            risk_level=risk_level,
            approver_id=data.get("approver_id"),
            approval_token=data.get("approval_token"),
            kanban_approver_id=data.get("kanban_approver_id"),
            worker_approver_id=data.get("worker_approver_id"),
            external_approver_id=data.get("external_approver_id"),
            approval_reason=str(data.get("approval_reason", "")),
            scope=tuple(data.get("scope") or ()),
            expiry=data.get("expiry"),
            created_at=str(data.get("created_at", "")),
            request_fingerprint=str(data.get("request_fingerprint", "")),
            policy_decision_fingerprint=str(
                data.get("policy_decision_fingerprint", "")
            ),
        )



# ── Phase 5 Worker Dispatch types ────────────────────────────────

# Phase 5 is the "dispatch" layer that consumes Phase 4B's Kanban
# tasks and the Phase 4A approval gates, and produces real worker
# runs through the existing agent/orchestrator/* infrastructure.
#
# The dispatch layer is **hermetic**:
# - It re-validates the 8 Phase 4A approval gates (incl. Layer 6
#   Worker_spawn R5 which requires `worker_approver_id`).
# - It reuses `agent.orchestrator.dispatcher.Dispatcher.dispatch`
#   and `agent.orchestrator.batch_runner.BatchRunner.run_batch`
#   as the ONLY entry points for worker decision and execution.
# - It does NOT call `ExecutionRouter`, `ExecutionDispatcher`,
#   `OrchestratorInterface.execute`, or any LLM/provider API.
# - It does NOT spawn workers directly; the only public worker
#   spawn function is `agent.orchestrator.worker_runner.run_worker_subprocess`
#   which is invoked by the `RUN_WORKER` handler built by
#   `agent.orchestrator.handlers.make_handlers`.

OBJECTIVE_WORKER_DISPATCH_PREFIX = "objective_worker_dispatch:"
OBJECTIVE_WORKER_DISPATCH_TASKS_PREFIX = "objective_worker_dispatch_tasks:"


def objective_worker_dispatch_key(objective_id: str) -> str:
    """Return the state_meta key for the dispatch record of an objective."""
    return f"{OBJECTIVE_WORKER_DISPATCH_PREFIX}{objective_id}"


def objective_worker_dispatch_tasks_key(objective_id: str) -> str:
    """Return the state_meta key for the task_ids list of a dispatch."""
    return f"{OBJECTIVE_WORKER_DISPATCH_TASKS_PREFIX}{objective_id}"


def compute_dispatch_fingerprint(  # canonical name; same algorithm as worker_mapping.compute_dispatch_fingerprint
    task_ids,
    *,
    restrictions,
    decision_fingerprint,
    request_fingerprint,
    kanban_apply_fingerprint,
):
    """Stable sha256 fingerprint of canonical dispatch inputs.

    Excludes `created_at` so re-applies with the same inputs are
    considered identical.
    """
    import hashlib
    import json
    payload = {
        "task_ids": sorted(task_ids),
        "restrictions": sorted(restrictions),
        "decision_fingerprint": decision_fingerprint,
        "request_fingerprint": request_fingerprint,
        "kanban_apply_fingerprint": kanban_apply_fingerprint,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WorkerDispatchTaskLink:
    """Link a single kanban task to its worker dispatch record.

    Phase 5 emits one of these per task in a dispatch. It is the
    analog of Phase 4B's ``KanbanTaskLink`` but holds the worker's
    own metadata (worker_id, batch_run_id, decision, action_executed).
    """
    objective_id: str
    task_id: str
    worker_id: str
    decision: str  # raw JSON-serializable decision dict
    action_executed: str
    trace_line: Tuple[Tuple[str, str], ...]  # (key, json-serialized-value) pairs
    created_at: str

    def to_dict(self) -> dict:
        d = {
            "objective_id": self.objective_id,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "decision": self.decision,
            "action_executed": self.action_executed,
            "trace_line": dict(self.trace_line),
            "created_at": self.created_at,
        }
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "WorkerDispatchTaskLink":
        trace = data.get("trace_line") or {}
        # Normalize trace_line to a tuple of (key, str) pairs.
        if isinstance(trace, dict):
            trace_items = tuple(
                (str(k), json.dumps(v, sort_keys=True, default=str))
                for k, v in trace.items()
            )
        else:
            trace_items = ()
        return cls(
            objective_id=str(data.get("objective_id", "")),
            task_id=str(data.get("task_id", "")),
            worker_id=str(data.get("worker_id", "")),
            decision=json.dumps(data.get("decision", {}), sort_keys=True, default=str),
            action_executed=str(data.get("action_executed", "")),
            trace_line=trace_items,
            created_at=str(data.get("created_at", "")),
        )


@dataclass(frozen=True)
class WorkerDispatchPreview:
    """Phase 5 dry-run output.

    The preview is pure compute: no kanban writes, no
    BatchRunner.run_batch call, no subprocess spawn. It is
    produced by ``WorkerDispatchEngine.dry_run`` and
    ``worker_dispatch_dry_run``.
    """
    objective_id: str
    kanban_task_ids: Tuple[str, ...]
    task_states: Tuple[dict, ...]  # one TaskState-shaped dict per task
    workers: Tuple[dict, ...]  # one WorkerRegistryEntry-shaped dict per unique assignee
    restrictions: FrozenSet[str]
    warnings: Tuple[str, ...]
    dispatch_fingerprint: str
    created_at: str

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "kanban_task_ids": list(self.kanban_task_ids),
            "task_states": list(self.task_states),
            "workers": list(self.workers),
            "restrictions": sorted(self.restrictions),
            "warnings": list(self.warnings),
            "dispatch_fingerprint": self.dispatch_fingerprint,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkerDispatchPreview":
        return cls(
            objective_id=str(data.get("objective_id", "")),
            kanban_task_ids=tuple(data.get("kanban_task_ids", []) or ()),
            task_states=tuple(
                dict(s) if not isinstance(s, dict) else s
                for s in (data.get("task_states") or ())
            ),
            workers=tuple(
                dict(w) if not isinstance(w, dict) else w
                for w in (data.get("workers") or ())
            ),
            restrictions=frozenset(data.get("restrictions") or ()),
            warnings=tuple(data.get("warnings") or ()),
            dispatch_fingerprint=str(data.get("dispatch_fingerprint", "")),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(frozen=True)
class WorkerDispatchResult:
    """Phase 5 apply output.

    Produced by ``WorkerDispatchEngine.apply``. Side effects:
    - 0 or 1 ``BatchRunner.run_batch`` call (only after all
      approval gates pass and the lineage check passes).
    - state_meta writes (objective_worker_dispatch:<oid> and
      objective_worker_dispatch_tasks:<oid>).
    """
    objective_id: str
    task_ids: Tuple[str, ...]
    worker_runs: Tuple[dict, ...]  # one per dispatched task (DispatchResult.to_dict())
    worker_runs_started: int
    worker_runs_failed: int
    dispatch_fingerprint: str
    decision_fingerprint: str
    request_fingerprint: str
    kanban_apply_fingerprint: str
    duplicate: bool
    errors: Tuple[dict, ...]
    created_at: str

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "task_ids": list(self.task_ids),
            "worker_runs": list(self.worker_runs),
            "worker_runs_started": self.worker_runs_started,
            "worker_runs_failed": self.worker_runs_failed,
            "dispatch_fingerprint": self.dispatch_fingerprint,
            "decision_fingerprint": self.decision_fingerprint,
            "request_fingerprint": self.request_fingerprint,
            "kanban_apply_fingerprint": self.kanban_apply_fingerprint,
            "duplicate": self.duplicate,
            "errors": list(self.errors),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkerDispatchResult":
        return cls(
            objective_id=str(data.get("objective_id", "")),
            task_ids=tuple(data.get("task_ids", []) or ()),
            worker_runs=tuple(data.get("worker_runs") or ()),
            worker_runs_started=int(data.get("worker_runs_started", 0) or 0),
            worker_runs_failed=int(data.get("worker_runs_failed", 0) or 0),
            dispatch_fingerprint=str(data.get("dispatch_fingerprint", "")),
            decision_fingerprint=str(data.get("decision_fingerprint", "")),
            request_fingerprint=str(data.get("request_fingerprint", "")),
            kanban_apply_fingerprint=str(data.get("kanban_apply_fingerprint", "")),
            duplicate=bool(data.get("duplicate", False)),
            errors=tuple(data.get("errors") or ()),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(frozen=True)
class WorkerDispatchRollbackPlan:
    """Plan for rolling back a worker dispatch.

    Produced by ``WorkerDispatchRollbackPlan.from_dispatch_record``.
    Holds the list of task_ids to archive (in reverse-creation order)
    and the mode ("archive" or "hard_delete").
    """
    objective_id: str
    task_ids: Tuple[str, ...]  # in reverse-creation order
    dispatch_fingerprint: str
    mode: str  # "archive" or "hard_delete"

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "task_ids": list(self.task_ids),
            "dispatch_fingerprint": self.dispatch_fingerprint,
            "mode": self.mode,
        }

    @classmethod
    def from_dispatch_record(
        cls,
        record: "WorkerDispatchResult",
        *,
        mode: str = "archive",
    ) -> "WorkerDispatchRollbackPlan":
        if mode not in ("archive", "hard_delete"):
            raise ValueError(
                f"WorkerDispatchRollbackPlan: invalid mode={mode!r} "
                f"(expected 'archive' or 'hard_delete')"
            )
        # Reverse the task_ids order so the most-recently-created
        # is rolled back first.
        return cls(
            objective_id=record.objective_id,
            task_ids=tuple(reversed(record.task_ids)),
            dispatch_fingerprint=record.dispatch_fingerprint,
            mode=mode,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "WorkerDispatchRollbackPlan":
        return cls(
            objective_id=str(data.get("objective_id", "")),
            task_ids=tuple(data.get("task_ids", []) or ()),
            dispatch_fingerprint=str(data.get("dispatch_fingerprint", "")),
            mode=str(data.get("mode", "archive")),
        )



# ── Phase 6 Success Evaluator types ───────────────────────────────

# Phase 6 is a **read-only aggregator** over Phase 5's
# WorkerDispatchResult. It produces an EvaluationReport (deterministic;
# no LLM, no provider, no subprocess). It does NOT spawn workers,
# does NOT call Orchestrator/Dispatcher/BatchRunner, does NOT modify
# Runtime.

# Status enum (the 5 possible outcomes).
class SuccessStatus(str, Enum):
    """The 5 possible outcomes of a Phase 6 evaluation."""
    SUCCESS          = "success"
    PARTIAL_SUCCESS  = "partial_success"
    FAILED           = "failed"
    BLOCKED          = "blocked"
    ABORTED          = "aborted"


# Per-task outcome (internal classification).
class TaskOutcome(str, Enum):
    """The 5 possible per-task outcomes (internal use only)."""
    SUCCESSFUL = "successful"
    FAILED     = "failed"
    BLOCKED    = "blocked"
    CANCELLED  = "cancelled"
    MISSING    = "missing"


# Prefix constants for state_meta keys.
OBJECTIVE_EVALUATION_PREFIX = "objective_evaluation:"
OBJECTIVE_SUCCESS_REPORT_PREFIX = "objective_success_report:"


def objective_evaluation_key(objective_id: str) -> str:
    """Return the state_meta key for the evaluation record of an objective."""
    return f"{OBJECTIVE_EVALUATION_PREFIX}{objective_id}"


def objective_success_report_key(objective_id: str) -> str:
    """Return the state_meta key for the success report of an objective."""
    return f"{OBJECTIVE_SUCCESS_REPORT_PREFIX}{objective_id}"


def compute_evaluation_fingerprint(
    objective_id: str,
    dispatch_fingerprint: str,
    decision_fingerprint: str,
    request_fingerprint: str,
    plan_fingerprint: str,
    goal_fingerprint: str,
    status: str,
) -> str:
    """Stable sha256 fingerprint of canonical evaluation inputs.

    Excludes `created_at` so re-evaluations with the same inputs are
    considered identical.
    """
    import hashlib
    import json
    payload = {
        "objective_id": objective_id,
        "dispatch_fingerprint": dispatch_fingerprint,
        "decision_fingerprint": decision_fingerprint,
        "request_fingerprint": request_fingerprint,
        "plan_fingerprint": plan_fingerprint,
        "goal_fingerprint": goal_fingerprint,
        "status": status,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SuccessMetricBreakdown:
    """Detailed metric breakdown for an EvaluationReport.

    Phase 6 emits this as part of EvaluationReport so the operator
    can see not just the headline number but its components.
    """
    successful_tasks: int
    failed_tasks: int
    blocked_tasks: int
    cancelled_tasks: int
    missing_tasks: int
    total_tasks: int
    per_task_completion_sum: float
    coverage: float
    worker_success_rate: float
    mean_score: float
    evidence_score: float
    confidence_score: float
    completion_percentage: float

    def to_dict(self) -> dict:
        return {
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "blocked_tasks": self.blocked_tasks,
            "cancelled_tasks": self.cancelled_tasks,
            "missing_tasks": self.missing_tasks,
            "total_tasks": self.total_tasks,
            "per_task_completion_sum": self.per_task_completion_sum,
            "coverage": self.coverage,
            "worker_success_rate": self.worker_success_rate,
            "mean_score": self.mean_score,
            "evidence_score": self.evidence_score,
            "confidence_score": self.confidence_score,
            "completion_percentage": self.completion_percentage,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SuccessMetricBreakdown":
        return cls(
            successful_tasks=int(data.get("successful_tasks", 0) or 0),
            failed_tasks=int(data.get("failed_tasks", 0) or 0),
            blocked_tasks=int(data.get("blocked_tasks", 0) or 0),
            cancelled_tasks=int(data.get("cancelled_tasks", 0) or 0),
            missing_tasks=int(data.get("missing_tasks", 0) or 0),
            total_tasks=int(data.get("total_tasks", 0) or 0),
            per_task_completion_sum=float(data.get("per_task_completion_sum", 0.0) or 0.0),
            coverage=float(data.get("coverage", 0.0) or 0.0),
            worker_success_rate=float(data.get("worker_success_rate", 0.0) or 0.0),
            mean_score=float(data.get("mean_score", 0.0) or 0.0),
            evidence_score=float(data.get("evidence_score", 0.0) or 0.0),
            confidence_score=float(data.get("confidence_score", 0.0) or 0.0),
            completion_percentage=float(data.get("completion_percentage", 0.0) or 0.0),
        )


@dataclass(frozen=True)
class EvaluationReport:
    """Phase 6 evaluation output.

    Produced by ``SuccessEvaluatorEngine.evaluate`` and persisted to
    ``state_meta[objective_evaluation:<oid>]``.

    This is a **read-only** artifact over Phase 1+5: it does NOT mutate
    any of the source artifacts (WorkerDispatchResult, KanbanApplyResult,
    PolicyDecision, ApprovalRequest, ObjectivePlan, etc.).
    """
    objective_id: str
    execution_fingerprint: str
    worker_dispatch_fingerprint: str
    policy_fingerprint: str
    approval_fingerprint: str
    plan_fingerprint: str
    goal_fingerprint: str
    objective_fingerprint: str
    status: SuccessStatus
    completion_percentage: float
    successful_tasks: int
    failed_tasks: int
    blocked_tasks: int
    cancelled_tasks: int
    worker_success_rate: float
    evidence_score: float
    confidence_score: float
    retry_recommended: bool
    retry_reason: str
    manual_intervention_required: bool
    remaining_tasks: Tuple[str, ...]
    summary: str
    metrics: SuccessMetricBreakdown
    created_at: str
    created_by: str

    @property
    def missing_tasks(self) -> int:
        """Number of task_ids without a corresponding worker outcome.

        Kept as a derived top-level convenience so callers can inspect all
        task outcome counters uniformly while the persisted canonical value
        remains in ``metrics.missing_tasks``.
        """
        return self.metrics.missing_tasks

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "execution_fingerprint": self.execution_fingerprint,
            "worker_dispatch_fingerprint": self.worker_dispatch_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "approval_fingerprint": self.approval_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "goal_fingerprint": self.goal_fingerprint,
            "objective_fingerprint": self.objective_fingerprint,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "completion_percentage": self.completion_percentage,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "blocked_tasks": self.blocked_tasks,
            "cancelled_tasks": self.cancelled_tasks,
            "worker_success_rate": self.worker_success_rate,
            "evidence_score": self.evidence_score,
            "confidence_score": self.confidence_score,
            "retry_recommended": self.retry_recommended,
            "retry_reason": self.retry_reason,
            "manual_intervention_required": self.manual_intervention_required,
            "remaining_tasks": list(self.remaining_tasks),
            "summary": self.summary,
            "metrics": self.metrics.to_dict(),
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationReport":
        status_str = data.get("status", "failed")
        try:
            status = SuccessStatus(status_str)
        except (ValueError, KeyError):
            status = SuccessStatus.FAILED
        return cls(
            objective_id=str(data.get("objective_id", "")),
            execution_fingerprint=str(data.get("execution_fingerprint", "")),
            worker_dispatch_fingerprint=str(data.get("worker_dispatch_fingerprint", "")),
            policy_fingerprint=str(data.get("policy_fingerprint", "")),
            approval_fingerprint=str(data.get("approval_fingerprint", "")),
            plan_fingerprint=str(data.get("plan_fingerprint", "")),
            goal_fingerprint=str(data.get("goal_fingerprint", "")),
            objective_fingerprint=str(data.get("objective_fingerprint", "")),
            status=status,
            completion_percentage=float(data.get("completion_percentage", 0.0) or 0.0),
            successful_tasks=int(data.get("successful_tasks", 0) or 0),
            failed_tasks=int(data.get("failed_tasks", 0) or 0),
            blocked_tasks=int(data.get("blocked_tasks", 0) or 0),
            cancelled_tasks=int(data.get("cancelled_tasks", 0) or 0),
            worker_success_rate=float(data.get("worker_success_rate", 0.0) or 0.0),
            evidence_score=float(data.get("evidence_score", 0.0) or 0.0),
            confidence_score=float(data.get("confidence_score", 0.0) or 0.0),
            retry_recommended=bool(data.get("retry_recommended", False)),
            retry_reason=str(data.get("retry_reason", "")),
            manual_intervention_required=bool(data.get("manual_intervention_required", False)),
            remaining_tasks=tuple(data.get("remaining_tasks") or ()),
            summary=str(data.get("summary", "")),
            metrics=SuccessMetricBreakdown.from_dict(data.get("metrics") or {}),
            created_at=str(data.get("created_at", "")),
            created_by=str(data.get("created_by", "")),
        )


@dataclass(frozen=True)
class SuccessReport:
    """Slimmed-down public report (no fingerprints).

    Persisted to ``state_meta[objective_success_report:<oid>]`` for
    human consumption.
    """
    objective_id: str
    status: SuccessStatus
    completion_percentage: float
    successful_tasks: int
    failed_tasks: int
    blocked_tasks: int
    cancelled_tasks: int
    summary: str
    created_at: str
    created_by: str

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "completion_percentage": self.completion_percentage,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "blocked_tasks": self.blocked_tasks,
            "cancelled_tasks": self.cancelled_tasks,
            "summary": self.summary,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SuccessReport":
        status_str = data.get("status", "failed")
        try:
            status = SuccessStatus(status_str)
        except (ValueError, KeyError):
            status = SuccessStatus.FAILED
        return cls(
            objective_id=str(data.get("objective_id", "")),
            status=status,
            completion_percentage=float(data.get("completion_percentage", 0.0) or 0.0),
            successful_tasks=int(data.get("successful_tasks", 0) or 0),
            failed_tasks=int(data.get("failed_tasks", 0) or 0),
            blocked_tasks=int(data.get("blocked_tasks", 0) or 0),
            cancelled_tasks=int(data.get("cancelled_tasks", 0) or 0),
            summary=str(data.get("summary", "")),
            created_at=str(data.get("created_at", "")),
            created_by=str(data.get("created_by", "")),
        )



# ── Phase 7 Objective Recovery types ───────────────────────────────

# Phase 7 is a **read-and-recommend** layer that sits on top of
# Phase 6 (Success Evaluator). It reads EvaluationReport and other
# Phase 1-6 state, classifies the outcome into a RecoveryStatus,
# recommends a RecoveryAction, and produces a RecoveryPlanPreview.
# It does NOT spawn workers, does NOT create Kanban, does NOT call
# Orchestrator/Dispatcher/BatchRunner, does NOT modify Runtime.


class RecoveryStatus(str, Enum):
    """The 7 possible recovery outcomes of a Phase 7 evaluation."""
    RECOVERABLE         = "recoverable"
    NEEDS_HUMAN         = "needs_human"
    REPLAN_RECOMMENDED  = "replan_recommended"
    ACCEPT_PARTIAL      = "accept_partial"
    ABORT_RECOMMENDED   = "abort_recommended"
    NOT_RECOVERABLE     = "not_recoverable"
    NO_ACTION_NEEDED    = "no_action_needed"


class RecoveryAction(str, Enum):
    """The 8 possible recovery actions recommended by Phase 7."""
    RETRY_FAILED_TASKS        = "retry_failed_tasks"
    RETRY_BLOCKED_TASKS       = "retry_blocked_tasks"
    REQUEST_WORKER            = "request_worker"
    REQUEST_APPROVAL          = "request_approval"
    REPLAN_OBJECTIVE          = "replan_objective"
    ACCEPT_PARTIAL_SUCCESS    = "accept_partial_success"
    ABORT_OBJECTIVE           = "abort_objective"
    NOOP                      = "noop"


OBJECTIVE_RECOVERY_DIAGNOSIS_PREFIX = "objective_recovery_diagnosis:"
OBJECTIVE_RECOVERY_PLAN_PREFIX = "objective_recovery_plan:"


def objective_recovery_diagnosis_key(objective_id: str) -> str:
    """Return the state_meta key for the recovery diagnosis of an objective."""
    return f"{OBJECTIVE_RECOVERY_DIAGNOSIS_PREFIX}{objective_id}"


def objective_recovery_plan_key(objective_id: str) -> str:
    """Return the state_meta key for the recovery plan of an objective."""
    return f"{OBJECTIVE_RECOVERY_PLAN_PREFIX}{objective_id}"


def compute_recovery_diagnosis_fingerprint(
    objective_id: str,
    evaluation_fingerprint: str,
    worker_dispatch_fingerprint: str,
    policy_fingerprint: str,
    approval_fingerprint: str,
    plan_fingerprint: str,
    goal_fingerprint: str,
    objective_fingerprint: str,
    recovery_status: str,
    failed_task_ids: Tuple[str, ...],
    blocked_task_ids: Tuple[str, ...],
    cancelled_task_ids: Tuple[str, ...],
    missing_task_ids: Tuple[str, ...],
    transient_failures: int,
    permanent_failures: int,
) -> str:
    """Stable sha256 fingerprint of canonical recovery diagnosis inputs.

    Excludes `created_at` so re-diagnoses with the same inputs are
    considered identical.
    """
    import hashlib
    import json
    payload = {
        "objective_id": objective_id,
        "evaluation_fingerprint": evaluation_fingerprint,
        "worker_dispatch_fingerprint": worker_dispatch_fingerprint,
        "policy_fingerprint": policy_fingerprint,
        "approval_fingerprint": approval_fingerprint,
        "plan_fingerprint": plan_fingerprint,
        "goal_fingerprint": goal_fingerprint,
        "objective_fingerprint": objective_fingerprint,
        "recovery_status": recovery_status,
        "failed_task_ids": list(failed_task_ids),
        "blocked_task_ids": list(blocked_task_ids),
        "cancelled_task_ids": list(cancelled_task_ids),
        "missing_task_ids": list(missing_task_ids),
        "transient_failures": transient_failures,
        "permanent_failures": permanent_failures,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RecoveryDiagnosis:
    """Phase 7 diagnosis output.

    Produced by ``recovery_diagnose`` and persisted to
    ``state_meta[objective_recovery_diagnosis:<oid>]``.

    This is a **read-only** artifact over Phase 1-6: it does NOT
    mutate any of the source artifacts.
    """
    objective_id: str
    evaluation_fingerprint: str
    worker_dispatch_fingerprint: str
    policy_fingerprint: str
    approval_fingerprint: str
    plan_fingerprint: str
    goal_fingerprint: str
    objective_fingerprint: str
    evaluation_status: SuccessStatus
    recovery_status: RecoveryStatus
    failed_task_ids: Tuple[str, ...]
    blocked_task_ids: Tuple[str, ...]
    cancelled_task_ids: Tuple[str, ...]
    missing_task_ids: Tuple[str, ...]
    transient_failures: int
    permanent_failures: int
    blocked_reasons: Tuple[str, ...]
    aborted_flag: bool
    manual_intervention_required: bool
    rationale: str
    created_at: str
    created_by: str

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "worker_dispatch_fingerprint": self.worker_dispatch_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "approval_fingerprint": self.approval_fingerprint,
            "plan_fingerprint": self.plan_fingerprint,
            "goal_fingerprint": self.goal_fingerprint,
            "objective_fingerprint": self.objective_fingerprint,
            "evaluation_status": self.evaluation_status.value if hasattr(self.evaluation_status, "value") else str(self.evaluation_status),
            "recovery_status": self.recovery_status.value if hasattr(self.recovery_status, "value") else str(self.recovery_status),
            "failed_task_ids": list(self.failed_task_ids),
            "blocked_task_ids": list(self.blocked_task_ids),
            "cancelled_task_ids": list(self.cancelled_task_ids),
            "missing_task_ids": list(self.missing_task_ids),
            "transient_failures": self.transient_failures,
            "permanent_failures": self.permanent_failures,
            "blocked_reasons": list(self.blocked_reasons),
            "aborted_flag": self.aborted_flag,
            "manual_intervention_required": self.manual_intervention_required,
            "rationale": self.rationale,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecoveryDiagnosis":
        try:
            eval_status = SuccessStatus(data.get("evaluation_status", "failed"))
        except (ValueError, KeyError):
            eval_status = SuccessStatus.FAILED
        try:
            rec_status = RecoveryStatus(data.get("recovery_status", "abort_recommended"))
        except (ValueError, KeyError):
            rec_status = RecoveryStatus.ABORT_RECOMMENDED
        return cls(
            objective_id=str(data.get("objective_id", "")),
            evaluation_fingerprint=str(data.get("evaluation_fingerprint", "")),
            worker_dispatch_fingerprint=str(data.get("worker_dispatch_fingerprint", "")),
            policy_fingerprint=str(data.get("policy_fingerprint", "")),
            approval_fingerprint=str(data.get("approval_fingerprint", "")),
            plan_fingerprint=str(data.get("plan_fingerprint", "")),
            goal_fingerprint=str(data.get("goal_fingerprint", "")),
            objective_fingerprint=str(data.get("objective_fingerprint", "")),
            evaluation_status=eval_status,
            recovery_status=rec_status,
            failed_task_ids=tuple(data.get("failed_task_ids") or ()),
            blocked_task_ids=tuple(data.get("blocked_task_ids") or ()),
            cancelled_task_ids=tuple(data.get("cancelled_task_ids") or ()),
            missing_task_ids=tuple(data.get("missing_task_ids") or ()),
            transient_failures=int(data.get("transient_failures", 0) or 0),
            permanent_failures=int(data.get("permanent_failures", 0) or 0),
            blocked_reasons=tuple(data.get("blocked_reasons") or ()),
            aborted_flag=bool(data.get("aborted_flag", False)),
            manual_intervention_required=bool(data.get("manual_intervention_required", False)),
            rationale=str(data.get("rationale", "")),
            created_at=str(data.get("created_at", "")),
            created_by=str(data.get("created_by", "")),
        )


@dataclass(frozen=True)
class RecoveryPlanPreview:
    """Phase 7 plan output (slim, no fingerprints).

    Persisted to ``state_meta[objective_recovery_plan:<oid>]`` for
    human consumption.
    """
    objective_id: str
    diagnosis_fingerprint: str
    recommended_action: RecoveryAction
    recommended_retry_task_ids: Tuple[str, ...]
    recommended_replan_rationale: str
    human_approval_required: bool
    human_approval_rationale: str
    rollback_safe: bool
    estimated_wasted_cycles: int
    summary: str
    next_step_recommendation: str
    created_at: str
    created_by: str

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "diagnosis_fingerprint": self.diagnosis_fingerprint,
            "recommended_action": self.recommended_action.value if hasattr(self.recommended_action, "value") else str(self.recommended_action),
            "recommended_retry_task_ids": list(self.recommended_retry_task_ids),
            "recommended_replan_rationale": self.recommended_replan_rationale,
            "human_approval_required": self.human_approval_required,
            "human_approval_rationale": self.human_approval_rationale,
            "rollback_safe": self.rollback_safe,
            "estimated_wasted_cycles": self.estimated_wasted_cycles,
            "summary": self.summary,
            "next_step_recommendation": self.next_step_recommendation,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RecoveryPlanPreview":
        try:
            action = RecoveryAction(data.get("recommended_action", "noop"))
        except (ValueError, KeyError):
            action = RecoveryAction.NOOP
        return cls(
            objective_id=str(data.get("objective_id", "")),
            diagnosis_fingerprint=str(data.get("diagnosis_fingerprint", "")),
            recommended_action=action,
            recommended_retry_task_ids=tuple(data.get("recommended_retry_task_ids") or ()),
            recommended_replan_rationale=str(data.get("recommended_replan_rationale", "")),
            human_approval_required=bool(data.get("human_approval_required", False)),
            human_approval_rationale=str(data.get("human_approval_rationale", "")),
            rollback_safe=bool(data.get("rollback_safe", True)),
            estimated_wasted_cycles=int(data.get("estimated_wasted_cycles", 0) or 0),
            summary=str(data.get("summary", "")),
            next_step_recommendation=str(data.get("next_step_recommendation", "")),
            created_at=str(data.get("created_at", "")),
            created_by=str(data.get("created_by", "")),
        )

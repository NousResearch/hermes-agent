"""ORCH V4 lifecycle transition and cancellation-fencing model.

This module is deliberately runtime-independent.  It models the durable CAS
rules from contract §7.2/§7.4 so the focused tests can exercise every edge
without opening the live Kanban database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})
NONTERMINAL_STATES = frozenset({
    "submitted", "decomposing", "waiting_lanes", "synthesizing",
    "work_accepted", "delivering", "delivery_blocked", "blocked", "cancelling",
})

# (from_state, event, to_state), copied from §7.2.
TRANSITION_TABLE = (
    ("submitted", "claim_decomposition", "decomposing"),
    ("decomposing", "valid_plan_materialized", "waiting_lanes"),
    ("decomposing", "needs_human", "blocked"),
    ("decomposing", "unrecoverable_or_exhausted", "failed"),
    ("waiting_lanes", "retriable_lane_failure", "waiting_lanes"),
    ("waiting_lanes", "required_needs_human", "blocked"),
    ("waiting_lanes", "required_exhausted", "failed"),
    ("waiting_lanes", "required_set_accepted", "synthesizing"),
    ("synthesizing", "needs_human", "blocked"),
    ("synthesizing", "result_accepted", "work_accepted"),
    ("work_accepted", "required_routes_exist", "delivering"),
    ("work_accepted", "explicit_board_only", "completed"),
    ("delivering", "delivery_satisfied", "completed"),
    ("delivering", "required_dead_letter_or_unknown_timeout", "delivery_blocked"),
    ("delivery_blocked", "retry_authorized", "delivering"),
    ("blocked", "matching_input_resolved", "resume_state"),
    ("cancelling", "all_children_claims_sends_drained", "cancelled"),
)


class LifecycleError(ValueError):
    """A lifecycle CAS or fencing rule rejected a transition."""


class StaleCallback(LifecycleError):
    """A callback carries an old lifecycle revision/cancel epoch."""


def transition_allowed(
    from_state: str,
    event: str,
    to_state: str,
    *,
    resume_state: str | None = None,
) -> bool:
    """Return whether one §7.2 edge is legal, including cancel and unblock."""
    if from_state in TERMINAL_STATES:
        return False
    if event == "cancel_authorized":
        return from_state in NONTERMINAL_STATES - {"cancelling"} and to_state == "cancelling"
    if event == "matching_input_resolved":
        return from_state == "blocked" and to_state == resume_state and resume_state in {
            "decomposing", "waiting_lanes", "synthesizing"
        }
    if event in {"block", "unblock", "retry"} and from_state in TERMINAL_STATES:
        return False
    for src, ev, dst in TRANSITION_TABLE:
        if (src, ev) == (from_state, event):
            return to_state == (resume_state if dst == "resume_state" else dst)
    return False


@dataclass
class Request:
    board: str
    tenant: str
    orch_id: str
    state: str
    lifecycle_revision: int = 0
    cancel_epoch: int = 0
    generation: int = 1
    resume_state: str | None = None
    blocked_from_state: str | None = None
    block_revision: int = 0


@dataclass
class Node:
    board: str
    tenant: str
    orch_id: str
    node_key: str
    required: bool
    state: str = "planned"
    plan_version: int = 0


@dataclass
class Task:
    board: str
    tenant: str
    orch_id: str
    task_id: str
    status: str = "planned"
    cancellation_requested_at: int | None = None


@dataclass
class TaskRun:
    board: str
    tenant: str
    orch_id: str
    task_id: str
    run_id: str
    ended_at: int | None = None
    cancellation_epoch: int = 0


@dataclass
class StageLease:
    board: str
    tenant: str
    orch_id: str
    stage: str
    state: str = "active"
    epoch: int = 1


@dataclass
class DeliveryObligation:
    board: str
    tenant: str
    orch_id: str
    obligation_id: str
    state: str = "pending"
    duplicate_possible: bool = False


@dataclass
class DeliveryAttempt:
    board: str
    tenant: str
    orch_id: str
    obligation_id: str
    attempt_id: str
    state: str = "started"


@dataclass(frozen=True)
class CancelResult:
    old_revision: int
    new_revision: int
    old_cancel_epoch: int
    new_cancel_epoch: int
    stopped_task_ids: tuple[str, ...]
    retained_obligation_ids: tuple[str, ...]


def _same_scope(row: object, request: Request) -> bool:
    return (
        getattr(row, "board", None) == request.board
        and getattr(row, "tenant", None) == request.tenant
        and getattr(row, "orch_id", None) == request.orch_id
    )


def apply_transition(
    request: Request,
    event: str,
    to_state: str,
    *,
    resume_state: str | None = None,
    require_evidence: bool = True,
    accepted_required_lanes: int | None = None,
    has_result: bool | None = None,
    delivery_satisfied: bool | None = None,
    origin_kind: str | None = None,
    retry_budget_remaining: int | None = None,
) -> Request:
    """Apply one revision-CAS transition and return a new request snapshot.

    When require_evidence=True (default), gated events need durable evidence:
    - required_set_accepted: accepted_required_lanes >= 2
    - result_accepted: has_result is True
    - delivery_satisfied: delivery_satisfied is True
    - explicit_board_only: origin_kind == 'board_only'
    - retriable_lane_failure: retry_budget_remaining > 0
    - required_exhausted / unrecoverable_or_exhausted: retry_budget_remaining <= 0
    """
    if not transition_allowed(request.state, event, to_state, resume_state=resume_state):
        raise LifecycleError(f"invalid_transition:{request.state}:{event}:{to_state}")

    if require_evidence:
        if event == "required_set_accepted":
            if accepted_required_lanes is None or accepted_required_lanes < 2:
                raise LifecycleError("missing_required_lane_acceptances")
        elif event == "result_accepted":
            if has_result is not True:
                raise LifecycleError("missing_synthesis_result")
        elif event == "delivery_satisfied":
            if delivery_satisfied is not True:
                raise LifecycleError("delivery_not_satisfied")
        elif event == "explicit_board_only":
            if origin_kind != "board_only":
                raise LifecycleError("board_only_origin_required")
        elif event == "retriable_lane_failure":
            if retry_budget_remaining is None or retry_budget_remaining <= 0:
                raise LifecycleError("retry_budget_exhausted_or_missing")
        elif event in {"required_exhausted", "unrecoverable_or_exhausted"}:
            if retry_budget_remaining is None or retry_budget_remaining > 0:
                raise LifecycleError("exhaustion_not_proven")

    next_resume = resume_state if to_state == "blocked" else None
    next_blocked = request.state if to_state == "blocked" else None
    next_cancel_epoch = request.cancel_epoch + (1 if to_state == "cancelling" else 0)
    return Request(
        board=request.board,
        tenant=request.tenant,
        orch_id=request.orch_id,
        state=to_state,
        lifecycle_revision=request.lifecycle_revision + 1,
        cancel_epoch=next_cancel_epoch,
        generation=request.generation,
        resume_state=next_resume,
        blocked_from_state=next_blocked,
        block_revision=request.block_revision + (1 if to_state == "blocked" else 0),
    )


def request_optional_cancellation(nodes: Iterable[Node]) -> int:
    """Fence every unfinished optional node at the synthesis gate."""
    changed = 0
    for node in nodes:
        if not node.required and node.state in {"planned", "ready", "running", "blocked"}:
            node.state = "cancellation_requested"
            changed += 1
    return changed


def optional_cancel_drained(nodes: Iterable[Node]) -> bool:
    """Cancellation_requested is not terminal; all optional work must drain."""
    return not any(
        not node.required and node.state in {
            "planned", "ready", "running", "blocked", "cancellation_requested"
        }
        for node in nodes
    )


def complete_cancelled_node(node: Node) -> None:
    if node.state != "cancellation_requested":
        raise LifecycleError("node_not_cancellation_requested")
    node.state = "cancelled"


def cancel_cascade(
    request: Request,
    *,
    tasks: list[Task],
    task_runs: list[TaskRun],
    nodes: list[Node],
    leases: list[StageLease],
    obligations: list[DeliveryObligation],
    attempts: list[DeliveryAttempt],
    now: int,
) -> CancelResult:
    """Fence exactly one request scope and preserve ambiguity from started sends."""
    if request.state in TERMINAL_STATES or request.state == "cancelling":
        raise LifecycleError("cancel_not_authorized")
    old_revision, old_epoch = request.lifecycle_revision, request.cancel_epoch
    request.state = "cancelling"
    request.lifecycle_revision += 1
    request.cancel_epoch += 1
    request.resume_state = None
    request.blocked_from_state = None
    request.block_revision += 1

    stopped: list[str] = []
    for task in tasks:
        if not _same_scope(task, request):
            continue
        if task.status in {"planned", "ready"}:
            task.status = "cancelled"
            stopped.append(task.task_id)
        elif task.status == "running":
            task.cancellation_requested_at = now
            stopped.append(task.task_id)
    for run in task_runs:
        if _same_scope(run, request) and run.ended_at is None:
            run.cancellation_epoch = request.cancel_epoch
    for node in nodes:
        if not _same_scope(node, request):
            continue
        if node.state in {"planned", "ready", "blocked"}:
            node.state = "cancelled"
        elif node.state == "running":
            node.state = "cancellation_requested"
    for lease in leases:
        if _same_scope(lease, request) and lease.state == "active":
            lease.state = "revoked"
            lease.epoch += 1

    retained: list[str] = []
    for obligation in obligations:
        if not _same_scope(obligation, request):
            continue
        related = [a for a in attempts if (
            _same_scope(a, request) and a.obligation_id == obligation.obligation_id
        )]
        has_started = any(a.state == "started" for a in related)
        if obligation.state == "pending" or (obligation.state == "claimed" and not has_started):
            obligation.state = "cancelled"
        elif obligation.state in {"claimed", "accepted", "unknown"}:
            obligation.duplicate_possible = True
            retained.append(obligation.obligation_id)
    return CancelResult(
        old_revision, request.lifecycle_revision, old_epoch, request.cancel_epoch,
        tuple(stopped), tuple(retained)
    )


def accept_worker_success(request: Request, *, callback_revision: int, callback_cancel_epoch: int) -> None:
    """Reject late success from a pre-cancel worker epoch."""
    if (
        request.state in {"cancelling", "cancelled"}
        or callback_revision != request.lifecycle_revision
        or callback_cancel_epoch != request.cancel_epoch
    ):
        raise StaleCallback("stale_worker_success")


def cancellation_satisfied(
    request: Request,
    *,
    tasks: Iterable[Task],
    task_runs: Iterable[TaskRun],
    nodes: Iterable[Node],
    leases: Iterable[StageLease],
    obligations: Iterable[DeliveryObligation],
    attempts: Iterable[DeliveryAttempt],
) -> bool:
    """Shared §7.4 predicate for the final cancelling→cancelled CAS."""
    if request.state != "cancelling":
        return False
    # Running/ready work must drain; cancellation_requested is not terminal.
    if any(
        _same_scope(task, request)
        and (
            task.status in {"planned", "ready", "running"}
            or task.cancellation_requested_at is not None and task.status != "cancelled"
        )
        for task in tasks
    ):
        return False
    if any(_same_scope(run, request) and run.ended_at is None for run in task_runs):
        return False
    if any(_same_scope(node, request) and node.state in {
        "planned", "ready", "running", "cancellation_requested", "blocked"
    } for node in nodes):
        return False
    if any(_same_scope(lease, request) and lease.state == "active" for lease in leases):
        return False
    if any(_same_scope(obligation, request) and obligation.state in {
        "pending", "claimed", "accepted", "unknown"
    } for obligation in obligations):
        return False
    if any(_same_scope(attempt, request) and attempt.state == "started" for attempt in attempts):
        return False
    return True


def finish_cancellation(
    request: Request,
    *,
    tasks: Iterable[Task],
    task_runs: Iterable[TaskRun],
    nodes: Iterable[Node],
    leases: Iterable[StageLease],
    obligations: Iterable[DeliveryObligation],
    attempts: Iterable[DeliveryAttempt],
) -> None:
    if not cancellation_satisfied(
        request,
        tasks=tasks,
        task_runs=task_runs,
        nodes=nodes,
        leases=leases,
        obligations=obligations,
        attempts=attempts,
    ):
        raise LifecycleError("cancellation_not_drained")
    request.state = "cancelled"
    request.lifecycle_revision += 1


@dataclass
class AcceptedRun:
    """Immutable record of a lane acceptance (T15)."""
    board: str
    tenant: str
    orch_id: str
    node_key: str
    plan_version: int
    run_id: str
    task_id: str
    outcome_digest: str
    accepted_at: int = 0


@dataclass
class SynthesisResult:
    """Synthesis result bound to a completed attempt (T16)."""
    board: str
    tenant: str
    orch_id: str
    result_digest: str
    synthesis_attempt_id: int
    plan_version: int


def accept_lane_run(
    request: Request,
    node: Node,
    *,
    plan_version: int,
    run_id: str,
    task_id: str,
    outcome_digest: str,
    callback_revision: int,
    callback_cancel_epoch: int,
    now: int = 0,
) -> AcceptedRun:
    """T15: Accept a lane run bound to current plan version and required lane.

    Rejects:
    - stale callback (wrong revision/cancel_epoch)
    - wrong state (not waiting_lanes or synthesizing)
    - stale plan version
    - accepting an already-accepted required node
    """
    accept_worker_success(
        request,
        callback_revision=callback_revision,
        callback_cancel_epoch=callback_cancel_epoch,
    )
    if request.state not in ("waiting_lanes", "synthesizing"):
        raise LifecycleError("accept_not_authorized")
    if node.plan_version != plan_version:
        raise LifecycleError("stale_plan_version")
    if node.state == "accepted":
        raise LifecycleError("node_already_accepted")
    node.state = "accepted"
    return AcceptedRun(
        board=request.board,
        tenant=request.tenant,
        orch_id=request.orch_id,
        node_key=node.node_key,
        plan_version=plan_version,
        run_id=run_id,
        task_id=task_id,
        outcome_digest=outcome_digest,
        accepted_at=now,
    )


def validate_result_provenance(
    result: SynthesisResult,
    *,
    completed_attempt_id: int,
    current_plan_version: int,
) -> None:
    """T16: Synthesis result must be bound to a completed synthesis attempt."""
    if result.synthesis_attempt_id != completed_attempt_id:
        raise LifecycleError("result_producer_unbound")
    if result.plan_version != current_plan_version:
        raise LifecycleError("stale_plan_version")


def supersede_request(
    predecessor: Request,
    *,
    new_orch_id: str,
) -> Request:
    """Create a superseding request: generation+1, new orch_id, fresh state.

    Predecessor must be terminal (failed or cancelled).
    """
    if predecessor.state not in ("failed", "cancelled"):
        raise LifecycleError("supersession_predecessor_not_terminal")
    return Request(
        board=predecessor.board,
        tenant=predecessor.tenant,
        orch_id=new_orch_id,
        state="submitted",
        lifecycle_revision=0,
        cancel_epoch=0,
        generation=predecessor.generation + 1,
    )


def is_terminal(state: str) -> bool:
    """Check if a lifecycle state is terminal."""
    return state in TERMINAL_STATES


__all__ = [
    "TERMINAL_STATES", "NONTERMINAL_STATES", "TRANSITION_TABLE", "LifecycleError",
    "StaleCallback", "Request", "Node", "Task", "TaskRun", "StageLease",
    "DeliveryObligation", "DeliveryAttempt", "CancelResult", "AcceptedRun",
    "SynthesisResult", "transition_allowed",
    "apply_transition", "request_optional_cancellation", "optional_cancel_drained",
    "complete_cancelled_node", "cancel_cascade", "accept_worker_success",
    "cancellation_satisfied", "finish_cancellation", "accept_lane_run",
    "validate_result_provenance", "supersede_request", "is_terminal",
]

"""Deterministic, bounded routing for one project repair action.

The router is deliberately separated from Kanban task creation.  Its injected
``register_repair`` boundary must perform one atomic compare-and-set operation:
validate the supplied project version/lock token, create (or find) the task by
``RepairAction.idempotency_key``, inherit the supplied contract and notification
route identities, register required ``repair`` membership, and advance project
repair state.  This module never executes a worker or rewrites failed history.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Mapping

from hermes_cli.project_failure_envelope import (
    FAILURE_CLASSES,
    ProjectFailureEnvelope,
)
from hermes_cli.project_finalizer import EVALUATION_STATES, ProjectEvaluation


NO_ACTION = "NO_ACTION"
REPAIR_CREATED = "REPAIR_CREATED"
REPAIR_ALREADY_EXISTS = "REPAIR_ALREADY_EXISTS"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
UNSUPPORTED_FAILURE = "UNSUPPORTED_FAILURE"
STALE_SNAPSHOT = "STALE_SNAPSHOT"
MALFORMED = "MALFORMED"

ROUTE_OUTCOMES: tuple[str, ...] = (
    NO_ACTION,
    REPAIR_CREATED,
    REPAIR_ALREADY_EXISTS,
    BUDGET_EXHAUSTED,
    UNSUPPORTED_FAILURE,
    STALE_SNAPSHOT,
    MALFORMED,
)

REGISTRATION_CREATED = "created"
REGISTRATION_ALREADY_EXISTS = "already_exists"
REGISTRATION_STALE_SNAPSHOT = "stale_snapshot"
_REGISTRATION_DISPOSITIONS = frozenset(
    {
        REGISTRATION_CREATED,
        REGISTRATION_ALREADY_EXISTS,
        REGISTRATION_STALE_SNAPSHOT,
    }
)

# Only bounded failures that a builder can act on without credentials, billing,
# delivery, or external human intervention are eligible.  REPAIRABLE remains a
# necessary evaluator state; this allow-list is an additional safety boundary.
REPAIRABLE_FAILURE_CLASSES: tuple[str, ...] = (
    "provider_rate_limit",
    "provider_timeout",
    "process_crash",
    "task_timeout",
    "iteration_budget",
    "protocol_violation",
    "artifact_failure",
)
_REPAIRABLE_FAILURE_CLASS_SET = frozenset(REPAIRABLE_FAILURE_CLASSES)
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_NON_REPAIRABLE_EVALUATION_STATES = frozenset(
    {"WAITING", "COMPLETE_ELIGIBLE", "BLOCKED", "FAILED", "MALFORMED"}
)


@dataclass(frozen=True)
class ProjectIdentity:
    """Stable project identity for one finalization generation."""

    project_id: str
    board_id: str
    root_task_id: str
    generation: int


@dataclass(frozen=True)
class ProjectVersionToken:
    """Caller-owned snapshot plus durable project CAS/lock identity."""

    snapshot_version: str
    project_version: int
    lock_token: str


@dataclass(frozen=True)
class RepairMembership:
    """Existing required repair membership supplied from one project snapshot."""

    project: ProjectIdentity
    repair_identity: str
    repair_task_id: str
    failed_task_id: str
    failed_run_id: int | None
    failure_fingerprint: str
    repair_index: int


@dataclass(frozen=True)
class ProjectRepairRequest:
    """All value inputs required to make one bounded routing decision."""

    project: ProjectIdentity
    evaluation: ProjectEvaluation
    failed_task_id: str
    failed_run_id: int | None
    failure_envelope: ProjectFailureEnvelope
    project_repair_budget: int
    task_retry_limit: int
    existing_repairs: tuple[RepairMembership, ...]
    worker_profile: str
    allowed_worker_profiles: tuple[str, ...]
    task_contract: Mapping[str, Any]
    notification_route_identities: tuple[str, ...]
    version_token: ProjectVersionToken


@dataclass(frozen=True)
class RepairAction:
    """Mutation intent consumed by the injected atomic registration boundary."""

    project: ProjectIdentity
    repair_identity: str
    idempotency_key: str
    failed_task_id: str
    failed_run_id: int | None
    failure_fingerprint: str
    repair_index: int
    task_retry_index: int
    worker_profile: str
    task_contract: Mapping[str, Any]
    notification_route_identities: tuple[str, ...]
    membership_kind: str = "repair"
    required: bool = True


@dataclass(frozen=True)
class AtomicRepairRegistration:
    """Stable response from the injected atomic registration boundary."""

    disposition: str
    repair_task_id: str | None = None


@dataclass(frozen=True)
class RepairRouteResult:
    """Stable, non-secret route result."""

    outcome: str
    reason: str
    repair_identity: str | None = None
    repair_task_id: str | None = None


RegisterRepair = Callable[
    [RepairAction, ProjectVersionToken], AtomicRepairRegistration
]


def route_project_repair(
    request: ProjectRepairRequest,
    *,
    register_repair: RegisterRepair,
) -> RepairRouteResult:
    """Route at most one repair for a stable failure identity.

    Refusal precedence is deterministic: non-REPAIRABLE evaluator states,
    malformed identity/envelope inputs, existing repair membership, unsupported
    failure class, stale snapshot, profile permission, project budget, task retry
    limit, then the injected atomic compare-and-set boundary.
    """
    if not isinstance(request, ProjectRepairRequest):
        return _result(MALFORMED, "request_type_invalid")
    if not isinstance(request.evaluation, ProjectEvaluation):
        return _result(MALFORMED, "evaluation_type_invalid")

    evaluation_state = request.evaluation.evaluation_state
    if evaluation_state in _NON_REPAIRABLE_EVALUATION_STATES:
        return _result(NO_ACTION, "evaluation_state_not_repairable")
    if evaluation_state not in EVALUATION_STATES:
        return _result(MALFORMED, "evaluation_state_invalid")
    if evaluation_state != "REPAIRABLE":
        return _result(NO_ACTION, "evaluation_state_not_repairable")

    malformed_reason = _validate_repairable_request(request)
    if malformed_reason is not None:
        return _result(MALFORMED, malformed_reason)

    repair_index = request.evaluation.repair_generation + 1
    repair_identity = _repair_identity(request, repair_index=repair_index)

    matching_membership = next(
        (
            member
            for member in request.existing_repairs
            if member.project == request.project
            and member.repair_identity == repair_identity
        ),
        None,
    )
    if matching_membership is not None:
        return _result(
            REPAIR_ALREADY_EXISTS,
            "repair_identity_already_registered",
            repair_identity=repair_identity,
            repair_task_id=matching_membership.repair_task_id,
        )

    failure_class = request.failure_envelope.failure_class
    if failure_class not in _REPAIRABLE_FAILURE_CLASS_SET:
        return _result(
            UNSUPPORTED_FAILURE,
            "failure_class_not_repairable",
            repair_identity=repair_identity,
        )

    if request.version_token.snapshot_version != request.evaluation.snapshot_version:
        return _result(
            STALE_SNAPSHOT,
            "evaluation_snapshot_version_mismatch",
            repair_identity=repair_identity,
        )
    if request.project_repair_budget != request.evaluation.repair_budget:
        return _result(
            STALE_SNAPSHOT,
            "project_repair_budget_changed",
            repair_identity=repair_identity,
        )

    if request.worker_profile not in request.allowed_worker_profiles:
        return _result(
            MALFORMED,
            "worker_profile_not_permitted",
            repair_identity=repair_identity,
        )

    if request.evaluation.repair_generation >= request.project_repair_budget:
        return _result(
            BUDGET_EXHAUSTED,
            "project_repair_budget_exhausted",
            repair_identity=repair_identity,
        )

    same_project_repairs = tuple(
        member
        for member in request.existing_repairs
        if member.project == request.project
    )
    if any(
        member.repair_index > request.evaluation.repair_generation
        for member in same_project_repairs
    ):
        return _result(
            STALE_SNAPSHOT,
            "repair_membership_newer_than_evaluation",
            repair_identity=repair_identity,
        )

    failed_task_repair_count = sum(
        member.failed_task_id == request.failed_task_id
        for member in same_project_repairs
    )
    if failed_task_repair_count >= request.task_retry_limit:
        return _result(
            BUDGET_EXHAUSTED,
            "failed_task_retry_limit_exhausted",
            repair_identity=repair_identity,
        )

    action = RepairAction(
        project=request.project,
        repair_identity=repair_identity,
        idempotency_key=repair_identity,
        failed_task_id=request.failed_task_id,
        failed_run_id=request.failed_run_id,
        failure_fingerprint=request.failure_envelope.error_fingerprint or "",
        repair_index=repair_index,
        task_retry_index=failed_task_repair_count + 1,
        worker_profile=request.worker_profile,
        task_contract=copy.deepcopy(dict(request.task_contract)),
        notification_route_identities=tuple(request.notification_route_identities),
    )
    registration = register_repair(action, request.version_token)
    return _registration_result(registration, repair_identity=repair_identity)


def _validate_repairable_request(request: ProjectRepairRequest) -> str | None:
    project = request.project
    if not isinstance(project, ProjectIdentity):
        return "project_identity_type_invalid"
    for name, value in (
        ("project_id", project.project_id),
        ("board_id", project.board_id),
        ("root_task_id", project.root_task_id),
        ("failed_task_id", request.failed_task_id),
        ("worker_profile", request.worker_profile),
    ):
        if not _is_text(value):
            return f"{name}_invalid"
    if not _is_nonnegative_int(project.generation) or project.generation < 1:
        return "project_generation_invalid"

    evaluation = request.evaluation
    if (
        evaluation.board_id != project.board_id
        or evaluation.root_task_id != project.root_task_id
        or evaluation.generation != project.generation
    ):
        return "evaluation_project_identity_mismatch"
    if (
        evaluation.repair_eligible is not True
        or evaluation.finalization_eligible is not False
        or evaluation.terminal_outcome is not None
        or evaluation.checker_verdict != "FAIL_REPAIRABLE"
    ):
        return "repairable_evaluation_inconsistent"
    if not _is_nonnegative_int(evaluation.repair_generation):
        return "evaluation_repair_generation_invalid"
    if not _is_nonnegative_int(evaluation.repair_budget):
        return "evaluation_repair_budget_invalid"
    if not _is_text(evaluation.snapshot_version):
        return "evaluation_snapshot_version_invalid"

    envelope = request.failure_envelope
    if not isinstance(envelope, ProjectFailureEnvelope):
        return "failure_envelope_type_invalid"
    if (
        envelope.board_id != project.board_id
        or envelope.root_task_id != project.root_task_id
        or envelope.generation != project.generation
        or envelope.task_id != request.failed_task_id
    ):
        return "failure_envelope_identity_mismatch"
    if request.failed_run_id is not None and not _is_nonnegative_int(request.failed_run_id):
        return "failed_run_id_invalid"
    if envelope.run_id != request.failed_run_id:
        return "failure_envelope_run_mismatch"
    if not isinstance(envelope.failure_class, str) or envelope.failure_class not in FAILURE_CLASSES:
        return "failure_class_invalid"
    if not isinstance(envelope.error_fingerprint, str) or not _SHA256_RE.fullmatch(
        envelope.error_fingerprint
    ):
        return "failure_fingerprint_invalid"

    if not _is_nonnegative_int(request.project_repair_budget):
        return "project_repair_budget_invalid"
    if not _is_nonnegative_int(request.task_retry_limit):
        return "task_retry_limit_invalid"

    if not isinstance(request.existing_repairs, tuple):
        return "existing_repairs_invalid"
    seen_identities: set[tuple[ProjectIdentity, str]] = set()
    for member in request.existing_repairs:
        if not isinstance(member, RepairMembership):
            return "existing_repair_membership_invalid"
        if (
            not _is_text(member.repair_identity)
            or not _is_text(member.repair_task_id)
            or not _is_text(member.failed_task_id)
            or not isinstance(member.project, ProjectIdentity)
            or not _is_nonnegative_int(member.repair_index)
            or member.repair_index < 1
        ):
            return "existing_repair_membership_invalid"
        if member.failed_run_id is not None and not _is_nonnegative_int(member.failed_run_id):
            return "existing_repair_membership_invalid"
        if not _SHA256_RE.fullmatch(member.failure_fingerprint):
            return "existing_repair_membership_invalid"
        membership_key = (member.project, member.repair_identity)
        if membership_key in seen_identities:
            return "duplicate_existing_repair_identity"
        seen_identities.add(membership_key)

    if not isinstance(request.allowed_worker_profiles, tuple) or not request.allowed_worker_profiles:
        return "allowed_worker_profiles_invalid"
    if any(not _is_text(profile) for profile in request.allowed_worker_profiles):
        return "allowed_worker_profiles_invalid"
    if len(set(request.allowed_worker_profiles)) != len(request.allowed_worker_profiles):
        return "allowed_worker_profiles_invalid"

    if not isinstance(request.task_contract, Mapping) or not request.task_contract:
        return "task_contract_invalid"
    try:
        copy.deepcopy(dict(request.task_contract))
    except Exception:
        return "task_contract_invalid"

    routes = request.notification_route_identities
    if not isinstance(routes, tuple) or not routes:
        return "notification_route_identity_missing"
    if any(not _is_text(route) for route in routes) or len(set(routes)) != len(routes):
        return "notification_route_identity_invalid"

    token = request.version_token
    if not isinstance(token, ProjectVersionToken):
        return "project_version_token_type_invalid"
    if (
        not _is_text(token.snapshot_version)
        or not _is_nonnegative_int(token.project_version)
        or token.project_version < 1
        or not _is_text(token.lock_token)
    ):
        return "project_version_token_invalid"
    return None


def _repair_identity(
    request: ProjectRepairRequest,
    *,
    repair_index: int,
) -> str:
    if request.failed_run_id is not None:
        occurrence: dict[str, object] = {"run_id": request.failed_run_id}
    else:
        occurrence = {
            "failure_fingerprint": request.failure_envelope.error_fingerprint
        }
    payload = {
        "project": {
            "project_id": request.project.project_id,
            "board_id": request.project.board_id,
            "root_task_id": request.project.root_task_id,
            "generation": request.project.generation,
        },
        "failed_task_id": request.failed_task_id,
        "failure_occurrence": occurrence,
        "repair_index": repair_index,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"repair:sha256:{digest}"


def _registration_result(
    registration: AtomicRepairRegistration,
    *,
    repair_identity: str,
) -> RepairRouteResult:
    if not isinstance(registration, AtomicRepairRegistration):
        return _result(
            MALFORMED,
            "atomic_registration_result_invalid",
            repair_identity=repair_identity,
        )
    if registration.disposition not in _REGISTRATION_DISPOSITIONS:
        return _result(
            MALFORMED,
            "atomic_registration_disposition_invalid",
            repair_identity=repair_identity,
        )
    if registration.disposition == REGISTRATION_STALE_SNAPSHOT:
        return _result(
            STALE_SNAPSHOT,
            "atomic_registration_rejected_stale_token",
            repair_identity=repair_identity,
        )
    if not _is_text(registration.repair_task_id):
        return _result(
            MALFORMED,
            "atomic_registration_task_identity_invalid",
            repair_identity=repair_identity,
        )
    if registration.disposition == REGISTRATION_ALREADY_EXISTS:
        return _result(
            REPAIR_ALREADY_EXISTS,
            "atomic_registration_found_existing_repair",
            repair_identity=repair_identity,
            repair_task_id=registration.repair_task_id,
        )
    return _result(
        REPAIR_CREATED,
        "eligible_repair_registered",
        repair_identity=repair_identity,
        repair_task_id=registration.repair_task_id,
    )


def _result(
    outcome: str,
    reason: str,
    *,
    repair_identity: str | None = None,
    repair_task_id: str | None = None,
) -> RepairRouteResult:
    return RepairRouteResult(
        outcome=outcome,
        reason=reason,
        repair_identity=repair_identity,
        repair_task_id=repair_task_id,
    )


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and "\x00" not in value


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


__all__ = [
    "AtomicRepairRegistration",
    "BUDGET_EXHAUSTED",
    "MALFORMED",
    "NO_ACTION",
    "ProjectIdentity",
    "ProjectRepairRequest",
    "ProjectVersionToken",
    "REPAIR_ALREADY_EXISTS",
    "REPAIR_CREATED",
    "REPAIRABLE_FAILURE_CLASSES",
    "REGISTRATION_ALREADY_EXISTS",
    "REGISTRATION_CREATED",
    "REGISTRATION_STALE_SNAPSHOT",
    "ROUTE_OUTCOMES",
    "RepairAction",
    "RepairMembership",
    "RepairRouteResult",
    "STALE_SNAPSHOT",
    "UNSUPPORTED_FAILURE",
    "route_project_repair",
]

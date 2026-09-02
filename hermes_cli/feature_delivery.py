"""Deterministic contracts and guards for Feature Delivery V1.

This module defines validation only.  It does not run agents, mutate Git, or
perform delivery.  ``DELIVERED`` means the feature delivery gate passed; it
does not mean merged, pushed, deployed, or released.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Annotated, Iterable, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


FEATURE_DELIVERY_WORKFLOW = "feature_delivery_v1"
ACCEPTANCE_FINAL_MARKER = "FINAL: ACCEPT"
MAX_FIX_LOOPS = 5

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
FullCommitSha = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{40}$"),
]


class FeatureDeliveryState(str, Enum):
    NEW = "NEW"
    CONTRACT_READY = "CONTRACT_READY"
    DEVELOPING = "DEVELOPING"
    READY_FOR_TEST = "READY_FOR_TEST"
    TESTING = "TESTING"
    TEST_FAILED = "TEST_FAILED"
    TEST_PASSED = "TEST_PASSED"
    ACCEPTANCE = "ACCEPTANCE"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    DELIVERED = "DELIVERED"


_BLOCKABLE_STATES = {
    FeatureDeliveryState.CONTRACT_READY,
    FeatureDeliveryState.DEVELOPING,
    FeatureDeliveryState.READY_FOR_TEST,
    FeatureDeliveryState.TESTING,
    FeatureDeliveryState.TEST_FAILED,
    FeatureDeliveryState.TEST_PASSED,
    FeatureDeliveryState.ACCEPTANCE,
    FeatureDeliveryState.REJECTED,
}

LEGAL_TRANSITIONS: dict[FeatureDeliveryState, frozenset[FeatureDeliveryState]] = {
    FeatureDeliveryState.NEW: frozenset({FeatureDeliveryState.CONTRACT_READY}),
    FeatureDeliveryState.CONTRACT_READY: frozenset(
        {FeatureDeliveryState.DEVELOPING, FeatureDeliveryState.BLOCKED}
    ),
    FeatureDeliveryState.DEVELOPING: frozenset(
        {FeatureDeliveryState.READY_FOR_TEST, FeatureDeliveryState.BLOCKED}
    ),
    FeatureDeliveryState.READY_FOR_TEST: frozenset(
        {FeatureDeliveryState.TESTING, FeatureDeliveryState.BLOCKED}
    ),
    FeatureDeliveryState.TESTING: frozenset(
        {
            FeatureDeliveryState.TEST_FAILED,
            FeatureDeliveryState.TEST_PASSED,
            FeatureDeliveryState.BLOCKED,
        }
    ),
    FeatureDeliveryState.TEST_FAILED: frozenset(
        {FeatureDeliveryState.DEVELOPING, FeatureDeliveryState.BLOCKED}
    ),
    FeatureDeliveryState.TEST_PASSED: frozenset(
        {FeatureDeliveryState.ACCEPTANCE, FeatureDeliveryState.BLOCKED}
    ),
    FeatureDeliveryState.ACCEPTANCE: frozenset(
        {
            FeatureDeliveryState.REJECTED,
            FeatureDeliveryState.BLOCKED,
            FeatureDeliveryState.DELIVERED,
        }
    ),
    FeatureDeliveryState.REJECTED: frozenset(
        {FeatureDeliveryState.DEVELOPING, FeatureDeliveryState.BLOCKED}
    ),
    FeatureDeliveryState.BLOCKED: frozenset(),
    FeatureDeliveryState.DELIVERED: frozenset(),
}


def is_legal_transition(
    current: FeatureDeliveryState,
    target: FeatureDeliveryState,
) -> bool:
    """Return whether an explicit Feature Delivery transition is permitted."""

    return target in LEGAL_TRANSITIONS[current]


def can_transition_to_blocked(state: FeatureDeliveryState) -> bool:
    """Return whether infrastructure failure may terminally block ``state``."""

    return state in _BLOCKABLE_STATES


def count_fix_loops(
    transitions: Iterable[tuple[FeatureDeliveryState, FeatureDeliveryState]],
) -> int:
    """Count only test- or acceptance-driven returns to development."""

    fix_transitions = {
        (FeatureDeliveryState.TEST_FAILED, FeatureDeliveryState.DEVELOPING),
        (FeatureDeliveryState.REJECTED, FeatureDeliveryState.DEVELOPING),
    }
    return sum(transition in fix_transitions for transition in transitions)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AcceptanceCriterion(FrozenModel):
    id: NonEmptyStr
    requirement: NonEmptyStr


class TaskContract(FrozenModel):
    task_id: NonEmptyStr
    title: NonEmptyStr
    objective: NonEmptyStr
    repository: NonEmptyStr
    base_commit: FullCommitSha
    branch: NonEmptyStr
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = Field(min_length=1)
    constraints: tuple[NonEmptyStr, ...] = ()
    required_tests: tuple[NonEmptyStr, ...] = Field(min_length=1)
    required_evidence: tuple[NonEmptyStr, ...] = Field(min_length=1)
    out_of_scope: tuple[NonEmptyStr, ...] = ()
    delivery_gate: Literal["acceptance_agent"] = "acceptance_agent"

    @model_validator(mode="after")
    def acceptance_criterion_ids_are_unique(self) -> Self:
        ids = [criterion.id for criterion in self.acceptance_criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("acceptance criterion ids must be unique")
        return self


def canonicalize_contract(contract: TaskContract) -> bytes:
    """Return stable, compact, UTF-8 canonical JSON for ``contract``."""

    return json.dumps(
        contract.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def compute_contract_hash(contract: TaskContract) -> str:
    return hashlib.sha256(canonicalize_contract(contract)).hexdigest()


class DeveloperReportStatus(str, Enum):
    READY_FOR_TEST = "READY_FOR_TEST"
    BLOCKED = "BLOCKED"


class TesterReportStatus(str, Enum):
    TEST_PASS = "TEST_PASS"
    TEST_FAIL = "TEST_FAIL"
    BLOCKED = "BLOCKED"


class AcceptanceReportStatus(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    BLOCKED = "BLOCKED"


class DeveloperReport(FrozenModel):
    task_id: NonEmptyStr
    agent: Literal["developer"]
    status: DeveloperReportStatus
    commit: FullCommitSha | None = None
    changed_files: tuple[NonEmptyStr, ...] = ()
    implementation_summary: NonEmptyStr
    self_checks: tuple[NonEmptyStr, ...] = ()
    known_risks: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def ready_report_has_commit(self) -> Self:
        if self.status == DeveloperReportStatus.READY_FOR_TEST and self.commit is None:
            raise ValueError("READY_FOR_TEST requires a full commit SHA")
        return self


class TesterReport(FrozenModel):
    task_id: NonEmptyStr
    agent: Literal["tester"]
    tested_commit: FullCommitSha | None = None
    status: TesterReportStatus
    test_results: tuple[NonEmptyStr, ...] = ()
    blocking_issues: tuple[NonEmptyStr, ...] = ()
    non_blocking_issues: tuple[NonEmptyStr, ...] = ()
    evidence: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def completed_test_has_commit(self) -> Self:
        if self.status in {
            TesterReportStatus.TEST_PASS,
            TesterReportStatus.TEST_FAIL,
        } and self.tested_commit is None:
            raise ValueError("TEST_PASS and TEST_FAIL require a full commit SHA")
        return self


class AcceptanceCriterionResult(FrozenModel):
    id: NonEmptyStr
    met: bool
    evidence: NonEmptyStr


class AcceptanceReport(FrozenModel):
    task_id: NonEmptyStr
    agent: Literal["acceptance"]
    accepted_commit: FullCommitSha | None = None
    status: AcceptanceReportStatus
    criteria: tuple[AcceptanceCriterionResult, ...] = ()
    blocking_issues: tuple[NonEmptyStr, ...] = ()
    evidence: tuple[NonEmptyStr, ...] = ()
    final_marker: str | None = None

    @model_validator(mode="after")
    def validate_acceptance_shape(self) -> Self:
        ids = [criterion.id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("acceptance criterion ids must be unique")
        if self.status in {
            AcceptanceReportStatus.ACCEPT,
            AcceptanceReportStatus.REJECT,
        } and self.accepted_commit is None:
            raise ValueError("ACCEPT and REJECT require a full commit SHA")
        if self.status == AcceptanceReportStatus.ACCEPT:
            if self.final_marker != ACCEPTANCE_FINAL_MARKER:
                raise ValueError("ACCEPT requires exact final marker FINAL: ACCEPT")
        elif self.final_marker == ACCEPTANCE_FINAL_MARKER:
            raise ValueError("REJECT or BLOCKED cannot use FINAL: ACCEPT")
        return self


StageRole = Literal["developer", "tester", "acceptance"]
StageReport = DeveloperReport | TesterReport | AcceptanceReport


def validate_stage_report(expected_role: StageRole, report: object) -> bool:
    """Validate report ownership by concrete type and exact agent identifier."""

    expected = {
        "developer": (DeveloperReport, "developer"),
        "tester": (TesterReport, "tester"),
        "acceptance": (AcceptanceReport, "acceptance"),
    }[expected_role]
    return isinstance(report, expected[0]) and report.agent == expected[1]


class DeliveryCommitContext(FrozenModel):
    developer_commit: FullCommitSha
    tested_commit: FullCommitSha | None = None
    accepted_commit: FullCommitSha | None = None
    branch_head: FullCommitSha


class DeliveryGateResult(FrozenModel):
    allowed: bool
    reasons: tuple[str, ...] = ()


def validate_delivery_commit_integrity(
    context: DeliveryCommitContext,
) -> DeliveryGateResult:
    """Require one exact commit to be developed, tested, accepted, and at HEAD."""

    reasons: list[str] = []
    if context.tested_commit is None:
        reasons.append("tested commit is missing")
    elif context.tested_commit != context.developer_commit:
        reasons.append("tested commit does not match developer commit")
    if context.accepted_commit is None:
        reasons.append("accepted commit is missing")
    elif context.accepted_commit != context.tested_commit:
        reasons.append("accepted commit does not match tested commit")
    if context.branch_head != context.accepted_commit:
        reasons.append("branch HEAD does not match accepted commit")
    return DeliveryGateResult(allowed=not reasons, reasons=tuple(reasons))


def invalidate_downstream_evidence_on_new_developer_commit(
    context: DeliveryCommitContext,
    new_developer_commit: FullCommitSha,
) -> DeliveryCommitContext:
    """Clear test and acceptance evidence after developer code changes."""

    if new_developer_commit == context.developer_commit:
        return context
    return DeliveryCommitContext(
        developer_commit=new_developer_commit,
        tested_commit=None,
        accepted_commit=None,
        branch_head=new_developer_commit,
    )


def _acceptance_criteria_reasons(
    contract: TaskContract,
    report: AcceptanceReport,
) -> list[str]:
    required_ids = {criterion.id for criterion in contract.acceptance_criteria}
    reported_ids = [criterion.id for criterion in report.criteria]
    reasons: list[str] = []
    if len(reported_ids) != len(set(reported_ids)):
        reasons.append("acceptance report contains duplicate criterion ids")
    missing = sorted(required_ids - set(reported_ids))
    unknown = sorted(set(reported_ids) - required_ids)
    if missing:
        reasons.append(f"missing acceptance criteria: {', '.join(missing)}")
    if unknown:
        reasons.append(f"unknown acceptance criteria: {', '.join(unknown)}")
    unmet = sorted(
        criterion.id
        for criterion in report.criteria
        if criterion.id in required_ids and not criterion.met
    )
    if unmet:
        reasons.append(f"unmet acceptance criteria: {', '.join(unmet)}")
    return reasons


def evaluate_delivery_gate(
    contract: TaskContract,
    acceptance_report: object,
    commit_context: DeliveryCommitContext,
    *,
    workflow_template_id: str,
    current_state: FeatureDeliveryState,
    expected_contract_hash: str,
    stage_evidence: Iterable[str] = (),
) -> DeliveryGateResult:
    """Return explicit reasons unless the acceptance-only gate fully passes."""

    reasons: list[str] = []
    if workflow_template_id != FEATURE_DELIVERY_WORKFLOW:
        reasons.append("workflow is not feature_delivery_v1")
    if current_state != FeatureDeliveryState.ACCEPTANCE:
        reasons.append("current state is not ACCEPTANCE")
    if not validate_stage_report("acceptance", acceptance_report):
        reasons.append("report is not a valid acceptance report")
        return DeliveryGateResult(allowed=False, reasons=tuple(reasons))

    report = acceptance_report
    if report.task_id != contract.task_id:
        reasons.append("acceptance report task does not match contract")
    if report.status != AcceptanceReportStatus.ACCEPT:
        reasons.append("acceptance status is not ACCEPT")
    if report.final_marker != ACCEPTANCE_FINAL_MARKER:
        reasons.append("acceptance final marker is invalid")
    if compute_contract_hash(contract) != expected_contract_hash:
        reasons.append("contract hash changed")
    reasons.extend(_acceptance_criteria_reasons(contract, report))

    present_evidence = set(report.evidence) | set(stage_evidence)
    missing_evidence = sorted(set(contract.required_evidence) - present_evidence)
    if missing_evidence:
        reasons.append(f"missing required evidence: {', '.join(missing_evidence)}")

    commit_result = validate_delivery_commit_integrity(commit_context)
    reasons.extend(commit_result.reasons)
    if report.accepted_commit != commit_context.accepted_commit:
        reasons.append("acceptance report commit does not match commit context")

    return DeliveryGateResult(allowed=not reasons, reasons=tuple(reasons))

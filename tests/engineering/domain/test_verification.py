from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

import engineering.domain.verification as verification_module
from engineering.domain.verification import (
    InvalidVerificationResult,
    VerificationCheckKind,
    VerificationCheckResult,
    VerificationCheckStatus,
    VerificationResult,
    VerificationVerdict,
)


STARTED_AT = datetime(2026, 8, 15, 8, 0, tzinfo=timezone.utc)
FINISHED_AT = STARTED_AT + timedelta(seconds=3, milliseconds=125)


def make_check(**overrides: object) -> VerificationCheckResult:
    values: dict[str, object] = {
        "kind": VerificationCheckKind.TEST,
        "status": VerificationCheckStatus.PASS,
        "required": True,
        "summary": "Focused tests passed.",
        "evidence_ids": ("evidence-1",),
    }
    values.update(overrides)
    return VerificationCheckResult(**values)  # type: ignore[arg-type]


def make_result(**overrides: object) -> VerificationResult:
    values: dict[str, object] = {
        "workflow_run_id": "workflow-123",
        "attempt": 1,
        "checks": (make_check(),),
        "started_at": STARTED_AT,
        "finished_at": FINISHED_AT,
    }
    values.update(overrides)
    return VerificationResult(**values)  # type: ignore[arg-type]


def test_declares_check_kinds_statuses_and_verdicts() -> None:
    assert [kind.value for kind in VerificationCheckKind] == [
        "BUILD",
        "TEST",
        "LINT",
        "GIT_DIFF",
        "POLICY",
        "CUSTOM",
    ]
    assert [status.value for status in VerificationCheckStatus] == [
        "PASS",
        "FAIL",
        "ERROR",
        "SKIPPED",
    ]
    assert [verdict.value for verdict in VerificationVerdict] == [
        "PASS",
        "FAIL",
        "ERROR",
    ]


def test_generates_uuid4_identities_by_default() -> None:
    first_check = make_check()
    second_check = make_check()
    first_result = make_result(checks=(first_check,))
    second_result = make_result(checks=(second_check,))

    assert UUID(first_check.check_id).version == 4
    assert UUID(first_result.verification_id).version == 4
    assert first_check.check_id != second_check.check_id
    assert first_result.verification_id != second_result.verification_id


def test_preserves_explicit_ids_and_workflow_lineage() -> None:
    check = make_check(check_id="check-imported-1")
    result = make_result(
        verification_id="verification-imported-1",
        workflow_run_id="workflow-99",
        attempt=3,
        checks=(check,),
    )

    assert check.check_id == "check-imported-1"
    assert result.verification_id == "verification-imported-1"
    assert result.workflow_run_id == "workflow-99"
    assert result.attempt == 3


@pytest.mark.parametrize("attempt", [0, -1, True, 1.5, "1"])
def test_attempt_must_be_a_positive_integer(attempt: object) -> None:
    with pytest.raises(
        ValueError, match="attempt must be an integer greater than zero"
    ):
        make_result(attempt=attempt)


def test_accepts_aware_timestamps_and_calculates_duration() -> None:
    offset = timezone(timedelta(hours=8))
    started_at = datetime(2026, 8, 15, 16, 0, tzinfo=offset)
    finished_at = started_at + timedelta(seconds=9, microseconds=75)

    result = make_result(started_at=started_at, finished_at=finished_at)

    assert result.started_at == started_at
    assert result.finished_at == finished_at
    assert result.duration == timedelta(seconds=9, microseconds=75)


@pytest.mark.parametrize("field_name", ["started_at", "finished_at"])
def test_rejects_naive_timestamps(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be timezone-aware"):
        make_result(**{field_name: datetime(2026, 8, 15, 8, 0)})


def test_rejects_finished_at_before_started_at() -> None:
    with pytest.raises(
        ValueError, match="finished_at cannot be earlier than started_at"
    ):
        make_result(finished_at=STARTED_AT - timedelta(microseconds=1))


def test_checks_and_evidence_references_are_immutable_tuples() -> None:
    check = make_check(evidence_ids=["evidence-1", "evidence-2"])
    result = make_result(checks=[check])

    assert check.evidence_ids == ("evidence-1", "evidence-2")
    assert result.checks == (check,)
    with pytest.raises(FrozenInstanceError):
        check.summary = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        check.evidence_ids[0] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.attempt = 2  # type: ignore[misc]


def test_check_can_have_zero_evidence_references() -> None:
    check = make_check(evidence_ids=())

    assert check.evidence_ids == ()


def test_empty_check_collection_is_rejected() -> None:
    with pytest.raises(
        InvalidVerificationResult,
        match="must contain at least one check",
    ):
        make_result(checks=())


def test_all_required_checks_passing_produces_pass() -> None:
    result = make_result(
        checks=(
            make_check(kind=VerificationCheckKind.BUILD),
            make_check(kind=VerificationCheckKind.TEST),
            make_check(kind=VerificationCheckKind.LINT),
        )
    )

    assert result.verdict is VerificationVerdict.PASS


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (VerificationCheckStatus.FAIL, VerificationVerdict.FAIL),
        (VerificationCheckStatus.ERROR, VerificationVerdict.ERROR),
        (VerificationCheckStatus.SKIPPED, VerificationVerdict.FAIL),
    ],
)
def test_required_non_pass_statuses_determine_verdict(
    status: VerificationCheckStatus,
    expected: VerificationVerdict,
) -> None:
    result = make_result(checks=(make_check(status=status, required=True),))

    assert result.verdict is expected


def test_optional_skipped_is_allowed() -> None:
    result = make_result(
        checks=(
            make_check(required=True, status=VerificationCheckStatus.PASS),
            make_check(required=False, status=VerificationCheckStatus.SKIPPED),
        )
    )

    assert result.verdict is VerificationVerdict.PASS


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (VerificationCheckStatus.PASS, VerificationVerdict.PASS),
        (VerificationCheckStatus.FAIL, VerificationVerdict.FAIL),
        (VerificationCheckStatus.ERROR, VerificationVerdict.ERROR),
    ],
)
def test_optional_check_statuses_determine_verdict(
    status: VerificationCheckStatus,
    expected: VerificationVerdict,
) -> None:
    result = make_result(checks=(make_check(status=status, required=False),))

    assert result.verdict is expected


def test_error_has_precedence_over_fail() -> None:
    result = make_result(
        checks=(
            make_check(status=VerificationCheckStatus.FAIL),
            make_check(status=VerificationCheckStatus.ERROR, required=False),
        )
    )

    assert result.verdict is VerificationVerdict.ERROR


def test_status_and_verdict_are_not_inferred_from_text_or_evidence_ids() -> None:
    check = make_check(
        status=VerificationCheckStatus.PASS,
        summary="The LLM says tests failed and the build broke.",
        details="exit_code=17 FAIL ERROR",
        evidence_ids=("evidence-failed-17",),
    )
    result = make_result(checks=(check,))

    assert check.status is VerificationCheckStatus.PASS
    assert result.verdict is VerificationVerdict.PASS


def test_verification_model_has_no_runtime_or_execution_dependencies() -> None:
    tree = ast.parse(inspect.getsource(verification_module))
    forbidden_roots = {
        "run_agent",
        "agent",
        "hermes_cli",
        "tools",
        "subprocess",
        "pathlib",
    }
    imported_roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.partition(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.partition(".")[0])

    assert imported_roots.isdisjoint(forbidden_roots)

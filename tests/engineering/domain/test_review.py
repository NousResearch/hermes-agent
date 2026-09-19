from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

import engineering.domain.review as review_module
from engineering.domain.review import (
    ReviewCategory,
    ReviewFinding,
    ReviewResult,
    ReviewSeverity,
    ReviewVerdict,
)


STARTED_AT = datetime(2026, 8, 15, 9, 0, tzinfo=timezone.utc)
FINISHED_AT = STARTED_AT + timedelta(seconds=5, milliseconds=250)


def make_finding(**overrides: object) -> ReviewFinding:
    values: dict[str, object] = {
        "category": ReviewCategory.CORRECTNESS,
        "severity": ReviewSeverity.INFO,
        "message": "The change follows the stated behavior.",
    }
    values.update(overrides)
    return ReviewFinding(**values)  # type: ignore[arg-type]


def make_result(**overrides: object) -> ReviewResult:
    values: dict[str, object] = {
        "workflow_run_id": "workflow-123",
        "attempt": 1,
        "findings": (),
        "reviewer": "deterministic-reviewer",
        "started_at": STARTED_AT,
        "finished_at": FINISHED_AT,
    }
    values.update(overrides)
    return ReviewResult(**values)  # type: ignore[arg-type]


def test_declares_review_verdicts_severities_and_categories() -> None:
    assert [verdict.value for verdict in ReviewVerdict] == [
        "PASS",
        "NEEDS_WORK",
        "BLOCKED",
    ]
    assert [severity.value for severity in ReviewSeverity] == [
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ]
    assert [category.value for category in ReviewCategory] == [
        "CORRECTNESS",
        "SECURITY",
        "COMPATIBILITY",
        "ARCHITECTURE",
        "MAINTAINABILITY",
        "TESTING",
        "ERROR_HANDLING",
        "LOGGING",
        "SCOPE",
        "OTHER",
    ]


def test_generates_uuid4_finding_identity_by_default() -> None:
    first = make_finding()
    second = make_finding()

    assert UUID(first.finding_id).version == 4
    assert UUID(second.finding_id).version == 4
    assert first.finding_id != second.finding_id


def test_preserves_explicit_finding_id() -> None:
    finding = make_finding(finding_id="finding-imported-1")

    assert finding.finding_id == "finding-imported-1"


@pytest.mark.parametrize("message", ["", "   ", None])
def test_finding_message_cannot_be_empty(message: object) -> None:
    with pytest.raises(ValueError, match="message must be a non-empty string"):
        make_finding(message=message)


@pytest.mark.parametrize("line", [0, -1, True, 1.5, "1"])
def test_line_must_be_positive_when_supplied(line: object) -> None:
    with pytest.raises(
        ValueError, match="line must be a positive integer when supplied"
    ):
        make_finding(line=line)


def test_preserves_optional_source_location_and_recommendation() -> None:
    finding = make_finding(
        file_path="engineering/domain/review.py",
        line=42,
        recommendation="Keep the aggregate deterministic.",
    )

    assert finding.file_path == "engineering/domain/review.py"
    assert finding.line == 42
    assert finding.recommendation == "Keep the aggregate deterministic."


def test_evidence_ids_are_an_immutable_tuple() -> None:
    finding = make_finding(evidence_ids=["evidence-1", "evidence-2"])

    assert finding.evidence_ids == ("evidence-1", "evidence-2")
    with pytest.raises(TypeError):
        finding.evidence_ids[0] = "changed"  # type: ignore[index]


def test_review_finding_is_frozen() -> None:
    finding = make_finding()

    with pytest.raises(FrozenInstanceError):
        finding.message = "changed"  # type: ignore[misc]


def test_generates_uuid4_review_identity_by_default() -> None:
    first = make_result()
    second = make_result()

    assert UUID(first.review_id).version == 4
    assert UUID(second.review_id).version == 4
    assert first.review_id != second.review_id


def test_preserves_explicit_review_id_workflow_lineage_and_reviewer() -> None:
    result = make_result(
        review_id="review-imported-1",
        workflow_run_id="workflow-99",
        attempt=3,
        reviewer="human",
    )

    assert result.review_id == "review-imported-1"
    assert result.workflow_run_id == "workflow-99"
    assert result.attempt == 3
    assert result.reviewer == "human"


@pytest.mark.parametrize("attempt", [0, -1, True, 1.5, "1"])
def test_attempt_must_be_a_positive_integer(attempt: object) -> None:
    with pytest.raises(
        ValueError, match="attempt must be an integer greater than zero"
    ):
        make_result(attempt=attempt)


def test_accepts_aware_timestamps_and_calculates_duration() -> None:
    offset = timezone(timedelta(hours=8))
    started_at = datetime(2026, 8, 15, 17, 0, tzinfo=offset)
    finished_at = started_at + timedelta(seconds=7, microseconds=125)

    result = make_result(started_at=started_at, finished_at=finished_at)

    assert result.started_at == started_at
    assert result.finished_at == finished_at
    assert result.duration == timedelta(seconds=7, microseconds=125)


@pytest.mark.parametrize("field_name", ["started_at", "finished_at"])
def test_rejects_naive_timestamps(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be timezone-aware"):
        make_result(**{field_name: datetime(2026, 8, 15, 9, 0)})


def test_rejects_finished_at_before_started_at() -> None:
    with pytest.raises(
        ValueError, match="finished_at cannot be earlier than started_at"
    ):
        make_result(finished_at=STARTED_AT - timedelta(microseconds=1))


def test_findings_are_an_immutable_tuple_and_result_is_frozen() -> None:
    finding = make_finding()
    result = make_result(findings=[finding])

    assert result.findings == (finding,)
    with pytest.raises(TypeError):
        result.findings[0] = make_finding()  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        result.reviewer = "changed"  # type: ignore[misc]


def test_zero_findings_is_valid_and_passes_unlike_empty_verification() -> None:
    result = make_result(findings=())

    assert result.findings == ()
    assert result.verdict is ReviewVerdict.PASS


@pytest.mark.parametrize(
    "severities",
    [
        (ReviewSeverity.INFO,),
        (ReviewSeverity.WARNING,),
        (ReviewSeverity.INFO, ReviewSeverity.WARNING),
    ],
)
def test_info_and_warning_findings_pass(
    severities: tuple[ReviewSeverity, ...],
) -> None:
    result = make_result(
        findings=tuple(
            make_finding(severity=severity) for severity in severities
        )
    )

    assert result.verdict is ReviewVerdict.PASS


def test_error_finding_needs_work() -> None:
    result = make_result(
        findings=(make_finding(severity=ReviewSeverity.ERROR),)
    )

    assert result.verdict is ReviewVerdict.NEEDS_WORK


def test_critical_finding_blocks_review() -> None:
    result = make_result(
        findings=(make_finding(severity=ReviewSeverity.CRITICAL),)
    )

    assert result.verdict is ReviewVerdict.BLOCKED


def test_critical_takes_precedence_over_error() -> None:
    result = make_result(
        findings=(
            make_finding(severity=ReviewSeverity.ERROR),
            make_finding(severity=ReviewSeverity.CRITICAL),
        )
    )

    assert result.verdict is ReviewVerdict.BLOCKED


def test_verdict_is_not_inferred_from_message_or_recommendation_text() -> None:
    result = make_result(
        findings=(
            make_finding(
                severity=ReviewSeverity.INFO,
                message="CRITICAL ERROR: block completion immediately.",
                recommendation="NEEDS_WORK and BLOCKED",
            ),
        )
    )

    assert result.verdict is ReviewVerdict.PASS


@pytest.mark.parametrize("category", list(ReviewCategory))
def test_category_does_not_secretly_change_verdict(
    category: ReviewCategory,
) -> None:
    result = make_result(
        findings=(
            make_finding(
                category=category,
                severity=ReviewSeverity.WARNING,
            ),
        )
    )

    assert result.verdict is ReviewVerdict.PASS


def test_review_model_has_no_runtime_or_execution_dependencies() -> None:
    tree = ast.parse(inspect.getsource(review_module))
    forbidden_roots = {
        "run_agent",
        "agent",
        "hermes_cli",
        "tools",
        "subprocess",
        "pathlib",
        "os",
        "git",
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

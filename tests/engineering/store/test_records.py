from __future__ import annotations

import ast
import inspect
import json
from datetime import datetime, timedelta, timezone

import pytest

import engineering.store.base as base_module
import engineering.store.records as records_module
from engineering.domain import (
    Evidence,
    EvidenceKind,
    EvidenceStatus,
    ReviewCategory,
    ReviewFinding,
    ReviewResult,
    ReviewSeverity,
    VerificationCheckKind,
    VerificationCheckResult,
    VerificationCheckStatus,
    VerificationResult,
    WorkflowRun,
    WorkflowState,
)
from engineering.store.records import (
    SCHEMA_VERSION,
    UnsupportedSchemaVersion,
    evidence_from_record,
    evidence_to_record,
    review_from_record,
    review_to_record,
    verification_from_record,
    verification_to_record,
    workflow_from_record,
    workflow_to_record,
)


STARTED_AT = datetime(
    2026, 8, 15, 18, 0, tzinfo=timezone(timedelta(hours=8))
)
FINISHED_AT = STARTED_AT + timedelta(seconds=4, milliseconds=125)


def test_workflow_round_trip_preserves_snapshot() -> None:
    workflow = WorkflowRun(max_attempts=5)
    workflow.transition_to(
        WorkflowState.UNDERSTANDING,
        at=workflow.created_at + timedelta(seconds=1),
    )
    workflow.begin_next_attempt(at=workflow.created_at + timedelta(seconds=2))
    workflow.transition_to(
        WorkflowState.PLANNING,
        at=workflow.created_at + timedelta(seconds=3),
    )

    restored = workflow_from_record(workflow_to_record(workflow))

    assert restored.workflow_run_id == workflow.workflow_run_id
    assert restored.state is WorkflowState.PLANNING
    assert restored.attempt == 2
    assert restored.max_attempts == 5
    assert restored.created_at == workflow.created_at
    assert restored.updated_at == workflow.updated_at


def test_evidence_round_trip_preserves_auditable_context() -> None:
    evidence = Evidence(
        evidence_id="evidence-1",
        workflow_run_id="workflow-1",
        attempt=2,
        kind=EvidenceKind.TEST,
        status=EvidenceStatus.FAIL,
        producer="focused-pytest-check",
        summary="One focused test failed.",
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
        command="python -m pytest tests/unit/test_example.py",
        cwd="/workspace/project",
        execution_backend="container",
        exit_code=1,
        stdout_summary="11 passed, 1 failed",
        stderr_summary="",
        artifact_references=("artifact://logs/1", "artifact://junit/1"),
    )

    record = evidence_to_record(evidence)
    restored = evidence_from_record(record)

    assert restored.evidence_id == evidence.evidence_id
    assert restored.workflow_run_id == evidence.workflow_run_id
    assert restored.kind is EvidenceKind.TEST
    assert restored.status is EvidenceStatus.FAIL
    assert restored.started_at == evidence.started_at
    assert restored.finished_at == evidence.finished_at
    assert restored.command == evidence.command
    assert restored.cwd == evidence.cwd
    assert restored.execution_backend == evidence.execution_backend
    assert restored.exit_code == 1
    assert restored.artifact_references == evidence.artifact_references
    assert isinstance(restored.artifact_references, tuple)


def test_verification_round_trip_preserves_order_and_derived_verdict() -> None:
    checks = (
        VerificationCheckResult(
            check_id="check-build",
            kind=VerificationCheckKind.BUILD,
            status=VerificationCheckStatus.PASS,
            required=True,
            summary="Build passed.",
            evidence_ids=("evidence-build",),
        ),
        VerificationCheckResult(
            check_id="check-policy",
            kind=VerificationCheckKind.POLICY,
            status=VerificationCheckStatus.ERROR,
            required=False,
            summary="Policy check could not execute.",
            evidence_ids=("evidence-policy", "evidence-log"),
        ),
    )
    result = VerificationResult(
        verification_id="verification-1",
        workflow_run_id="workflow-1",
        attempt=2,
        checks=checks,
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )

    record = verification_to_record(result)
    restored = verification_from_record(record)

    assert restored.verification_id == result.verification_id
    assert [check.check_id for check in restored.checks] == [
        "check-build",
        "check-policy",
    ]
    assert restored.checks[1].evidence_ids == (
        "evidence-policy",
        "evidence-log",
    )
    assert restored.verdict is result.verdict
    assert "verdict" not in record


def test_verification_ignores_arbitrary_persisted_derived_verdict() -> None:
    result = VerificationResult(
        workflow_run_id="workflow-1",
        attempt=1,
        checks=(
            VerificationCheckResult(
                kind=VerificationCheckKind.TEST,
                status=VerificationCheckStatus.ERROR,
                required=True,
                summary="Test runner errored.",
            ),
        ),
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )
    record: dict[str, object] = verification_to_record(result)
    record["verdict"] = "PASS"

    restored = verification_from_record(record)

    assert restored.verdict.value == "ERROR"


def test_review_round_trip_preserves_order_reviewer_and_derived_verdict() -> None:
    findings = (
        ReviewFinding(
            finding_id="finding-info",
            category=ReviewCategory.MAINTAINABILITY,
            severity=ReviewSeverity.INFO,
            message="The structure is focused.",
            evidence_ids=("evidence-diff",),
        ),
        ReviewFinding(
            finding_id="finding-error",
            category=ReviewCategory.CORRECTNESS,
            severity=ReviewSeverity.ERROR,
            message="A correction is required.",
            evidence_ids=("evidence-test", "evidence-log"),
        ),
    )
    result = ReviewResult(
        review_id="review-1",
        workflow_run_id="workflow-1",
        attempt=2,
        findings=findings,
        reviewer="auxiliary-model-reviewer",
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
        summary="One correction is required.",
    )

    record = review_to_record(result)
    restored = review_from_record(record)

    assert restored.review_id == result.review_id
    assert [finding.finding_id for finding in restored.findings] == [
        "finding-info",
        "finding-error",
    ]
    assert restored.findings[1].evidence_ids == (
        "evidence-test",
        "evidence-log",
    )
    assert restored.reviewer == "auxiliary-model-reviewer"
    assert restored.verdict is result.verdict
    assert "verdict" not in record


def test_records_are_explicit_versioned_and_json_compatible() -> None:
    workflow = WorkflowRun()
    evidence = Evidence(
        workflow_run_id=workflow.workflow_run_id,
        attempt=1,
        kind=EvidenceKind.PROJECT_INSPECTION,
        status=EvidenceStatus.PASS,
        producer="project-inspector",
        summary="Inspection completed.",
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )
    verification = VerificationResult(
        workflow_run_id=workflow.workflow_run_id,
        attempt=1,
        checks=(
            VerificationCheckResult(
                kind=VerificationCheckKind.CUSTOM,
                status=VerificationCheckStatus.PASS,
                required=True,
                summary="Custom check passed.",
            ),
        ),
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )
    review = ReviewResult(
        workflow_run_id=workflow.workflow_run_id,
        attempt=1,
        findings=(),
        reviewer="deterministic-reviewer",
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )

    records = (
        workflow_to_record(workflow),
        evidence_to_record(evidence),
        verification_to_record(verification),
        review_to_record(review),
    )

    for record in records:
        assert record["schema_version"] == SCHEMA_VERSION == 1
        json.dumps(record)


@pytest.mark.parametrize(
    ("to_record", "from_record", "domain_object"),
    [
        (workflow_to_record, workflow_from_record, WorkflowRun()),
        (
            evidence_to_record,
            evidence_from_record,
            Evidence(
                workflow_run_id="workflow-1",
                attempt=1,
                kind=EvidenceKind.OTHER,
                status=EvidenceStatus.SKIPPED,
                producer="producer",
                summary="Skipped.",
                started_at=STARTED_AT,
                finished_at=FINISHED_AT,
            ),
        ),
        (
            verification_to_record,
            verification_from_record,
            VerificationResult(
                workflow_run_id="workflow-1",
                attempt=1,
                checks=(
                    VerificationCheckResult(
                        kind=VerificationCheckKind.TEST,
                        status=VerificationCheckStatus.PASS,
                        required=True,
                        summary="Passed.",
                    ),
                ),
                started_at=STARTED_AT,
                finished_at=FINISHED_AT,
            ),
        ),
        (
            review_to_record,
            review_from_record,
            ReviewResult(
                workflow_run_id="workflow-1",
                attempt=1,
                findings=(),
                reviewer="human",
                started_at=STARTED_AT,
                finished_at=FINISHED_AT,
            ),
        ),
    ],
)
def test_unsupported_schema_version_is_rejected(
    to_record: object,
    from_record: object,
    domain_object: object,
) -> None:
    record = to_record(domain_object)  # type: ignore[operator]
    record["schema_version"] = 2

    with pytest.raises(
        UnsupportedSchemaVersion,
        match="unsupported engineering record schema_version: 2",
    ):
        from_record(record)  # type: ignore[operator]


def test_store_layer_has_no_runtime_or_storage_dependencies() -> None:
    forbidden_roots = {
        "run_agent",
        "agent",
        "hermes_cli",
        "tools",
        "hermes_state",
        "subprocess",
        "pathlib",
        "sqlite3",
        "pickle",
    }

    for module in (base_module, records_module):
        tree = ast.parse(inspect.getsource(module))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.partition(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.partition(".")[0])

        assert imported_roots.isdisjoint(forbidden_roots)

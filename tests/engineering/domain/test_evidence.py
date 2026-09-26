from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

import engineering.domain.evidence as evidence_module
from engineering.domain.evidence import Evidence, EvidenceKind, EvidenceStatus


STARTED_AT = datetime(2026, 8, 15, 8, 0, tzinfo=timezone.utc)
FINISHED_AT = STARTED_AT + timedelta(seconds=4, milliseconds=250)


def make_evidence(**overrides: object) -> Evidence:
    values: dict[str, object] = {
        "workflow_run_id": "workflow-123",
        "attempt": 1,
        "kind": EvidenceKind.TEST,
        "status": EvidenceStatus.PASS,
        "producer": "focused-pytest-check",
        "summary": "Focused tests completed.",
        "started_at": STARTED_AT,
        "finished_at": FINISHED_AT,
    }
    values.update(overrides)
    return Evidence(**values)  # type: ignore[arg-type]


def test_declares_evidence_kinds_and_statuses() -> None:
    assert [kind.value for kind in EvidenceKind] == [
        "BUILD",
        "TEST",
        "LINT",
        "GIT_DIFF",
        "POLICY",
        "PROJECT_INSPECTION",
        "COMMAND",
        "OTHER",
    ]
    assert [status.value for status in EvidenceStatus] == [
        "PASS",
        "FAIL",
        "ERROR",
        "SKIPPED",
    ]


def test_generates_uuid4_evidence_identity_by_default() -> None:
    first = make_evidence()
    second = make_evidence()

    assert UUID(first.evidence_id).version == 4
    assert UUID(second.evidence_id).version == 4
    assert first.evidence_id != second.evidence_id


def test_preserves_explicit_evidence_id_and_workflow_lineage() -> None:
    evidence = make_evidence(
        evidence_id="imported-evidence-7",
        workflow_run_id="workflow-run-99",
        attempt=3,
    )

    assert evidence.evidence_id == "imported-evidence-7"
    assert evidence.workflow_run_id == "workflow-run-99"
    assert evidence.attempt == 3


@pytest.mark.parametrize("attempt", [0, -1, True, 1.5, "1"])
def test_attempt_must_be_a_positive_integer(attempt: object) -> None:
    with pytest.raises(
        ValueError, match="attempt must be an integer greater than zero"
    ):
        make_evidence(attempt=attempt)


def test_accepts_timezone_aware_timestamps_and_calculates_duration() -> None:
    offset = timezone(timedelta(hours=8))
    started_at = datetime(2026, 8, 15, 16, 0, tzinfo=offset)
    finished_at = started_at + timedelta(seconds=7, microseconds=125)

    evidence = make_evidence(
        started_at=started_at,
        finished_at=finished_at,
    )

    assert evidence.started_at == started_at
    assert evidence.finished_at == finished_at
    assert evidence.duration == timedelta(seconds=7, microseconds=125)


@pytest.mark.parametrize("field_name", ["started_at", "finished_at"])
def test_rejects_naive_timestamps(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be timezone-aware"):
        make_evidence(**{field_name: datetime(2026, 8, 15, 8, 0)})


def test_rejects_finished_at_before_started_at() -> None:
    with pytest.raises(
        ValueError, match="finished_at cannot be earlier than started_at"
    ):
        make_evidence(finished_at=STARTED_AT - timedelta(microseconds=1))


def test_preserves_command_execution_context() -> None:
    evidence = make_evidence(
        command="python -m pytest tests/unit/test_example.py",
        cwd="/workspace/project",
        execution_backend="local",
        exit_code=0,
        stdout_summary="12 tests passed",
        stderr_summary="",
    )

    assert evidence.command == "python -m pytest tests/unit/test_example.py"
    assert evidence.cwd == "/workspace/project"
    assert evidence.execution_backend == "local"
    assert evidence.exit_code == 0
    assert evidence.stdout_summary == "12 tests passed"
    assert evidence.stderr_summary == ""


def test_command_backed_evidence_requires_auditable_environment_context() -> None:
    with pytest.raises(
        ValueError,
        match="requires command, cwd, and execution_backend",
    ):
        make_evidence(command="pytest", cwd="/workspace")


def test_non_command_evidence_does_not_require_execution_fields() -> None:
    evidence = make_evidence(
        kind=EvidenceKind.PROJECT_INSPECTION,
        status=EvidenceStatus.SKIPPED,
        producer="project-inspector",
        summary="Inspection was intentionally skipped.",
    )

    assert evidence.command is None
    assert evidence.cwd is None
    assert evidence.execution_backend is None
    assert evidence.exit_code is None


def test_evidence_and_artifact_references_are_immutable() -> None:
    evidence = make_evidence(
        artifact_references=[
            "artifact://logs/test-run-1",
            "artifact://diffs/change-1",
        ]
    )

    assert evidence.artifact_references == (
        "artifact://logs/test-run-1",
        "artifact://diffs/change-1",
    )
    with pytest.raises(FrozenInstanceError):
        evidence.summary = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        evidence.artifact_references[0] = "changed"  # type: ignore[index]


def test_pass_status_is_not_inferred_from_exit_code_or_text() -> None:
    evidence = make_evidence(
        status=EvidenceStatus.PASS,
        summary="An explanation says the tests failed.",
        command="custom-check",
        cwd="/workspace/project",
        execution_backend="container",
        exit_code=17,
        stdout_summary="FAIL written by an untrusted narrator",
        stderr_summary="build failed",
    )

    assert evidence.status is EvidenceStatus.PASS
    assert evidence.exit_code == 17


@pytest.mark.parametrize("exit_code", [None, 0, 1, 143])
def test_fail_status_can_coexist_with_any_exit_code(
    exit_code: int | None,
) -> None:
    execution_fields: dict[str, object] = {}
    if exit_code is not None:
        execution_fields = {
            "command": "policy-check",
            "cwd": "/workspace/project",
            "execution_backend": "ssh",
            "exit_code": exit_code,
        }

    evidence = make_evidence(
        kind=EvidenceKind.POLICY,
        status=EvidenceStatus.FAIL,
        **execution_fields,
    )

    assert evidence.status is EvidenceStatus.FAIL
    assert evidence.exit_code == exit_code


def test_evidence_model_has_no_runtime_or_execution_dependencies() -> None:
    tree = ast.parse(inspect.getsource(evidence_module))
    forbidden_roots = {
        "run_agent",
        "agent",
        "hermes_cli",
        "tools",
        "subprocess",
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

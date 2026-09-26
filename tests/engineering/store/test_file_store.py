from __future__ import annotations

import ast
import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import engineering.store.file as file_store_module
from engineering.domain import (
    Evidence,
    EvidenceKind,
    EvidenceStatus,
    ReviewCategory,
    ReviewFinding,
    ReviewResult,
    ReviewSeverity,
    ReviewVerdict,
    VerificationCheckKind,
    VerificationCheckResult,
    VerificationCheckStatus,
    VerificationResult,
    VerificationVerdict,
    WorkflowRun,
    WorkflowState,
)
from engineering.store import (
    EngineeringStoreCorruption,
    EngineeringStoreError,
    EvidenceAlreadyExists,
    EvidenceNotFound,
    FileEngineeringStore,
    InvalidWorkflowIdentifier,
    ReviewAlreadyExists,
    VerificationAlreadyExists,
    WorkflowAlreadyExists,
    WorkflowNotFound,
)
from engineering.store.records import UnsupportedSchemaVersion


STARTED_AT = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
FINISHED_AT = STARTED_AT + timedelta(seconds=3)


def create_store(tmp_path: Path) -> tuple[FileEngineeringStore, Path]:
    root = tmp_path / "configured-store"
    return FileEngineeringStore(root), root


def create_workflow(store: FileEngineeringStore) -> WorkflowRun:
    workflow = WorkflowRun(max_attempts=4)
    store.create_workflow(workflow)
    return workflow


def make_evidence(
    workflow: WorkflowRun,
    *,
    evidence_id: str,
    attempt: int = 1,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        workflow_run_id=workflow.workflow_run_id,
        attempt=attempt,
        kind=EvidenceKind.TEST,
        status=EvidenceStatus.PASS,
        producer="focused-test",
        summary="Focused test passed.",
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )


def make_verification(
    workflow_run_id: str,
    *,
    status: VerificationCheckStatus = VerificationCheckStatus.PASS,
) -> VerificationResult:
    return VerificationResult(
        workflow_run_id=workflow_run_id,
        attempt=1,
        checks=(
            VerificationCheckResult(
                kind=VerificationCheckKind.TEST,
                status=status,
                required=True,
                summary="Structured verification status.",
            ),
        ),
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )


def make_review(
    workflow_run_id: str,
    *,
    severity: ReviewSeverity = ReviewSeverity.INFO,
) -> ReviewResult:
    return ReviewResult(
        workflow_run_id=workflow_run_id,
        attempt=1,
        findings=(
            ReviewFinding(
                category=ReviewCategory.CORRECTNESS,
                severity=severity,
                message="Structured review finding.",
            ),
        ),
        reviewer="deterministic-reviewer",
        started_at=STARTED_AT,
        finished_at=FINISHED_AT,
    )


def run_dir(root: Path, workflow: WorkflowRun) -> Path:
    return root / "runs" / workflow.workflow_run_id


def test_create_workflow_creates_run_and_snapshot(tmp_path: Path) -> None:
    store, root = create_store(tmp_path)
    workflow = create_workflow(store)

    assert (run_dir(root, workflow) / "workflow.json").is_file()


def test_duplicate_workflow_create_is_rejected(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)
    workflow = create_workflow(store)

    with pytest.raises(WorkflowAlreadyExists):
        store.create_workflow(workflow)


def test_get_workflow_restores_snapshot(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)
    workflow = create_workflow(store)

    restored = store.get_workflow(workflow.workflow_run_id)

    assert restored.workflow_run_id == workflow.workflow_run_id
    assert restored.state is WorkflowState.CREATED


def test_save_existing_workflow_replaces_snapshot(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)
    workflow = create_workflow(store)
    workflow.transition_to(WorkflowState.UNDERSTANDING)

    store.save_workflow(workflow)

    restored = store.get_workflow(workflow.workflow_run_id)

    assert restored.state is WorkflowState.UNDERSTANDING


def test_save_missing_workflow_is_rejected(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)

    with pytest.raises(WorkflowNotFound):
        store.save_workflow(WorkflowRun())


def test_workflow_round_trip_preserves_state_attempts_and_time(
    tmp_path: Path,
) -> None:
    store, _ = create_store(tmp_path)
    workflow = create_workflow(store)
    workflow.transition_to(
        WorkflowState.UNDERSTANDING,
        at=workflow.created_at + timedelta(seconds=1),
    )
    workflow.begin_next_attempt(at=workflow.created_at + timedelta(seconds=2))
    store.save_workflow(workflow)

    restored = store.get_workflow(workflow.workflow_run_id)

    assert restored.state is WorkflowState.UNDERSTANDING
    assert restored.attempt == 2
    assert restored.max_attempts == 4
    assert restored.created_at == workflow.created_at
    assert restored.updated_at == workflow.updated_at


def test_workflow_replacement_leaves_readable_json_and_no_temp_file(
    tmp_path: Path,
) -> None:
    store, root = create_store(tmp_path)
    workflow = create_workflow(store)
    workflow.transition_to(WorkflowState.UNDERSTANDING)
    store.save_workflow(workflow)
    directory = run_dir(root, workflow)

    with (directory / "workflow.json").open(encoding="utf-8") as stream:
        assert json.load(stream)["state"] == "UNDERSTANDING"
    assert list(directory.glob("*.tmp")) == []
    assert list(directory.glob(".*.tmp")) == []


def test_append_evidence_writes_exactly_one_jsonl_record(tmp_path: Path) -> None:
    store, root = create_store(tmp_path)
    workflow = create_workflow(store)

    store.append_evidence(make_evidence(workflow, evidence_id="evidence-1"))

    lines = (run_dir(root, workflow) / "evidence.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["evidence_id"] == "evidence-1"


def test_multiple_evidence_records_preserve_append_order(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)
    workflow = create_workflow(store)
    for evidence_id in ("evidence-1", "evidence-2", "evidence-3"):
        store.append_evidence(make_evidence(workflow, evidence_id=evidence_id))

    restored = store.list_evidence(workflow.workflow_run_id)

    assert [item.evidence_id for item in restored] == [
        "evidence-1",
        "evidence-2",
        "evidence-3",
    ]


def test_duplicate_evidence_id_in_same_workflow_is_rejected(
    tmp_path: Path,
) -> None:
    store, _ = create_store(tmp_path)
    workflow = create_workflow(store)
    original = make_evidence(workflow, evidence_id="unique-evidence")
    store.append_evidence(original)

    with pytest.raises(EvidenceAlreadyExists):
        store.append_evidence(original)

    assert store.list_evidence(workflow.workflow_run_id) == (original,)


def test_duplicate_evidence_id_across_workflows_is_rejected(
    tmp_path: Path,
) -> None:
    store, _ = create_store(tmp_path)
    first_workflow = create_workflow(store)
    second_workflow = create_workflow(store)
    store.append_evidence(
        make_evidence(first_workflow, evidence_id="root-unique-evidence")
    )

    with pytest.raises(EvidenceAlreadyExists):
        store.append_evidence(
            make_evidence(second_workflow, evidence_id="root-unique-evidence")
        )

    assert store.list_evidence(second_workflow.workflow_run_id) == ()


def test_list_evidence_returns_all_records(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)
    workflow = create_workflow(store)
    store.append_evidence(make_evidence(workflow, evidence_id="evidence-1"))
    store.append_evidence(make_evidence(workflow, evidence_id="evidence-2"))

    assert len(store.list_evidence(workflow.workflow_run_id)) == 2


def test_list_evidence_filters_by_attempt(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)
    workflow = create_workflow(store)
    store.append_evidence(
        make_evidence(workflow, evidence_id="attempt-1", attempt=1)
    )
    store.append_evidence(
        make_evidence(workflow, evidence_id="attempt-2", attempt=2)
    )

    filtered = store.list_evidence(workflow.workflow_run_id, attempt=2)

    assert [item.evidence_id for item in filtered] == ["attempt-2"]


def test_get_evidence_finds_exact_identity(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)
    workflow = create_workflow(store)
    store.append_evidence(make_evidence(workflow, evidence_id="evidence-1"))

    restored = store.get_evidence("evidence-1")

    assert restored.workflow_run_id == workflow.workflow_run_id


def test_missing_evidence_is_rejected(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)

    with pytest.raises(EvidenceNotFound):
        store.get_evidence("missing-evidence")


def test_append_evidence_to_missing_workflow_is_rejected(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)
    missing = WorkflowRun()

    with pytest.raises(WorkflowNotFound):
        store.append_evidence(make_evidence(missing, evidence_id="orphan"))


def test_corrupt_evidence_line_is_not_skipped(tmp_path: Path) -> None:
    store, root = create_store(tmp_path)
    workflow = create_workflow(store)
    path = run_dir(root, workflow) / "evidence.jsonl"
    path.write_text('{"invalid":\n', encoding="utf-8")

    with pytest.raises(EngineeringStoreCorruption, match="line=1"):
        store.list_evidence(workflow.workflow_run_id)


def test_save_and_get_verification(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)
    workflow = create_workflow(store)
    result = make_verification(workflow.workflow_run_id)

    store.save_verification(result)

    restored = store.get_verification(workflow.workflow_run_id, 1)

    assert restored.verification_id == result.verification_id


def test_duplicate_verification_attempt_is_rejected(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)
    workflow = create_workflow(store)
    original = make_verification(workflow.workflow_run_id)
    store.save_verification(original)

    with pytest.raises(VerificationAlreadyExists):
        store.save_verification(make_verification(workflow.workflow_run_id))

    restored = store.get_verification(workflow.workflow_run_id, 1)
    assert restored.verification_id == original.verification_id


def test_verification_for_missing_workflow_is_rejected(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)

    with pytest.raises(WorkflowNotFound):
        store.save_verification(make_verification(WorkflowRun().workflow_run_id))


def test_verification_verdict_is_reconstructed_by_domain(tmp_path: Path) -> None:
    store, root = create_store(tmp_path)
    workflow = create_workflow(store)
    result = make_verification(
        workflow.workflow_run_id,
        status=VerificationCheckStatus.ERROR,
    )
    store.save_verification(result)
    path = run_dir(root, workflow) / "verifications" / "attempt-1.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["verdict"] = "PASS"
    path.write_text(json.dumps(record), encoding="utf-8")

    restored = store.get_verification(workflow.workflow_run_id, 1)

    assert restored.verdict is VerificationVerdict.ERROR


def test_save_and_get_review(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)
    workflow = create_workflow(store)
    result = make_review(workflow.workflow_run_id)

    store.save_review(result)

    restored = store.get_review(workflow.workflow_run_id, 1)

    assert restored.review_id == result.review_id


def test_duplicate_review_attempt_is_rejected(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)
    workflow = create_workflow(store)
    original = make_review(workflow.workflow_run_id)
    store.save_review(original)

    with pytest.raises(ReviewAlreadyExists):
        store.save_review(make_review(workflow.workflow_run_id))

    restored = store.get_review(workflow.workflow_run_id, 1)
    assert restored.review_id == original.review_id


def test_review_for_missing_workflow_is_rejected(tmp_path: Path) -> None:
    store, _ = create_store(tmp_path)

    with pytest.raises(WorkflowNotFound):
        store.save_review(make_review(WorkflowRun().workflow_run_id))


def test_review_verdict_is_reconstructed_by_domain(tmp_path: Path) -> None:
    store, root = create_store(tmp_path)
    workflow = create_workflow(store)
    result = make_review(
        workflow.workflow_run_id,
        severity=ReviewSeverity.CRITICAL,
    )
    store.save_review(result)
    path = run_dir(root, workflow) / "reviews" / "attempt-1.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["verdict"] = "PASS"
    path.write_text(json.dumps(record), encoding="utf-8")

    restored = store.get_review(workflow.workflow_run_id, 1)

    assert restored.verdict is ReviewVerdict.BLOCKED


def test_corrupt_workflow_json_raises_corruption(tmp_path: Path) -> None:
    store, root = create_store(tmp_path)
    workflow = create_workflow(store)
    (run_dir(root, workflow) / "workflow.json").write_text(
        "not-json", encoding="utf-8"
    )

    with pytest.raises(EngineeringStoreCorruption):
        store.get_workflow(workflow.workflow_run_id)


def test_failed_workflow_write_before_replace_preserves_previous_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, root = create_store(tmp_path)
    workflow = create_workflow(store)
    workflow.transition_to(WorkflowState.UNDERSTANDING)

    def fail_during_dump(
        record: object,
        stream: object,
        **kwargs: object,
    ) -> None:
        del record, kwargs
        stream.write('{"partial":')  # type: ignore[attr-defined]
        raise OSError("injected write failure")

    monkeypatch.setattr(file_store_module.json, "dump", fail_during_dump)

    with pytest.raises(EngineeringStoreError) as captured:
        store.save_workflow(workflow)

    assert isinstance(captured.value.__cause__, OSError)
    restored = store.get_workflow(workflow.workflow_run_id)
    assert restored.state is WorkflowState.CREATED
    assert list(run_dir(root, workflow).glob(".*.tmp")) == []


def test_unsupported_schema_is_wrapped_as_corruption(tmp_path: Path) -> None:
    store, root = create_store(tmp_path)
    workflow = create_workflow(store)
    path = run_dir(root, workflow) / "workflow.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["schema_version"] = 2
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(EngineeringStoreCorruption) as captured:
        store.get_workflow(workflow.workflow_run_id)

    assert isinstance(captured.value.__cause__, UnsupportedSchemaVersion)


def test_corrupt_verification_json_raises_corruption(tmp_path: Path) -> None:
    store, root = create_store(tmp_path)
    workflow = create_workflow(store)
    store.save_verification(make_verification(workflow.workflow_run_id))
    path = run_dir(root, workflow) / "verifications" / "attempt-1.json"
    path.write_text("[", encoding="utf-8")

    with pytest.raises(EngineeringStoreCorruption):
        store.get_verification(workflow.workflow_run_id, 1)


def test_corrupt_review_json_raises_corruption(tmp_path: Path) -> None:
    store, root = create_store(tmp_path)
    workflow = create_workflow(store)
    store.save_review(make_review(workflow.workflow_run_id))
    path = run_dir(root, workflow) / "reviews" / "attempt-1.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(EngineeringStoreCorruption):
        store.get_review(workflow.workflow_run_id, 1)


def test_configured_root_is_respected(tmp_path: Path) -> None:
    store, root = create_store(tmp_path)
    workflow = create_workflow(store)

    assert run_dir(root, workflow).is_dir()
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "workflow_run_id",
    ["../escape", "..\\escape", "/absolute", "nested/run", "C:\\escape"],
)
def test_path_traversal_identifiers_are_rejected(
    tmp_path: Path,
    workflow_run_id: str,
) -> None:
    store, _ = create_store(tmp_path)

    with pytest.raises(InvalidWorkflowIdentifier):
        store.get_workflow(workflow_run_id)


def test_file_store_has_no_hermes_or_session_dependency() -> None:
    tree = ast.parse(inspect.getsource(file_store_module))
    forbidden_roots = {
        "run_agent",
        "agent",
        "hermes_cli",
        "tools",
        "hermes_state",
        "sqlite3",
        "subprocess",
        "pickle",
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

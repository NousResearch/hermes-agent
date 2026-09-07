from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from engineering_evidence.experience import (
    promote_experience_candidate,
    search_experience_candidates,
    store_experience_candidate,
)
from engineering_evidence.correlation import build_deployment_correlation_receipt
from engineering_evidence.kanban import attach_receipt_to_task, list_task_receipts
from engineering_evidence.receipts import (
    ReceiptValidationError,
    validate_exact_head,
    validate_receipt,
)
from engineering_evidence.workflow import create_evidence_children, evaluate_issue_to_pr_gate
from engineering_evidence.scanner import build_codebase_snapshot
from engineering_evidence.testing import build_test_discovery_receipt


def test_receipts_require_diagnostic_authority_and_exact_head() -> None:
    with pytest.raises(ReceiptValidationError, match="authority"):
        validate_receipt(
            {
                "schema_name": "engineering_evidence_v1",
                "receipt_type": "codebase_snapshot",
                "receipt_id": "snap-1",
                "repository": "example/repo",
                "head_sha": "a" * 40,
                "authority": "accepted",
            }
        )


def test_receipts_reject_stale_repository_heads() -> None:
    receipt = {
        "schema_name": "engineering_evidence_v1",
        "receipt_type": "codebase_snapshot",
        "receipt_id": "snap-1",
        "repository": "example/repo",
        "head_sha": "a" * 40,
        "authority": "diagnostic-only",
    }

    with pytest.raises(ReceiptValidationError, match="stale head"):
        validate_exact_head(receipt, "b" * 40)


def test_snapshot_records_exact_head_and_tracked_file_evidence(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "app.py").write_text("import util\n\nVALUE = 1\n", encoding="utf-8")
    (tmp_path / "util.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "ignored.tmp").write_text("not tracked\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "app.py", "util.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
    )

    receipt = build_codebase_snapshot(tmp_path, query="app")

    assert receipt["receipt_type"] == "codebase_snapshot"
    assert len(receipt["head_sha"]) == 40
    assert "app.py" in receipt["tracked_files"]
    assert "ignored.tmp" not in receipt["tracked_files"]
    assert receipt["relevant_files"] == ["app.py"]
    assert receipt["authority"] == "diagnostic-only"


def test_snapshot_bounds_unreadable_file_diagnostics(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for index in range(60):
        (tmp_path / f"missing-{index}.txt").symlink_to(tmp_path / f"does-not-exist-{index}.txt")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "initial",
        ],
        check=True,
    )

    receipt = build_codebase_snapshot(tmp_path, query="does-not-match")

    assert len(receipt["incomplete_reasons"]) <= 51
    assert any(reason.startswith("unreadable_file_count:") for reason in receipt["incomplete_reasons"])


def test_test_discovery_preserves_unavailable_runs_as_degraded() -> None:
    receipt = build_test_discovery_receipt(
        repository="example/repo",
        head_sha="b" * 40,
        target="auth.login",
        cases=[{"name": "empty password", "kind": "invalid-input"}],
        commands=[{"argv": ["pytest", "-q"], "status": "unavailable", "reason": "pytest missing"}],
    )

    assert receipt["status"] == "degraded"
    assert receipt["authority"] == "diagnostic-only"
    assert receipt["incomplete_reasons"] == ["pytest missing"]


def test_experience_candidate_is_vault_compatible_and_secret_safe(tmp_path: Path) -> None:
    candidate = {
        "repository": "example/repo",
        "head_sha": "c" * 40,
        "task_id": "task-1",
        "problem": "login rejected valid tokens",
        "affected_files": ["auth.py"],
        "failure_mechanisms": ["stale cache"],
        "solution": "invalidate the cache on key rotation",
        "tests": [{"argv": ["pytest", "-q"], "status": "passed"}],
        "outcome": "resolved",
    }

    path = store_experience_candidate(tmp_path, candidate)

    metadata, body = path.read_text(encoding="utf-8").split("\n---\n", 1)
    assert "schema_name: agent_learning_record_v1" in metadata
    assert "classification: diagnostic-only" in metadata
    assert "sync_owned: true" in metadata
    assert "HERMES_GITHUB_BOT_TOKEN" not in body
    assert json.loads(body)["outcome"] == "resolved"


def test_experience_retrieval_requires_explicit_validation_and_exact_head(tmp_path: Path) -> None:
    candidate = {
        "repository": "example/repo",
        "head_sha": "d" * 40,
        "task_id": "task-2",
        "problem": "login rejected valid tokens",
        "affected_files": ["auth.py"],
        "failure_mechanisms": ["stale cache"],
        "solution": "invalidate the cache on key rotation",
        "tests": [{"argv": ["pytest", "-q"], "status": "passed"}],
        "outcome": "resolved",
    }
    path = store_experience_candidate(tmp_path, candidate)

    assert search_experience_candidates(tmp_path, "example/repo", "d" * 40, terms=["cache"]) == []

    promotion = promote_experience_candidate(
        path,
        validator_role="memory-validator",
        rationale="Receipt and focused test evidence were independently checked.",
    )
    assert promotion["status"] == "validated"
    matches = search_experience_candidates(tmp_path, "example/repo", "d" * 40, terms=["cache"])
    assert len(matches) == 1
    assert matches[0]["source_reference"] == str(path)
    assert search_experience_candidates(tmp_path, "example/repo", "e" * 40, terms=["cache"]) == []

    with pytest.raises(ValueError, match="validator role"):
        promote_experience_candidate(path, validator_role="coding-expert", rationale="not allowed")


def test_correlation_preserves_missing_operational_evidence() -> None:
    receipt = build_deployment_correlation_receipt(
        repository="example/repo",
        head_sha="e" * 40,
        identity={"commit": "e" * 40, "deployment": "deploy-1"},
        evidence={"ci": [{"receipt_id": "ci-1"}]},
        confidence="low",
        uncertainty=["runtime logs unavailable"],
    )

    assert receipt["status"] == "incomplete"
    assert "missing_evidence:runtime" in receipt["incomplete_reasons"]
    assert receipt["authority"] == "diagnostic-only"


def test_issue_to_pr_gate_is_fail_closed_and_side_effect_free() -> None:
    head = "f" * 40
    snapshot = {
        "schema_name": "engineering_evidence_v1",
        "receipt_type": "codebase_snapshot",
        "receipt_id": "snap-1",
        "repository": "example/repo",
        "head_sha": head,
        "authority": "diagnostic-only",
    }
    tests = {
        "schema_name": "engineering_evidence_v1",
        "receipt_type": "test_discovery",
        "receipt_id": "tests-1",
        "repository": "example/repo",
        "head_sha": head,
        "authority": "diagnostic-only",
    }

    blocked = evaluate_issue_to_pr_gate(
        [snapshot, tests],
        current_head=head,
        independent_review=False,
        operator_approved=False,
        hosted_checks_complete=False,
    )
    assert blocked["allowed"] is False
    assert blocked["publication_action"] == "none"
    assert "operator_approval_required" in blocked["reasons"]

    allowed = evaluate_issue_to_pr_gate(
        [snapshot, tests],
        current_head=head,
        independent_review=True,
        operator_approved=True,
        hosted_checks_complete=True,
    )
    assert allowed["allowed"] is True


def test_kanban_attachment_uses_existing_db_layer_and_binds_task(tmp_path: Path) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    db_path = tmp_path / "kanban.db"
    kbc.init_db(db_path)
    conn = kbc.connect(db_path)
    try:
        task_id = kb.create_task(conn, title="evidence task", body="test", initial_status="running")
    finally:
        conn.close()

    receipt = {
        "schema_name": "engineering_evidence_v1",
        "receipt_type": "test_discovery",
        "receipt_id": "tests-kanban-1",
        "repository": "example/repo",
        "head_sha": "1" * 40,
        "authority": "diagnostic-only",
        "task_id": task_id,
        "target": "auth.login",
        "cases": [],
        "commands": [],
        "status": "incomplete",
        "incomplete_reasons": ["not-run"],
    }

    attachment = attach_receipt_to_task(
        receipt,
        task_id=task_id,
        db_path=db_path,
        attachments_root=tmp_path / "attachments",
    )
    assert attachment["task_id"] == task_id
    records = list_task_receipts(task_id=task_id, db_path=db_path)
    assert records[0]["receipt_id"] == "tests-kanban-1"

    with pytest.raises(ValueError, match="task_id"):
        attach_receipt_to_task(receipt, task_id="missing-task", db_path=db_path)


def test_kanban_workflow_creates_explicit_bounded_children(tmp_path: Path) -> None:
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    db_path = tmp_path / "kanban.db"
    kbc.init_db(db_path)
    conn = kbc.connect(db_path)
    try:
        parent_id = kb.create_task(conn, title="parent evidence task", body="test", initial_status="running")
    finally:
        conn.close()

    result = create_evidence_children(
        parent_id=parent_id,
        repository="example/repo",
        head_sha="2" * 40,
        db_path=db_path,
    )
    assert result["automatic_authority"] == []
    assert len(result["children"]) == 4
    assert {child["assignee"] for child in result["children"]} == {
        "architecture-steward",
        "test-contract-steward",
        "review-verification-steward",
        "operations-steward",
    }

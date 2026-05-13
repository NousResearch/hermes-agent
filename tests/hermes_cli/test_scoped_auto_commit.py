from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli.scoped_auto_commit import (
    create_scoped_local_commit,
    evaluate_scoped_auto_commit_gate,
)

WORKFLOW_ID = "wf-gate10f-scoped-auto-commit"

CODER_METADATA = {
    "workflow_id": WORKFLOW_ID,
    "review_required": True,
    "coder_done_without_review": False,
    "scope_attestation": True,
    "scope_contract_version": 2,
    "forbidden_actions_taken": 0,
    "no_prose_only_terminal_output": True,
}

REVIEWER_APPROVED = {
    "workflow_id": WORKFLOW_ID,
    "verdict": "APPROVED",
    "blocking_findings": [],
    "required_verification": [],
    "residual_risk": "local scoped commit only",
    "evidence_audited": ["coder_handoff", "diff", "tests"],
    "scope_attestation": True,
    "scope_contract_version": 2,
    "forbidden_actions_taken": 0,
}

GREEN_CHECKS = {
    "implementation_tests_green": True,
    "postchecks_green": True,
    "reviewer_b_terminal_approved": True,
}

ANTI_SCOPE_GREEN = {
    "default_board_unchanged": True,
    "temp_credential_symlinks_removed": True,
    "no_deploy_reload": True,
    "no_mc_openclaw_discord": True,
    "no_config_systemd_cron_secret_changes": True,
    "no_unrelated_dirty_committed": True,
    "violations": [],
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    _write(repo / "README.md", "base\n")
    _write(repo / ".gitignore", "*.pyc\n")
    _git(repo, "add", "README.md", ".gitignore")
    _git(repo, "commit", "-m", "initial")
    return repo


def test_scoped_auto_commit_gate_blocks_without_reviewer_b_approved():
    decision = evaluate_scoped_auto_commit_gate(
        coder_metadata=CODER_METADATA,
        reviewer_metadata={**REVIEWER_APPROVED, "verdict": "NEEDS_REVISION"},
        acceptance_checks=GREEN_CHECKS,
        anti_scope=ANTI_SCOPE_GREEN,
        expected_workflow_id=WORKFLOW_ID,
    )

    assert not decision.allowed
    assert "reviewer verdict is NEEDS_REVISION" in decision.blocking_findings


@pytest.mark.parametrize(
    "coder_patch,checks_patch,anti_scope_patch,expected_fragment",
    [
        ({"review_required": False}, {}, {}, "coder review_required=true is required"),
        ({"forbidden_actions_taken": 1}, {}, {}, "coder forbidden_actions_taken must be 0"),
        ({}, {"postchecks_green": False}, {}, "acceptance check not green: postchecks_green"),
        ({}, {}, {"violations": ["deploy attempted"]}, "anti-scope violations present: deploy attempted"),
    ],
)
def test_scoped_auto_commit_gate_requires_green_checks_and_scope_zero(
    coder_patch, checks_patch, anti_scope_patch, expected_fragment
):
    decision = evaluate_scoped_auto_commit_gate(
        coder_metadata={**CODER_METADATA, **coder_patch},
        reviewer_metadata=REVIEWER_APPROVED,
        acceptance_checks={**GREEN_CHECKS, **checks_patch},
        anti_scope={**ANTI_SCOPE_GREEN, **anti_scope_patch},
        expected_workflow_id=WORKFLOW_ID,
    )

    assert not decision.allowed
    assert expected_fragment in decision.blocking_findings


def test_create_scoped_local_commit_commits_only_scoped_paths_and_leaves_preexisting_dirty(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / ".gitignore", "*.pyc\n# pre-existing dirty\n")
    _write(repo / "hermes_cli" / "scoped_change.py", "VALUE = 10\n")

    receipt = create_scoped_local_commit(
        repo_path=repo,
        scoped_paths=["hermes_cli/scoped_change.py"],
        message="test: scoped auto commit",
        coder_metadata=CODER_METADATA,
        reviewer_metadata=REVIEWER_APPROVED,
        acceptance_checks=GREEN_CHECKS,
        anti_scope=ANTI_SCOPE_GREEN,
        expected_workflow_id=WORKFLOW_ID,
    )

    assert receipt.reviewer_verdict == "APPROVED"
    assert receipt.committed_paths == ["hermes_cli/scoped_change.py"]
    assert ".gitignore" in receipt.preexisting_dirty_paths
    assert _git(repo, "show", "--name-only", "--format=", receipt.commit_hash).splitlines() == [
        "hermes_cli/scoped_change.py"
    ]
    assert ".gitignore" in _git(repo, "status", "--short")


def test_create_scoped_local_commit_rejects_staged_paths_outside_scope(tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "allowed.py", "allowed = True\n")
    _write(repo / "outside.py", "outside = True\n")
    _git(repo, "add", "outside.py")

    with pytest.raises(PermissionError, match="staged paths outside scoped commit: outside.py"):
        create_scoped_local_commit(
            repo_path=repo,
            scoped_paths=["allowed.py"],
            message="test: must not commit outside scope",
            coder_metadata=CODER_METADATA,
            reviewer_metadata=REVIEWER_APPROVED,
            acceptance_checks=GREEN_CHECKS,
            anti_scope=ANTI_SCOPE_GREEN,
            expected_workflow_id=WORKFLOW_ID,
        )

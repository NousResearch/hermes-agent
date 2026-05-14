from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb

WORKFLOW_ID = "wf-gate10g-review-commit"

CODER_METADATA = {
    "workflow_id": WORKFLOW_ID,
    "review_required": True,
    "coder_done_without_review": False,
    "scope_attestation": True,
    "scope_contract_version": 2,
    "forbidden_actions_taken": 0,
    "no_prose_only_terminal_output": True,
    "acceptance_checks": {
        "implementation_tests_green": True,
        "postchecks_green": True,
        "reviewer_b_terminal_approved": True,
    },
    "anti_scope": {
        "default_board_unchanged": True,
        "temp_credential_symlinks_removed": True,
        "no_deploy_reload": True,
        "no_mc_openclaw_discord": True,
        "no_config_systemd_cron_secret_changes": True,
        "no_unrelated_dirty_committed": True,
        "violations": [],
    },
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


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


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


def _task_id(output: str) -> str:
    match = re.search(r"(t_[a-f0-9]+)", output)
    assert match, output
    return match.group(1)


def _json_arg(payload: dict) -> str:
    return shlex.quote(json.dumps(payload))


def _create_completed_task(title: str, metadata: dict, assignee: str = "coder") -> str:
    tid = _task_id(kc.run_slash(f"create {shlex.quote(title)} --assignee {assignee}"))
    out = kc.run_slash(
        f"complete {tid} --summary {shlex.quote(title + ' done')} --metadata {_json_arg(metadata)}"
    )
    assert f"Completed {tid}" in out
    return tid


def test_review_commit_opt_in_commits_only_scoped_paths_and_leaves_dirty(kanban_home, tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / ".gitignore", "*.pyc\n# pre-existing dirty\n")
    _write(repo / "hermes_cli" / "wired.py", "VALUE = 10\n")
    coder_id = _create_completed_task("coder handoff", CODER_METADATA)
    reviewer_id = _create_completed_task("reviewer verdict", REVIEWER_APPROVED, assignee="reviewer")

    out = kc.run_slash(
        " ".join(
            [
                "review-commit",
                coder_id,
                "--reviewer-task",
                reviewer_id,
                "--repo",
                shlex.quote(str(repo)),
                "--scoped-path",
                "hermes_cli/wired.py",
                "--message",
                shlex.quote("test: scoped review commit"),
                "--json",
            ]
        )
    )
    payload = json.loads(out)

    assert payload["reviewer_verdict"] == "APPROVED"
    assert payload["committed_paths"] == ["hermes_cli/wired.py"]
    assert ".gitignore" in payload["preexisting_dirty_paths"]
    assert _git(repo, "show", "--name-only", "--format=", payload["commit_hash"]).splitlines() == [
        "hermes_cli/wired.py"
    ]
    assert ".gitignore" in _git(repo, "status", "--short")


def test_review_commit_blocks_missing_reviewer_task(kanban_home, tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "allowed.py", "allowed = True\n")
    coder_id = _create_completed_task("coder handoff", CODER_METADATA)

    out = kc.run_slash(
        f"review-commit {coder_id} --reviewer-task t_missing --repo {shlex.quote(str(repo))} "
        "--scoped-path allowed.py --message 'test: blocked'"
    )

    assert "reviewer task not found" in out
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"


def test_review_commit_blocks_non_approved_reviewer(kanban_home, tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "allowed.py", "allowed = True\n")
    coder_id = _create_completed_task("coder handoff", CODER_METADATA)
    reviewer_id = _create_completed_task(
        "reviewer verdict",
        {**REVIEWER_APPROVED, "verdict": "NEEDS_REVISION"},
        assignee="reviewer",
    )

    out = kc.run_slash(
        f"review-commit {coder_id} --reviewer-task {reviewer_id} --repo {shlex.quote(str(repo))} "
        "--scoped-path allowed.py --message 'test: blocked'"
    )

    assert "reviewer verdict is NEEDS_REVISION" in out
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"


def test_review_commit_blocks_staged_paths_outside_scope(kanban_home, tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "allowed.py", "allowed = True\n")
    _write(repo / "outside.py", "outside = True\n")
    _git(repo, "add", "outside.py")
    coder_id = _create_completed_task("coder handoff", CODER_METADATA)
    reviewer_id = _create_completed_task("reviewer verdict", REVIEWER_APPROVED, assignee="reviewer")

    out = kc.run_slash(
        f"review-commit {coder_id} --reviewer-task {reviewer_id} --repo {shlex.quote(str(repo))} "
        "--scoped-path allowed.py --message 'test: blocked'"
    )

    assert "staged paths outside scoped commit: outside.py" in out
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"


def test_review_commit_blocks_when_scoped_paths_have_no_staged_changes(kanban_home, tmp_path):
    repo = _init_repo(tmp_path)
    _write(repo / "allowed.py", "allowed = True\n")
    _git(repo, "add", "allowed.py")
    _git(repo, "commit", "--amend", "--no-edit")
    initial_commit_count = _git(repo, "rev-list", "--count", "HEAD")
    coder_id = _create_completed_task("coder handoff", CODER_METADATA)
    reviewer_id = _create_completed_task("reviewer verdict", REVIEWER_APPROVED, assignee="reviewer")

    out = kc.run_slash(
        f"review-commit {coder_id} --reviewer-task {reviewer_id} --repo {shlex.quote(str(repo))} "
        "--scoped-path allowed.py --message 'test: blocked'"
    )

    assert initial_commit_count == "1"
    assert "no scoped changes staged for commit" in out
    assert _git(repo, "rev-list", "--count", "HEAD") == initial_commit_count


def test_review_commit_blocks_unknown_scoped_path(kanban_home, tmp_path):
    repo = _init_repo(tmp_path)
    coder_id = _create_completed_task("coder handoff", CODER_METADATA)
    reviewer_id = _create_completed_task("reviewer verdict", REVIEWER_APPROVED, assignee="reviewer")

    out = kc.run_slash(
        f"review-commit {coder_id} --reviewer-task {reviewer_id} --repo {shlex.quote(str(repo))} "
        "--scoped-path missing.py --message 'test: blocked'"
    )

    assert "scoped path does not exist or is not tracked: missing.py" in out
    assert _git(repo, "rev-list", "--count", "HEAD") == "1"

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest


SCRIPT_ROOT = Path(os.environ.get("HERMES_SCRIPT_ROOT", str(Path.home() / ".hermes" / "scripts")))
PUBLISH_SCRIPT_PATH = SCRIPT_ROOT / "minos_publish.py"
publish_spec = importlib.util.spec_from_file_location("minos_publish", PUBLISH_SCRIPT_PATH)
minos_publish = importlib.util.module_from_spec(publish_spec)
publish_spec.loader.exec_module(minos_publish)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture
def repo_with_branch(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.com")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "checkout", "-b", "minos/run-001")
    return repo


def _write_decision(repo: Path, payload: dict) -> Path:
    artifact_dir = repo / ".hermes" / "runs" / "run-001"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "minos-decision.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def test_publish_rejects_without_decision_artifact(repo_with_branch: Path):
    with pytest.raises(FileNotFoundError, match="decision"):
        minos_publish.publish_run(
            repo_path=repo_with_branch,
            run_id="run-001",
            remote_name="origin",
            remote_url="git@github.com:soulsplitters-a11y/private-repo.git",
            remote_visibility="private",
            dry_run=True,
        )


def test_publish_rejects_without_gate_pass_even_if_publish_flags_claim_yes(repo_with_branch: Path, monkeypatch):
    monkeypatch.setattr(minos_publish, "resolve_github_visibility", lambda *_args, **_kwargs: "private")
    _write_decision(
        repo_with_branch,
        {
            "run_id": "run-001",
            "builder_status": "passed",
            "gate_status": "failed",
            "next_action": "publish",
            "decision_reason": "passed",
            "publish_recommended": True,
            "retry_recommended": False,
            "human_review_required": False,
            "blocking_findings": [],
        },
    )

    with pytest.raises(RuntimeError, match="gate pass"):
        minos_publish.publish_run(
            repo_path=repo_with_branch,
            run_id="run-001",
            remote_name="origin",
            remote_url="git@github.com:soulsplitters-a11y/private-repo.git",
            remote_visibility="private",
            dry_run=True,
        )


def test_publish_rejects_when_decision_bundle_failed(repo_with_branch: Path, monkeypatch):
    monkeypatch.setattr(minos_publish, "resolve_github_visibility", lambda *_args, **_kwargs: "private")
    _write_decision(
        repo_with_branch,
        {
            "run_id": "run-001",
            "builder_status": "passed",
            "gate_status": "failed",
            "next_action": "retry",
            "decision_reason": "verifier_failed",
            "publish_recommended": False,
            "retry_recommended": True,
            "human_review_required": False,
            "blocking_findings": ["verifier failed"],
        },
    )

    with pytest.raises(RuntimeError, match="publish"):
        minos_publish.publish_run(
            repo_path=repo_with_branch,
            run_id="run-001",
            remote_name="origin",
            remote_url="git@github.com:soulsplitters-a11y/private-repo.git",
            remote_visibility="private",
            dry_run=True,
        )


def test_publish_allows_when_decision_bundle_passed(repo_with_branch: Path, monkeypatch):
    monkeypatch.setattr(minos_publish, "resolve_github_visibility", lambda *_args, **_kwargs: "private")
    _write_decision(
        repo_with_branch,
        {
            "run_id": "run-001",
            "builder_status": "passed",
            "gate_status": "passed",
            "next_action": "publish",
            "decision_reason": "passed",
            "publish_recommended": True,
            "retry_recommended": False,
            "human_review_required": False,
            "blocking_findings": [],
        },
    )

    result = minos_publish.publish_run(
        repo_path=repo_with_branch,
        run_id="run-001",
        remote_name="origin",
        remote_url="git@github.com:soulsplitters-a11y/private-repo.git",
        remote_visibility="private",
        dry_run=True,
    )

    assert result["allowed"] is True
    assert result["branch_name"] == "minos/run-001"
    assert result["remote_name"] == "origin"


def test_publish_rejects_public_github_destination(repo_with_branch: Path, monkeypatch):
    monkeypatch.setattr(minos_publish, "resolve_github_visibility", lambda *_args, **_kwargs: "public")
    _write_decision(
        repo_with_branch,
        {
            "run_id": "run-001",
            "builder_status": "passed",
            "gate_status": "passed",
            "next_action": "publish",
            "decision_reason": "passed",
            "publish_recommended": True,
            "retry_recommended": False,
            "human_review_required": False,
            "blocking_findings": [],
        },
    )

    with pytest.raises(RuntimeError, match="public GitHub"):
        minos_publish.publish_run(
            repo_path=repo_with_branch,
            run_id="run-001",
            remote_name="origin",
            remote_url="git@github.com:soulsplitters-a11y/public-repo.git",
            remote_visibility="public",
            dry_run=True,
        )


def test_publish_rejects_branch_mismatch(repo_with_branch: Path, monkeypatch):
    monkeypatch.setattr(minos_publish, "resolve_github_visibility", lambda *_args, **_kwargs: "private")
    _write_decision(
        repo_with_branch,
        {
            "run_id": "run-001",
            "builder_status": "passed",
            "gate_status": "passed",
            "next_action": "publish",
            "decision_reason": "passed",
            "publish_recommended": True,
            "retry_recommended": False,
            "human_review_required": False,
            "blocking_findings": [],
        },
    )
    _git(repo_with_branch, "checkout", "-b", "some-other-branch")

    with pytest.raises(RuntimeError, match="approved branch"):
        minos_publish.publish_run(
            repo_path=repo_with_branch,
            run_id="run-001",
            remote_name="origin",
            remote_url="git@github.com:soulsplitters-a11y/private-repo.git",
            remote_visibility="private",
            dry_run=True,
        )

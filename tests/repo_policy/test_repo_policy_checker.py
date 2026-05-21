from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agent.repo_policy import check_repo_policy

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "repo-policy"
SCRIPT = ROOT / "scripts" / "check_repo_policy.py"


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    run(["git", "init", "-b", "main"], repo)
    run(["git", "config", "user.email", "repo-policy-test@example.invalid"], repo)
    run(["git", "config", "user.name", "Repo Policy Test"], repo)
    (repo / "README.md").write_text("test repo\n", encoding="utf-8")
    run(["git", "add", "README.md"], repo)
    run(["git", "commit", "-m", "init"], repo)
    return repo


def install_policy(repo: Path, example_name: str) -> None:
    policy_dir = repo / ".hermes"
    policy_dir.mkdir()
    shutil.copy2(EXAMPLES / example_name, policy_dir / "repo-policy.yaml")


def issue_codes(result: dict) -> set[str]:
    return {issue["code"] for issue in result["issues"]}


def test_product_policy_passes_when_develop_is_observed(git_repo: Path) -> None:
    run(["git", "branch", "develop"], git_repo)
    install_policy(git_repo, "product-develop.yaml")

    result = check_repo_policy(git_repo).as_dict()

    assert result["ok"] is True
    assert result["status"] == "pass"
    assert result["repo_class"] == "product"
    assert result["observed"]["has_develop"] is True
    assert result["authority"]["workflow"]["work_done_means"] == "pushed_to_develop"
    assert result["authority"]["workflow"]["release_path"] == "develop->main"
    assert result["authority"]["green_only"] is True
    assert result["authority"]["red_still_requires_explicit_approval"] is True


def test_product_policy_allows_master_release_target_when_develop_is_observed(git_repo: Path) -> None:
    run(["git", "branch", "develop"], git_repo)
    install_policy(git_repo, "product-develop-master.yaml")

    result = check_repo_policy(git_repo).as_dict()

    assert result["ok"] is True
    assert result["status"] == "pass"
    assert result["repo_class"] == "product"
    assert result["authority"]["workflow"]["release_path"] == "develop->master"


def test_product_policy_fails_when_release_base_mismatches_target(git_repo: Path) -> None:
    run(["git", "branch", "develop"], git_repo)
    install_policy(git_repo, "product-develop-master.yaml")
    policy_path = git_repo / ".hermes" / "repo-policy.yaml"
    text = policy_path.read_text(encoding="utf-8")
    policy_path.write_text(text.replace("release_base: master", "release_base: main"), encoding="utf-8")

    result = check_repo_policy(git_repo).as_dict()

    assert result["ok"] is False
    assert "product_release_base_mismatch" in issue_codes(result)


def test_product_policy_fails_when_release_target_is_not_allowed(git_repo: Path) -> None:
    run(["git", "branch", "develop"], git_repo)
    install_policy(git_repo, "product-develop.yaml")
    policy_path = git_repo / ".hermes" / "repo-policy.yaml"
    text = policy_path.read_text(encoding="utf-8")
    policy_path.write_text(text.replace("release_target: main", "release_target: production"), encoding="utf-8")

    result = check_repo_policy(git_repo).as_dict()

    assert result["ok"] is False
    assert "product_release_path_not_develop_to_release_target" in issue_codes(result)


def test_product_policy_fails_when_workflow_does_not_define_develop_landing(git_repo: Path) -> None:
    run(["git", "branch", "develop"], git_repo)
    install_policy(git_repo, "product-develop.yaml")
    policy_path = git_repo / ".hermes" / "repo-policy.yaml"
    text = policy_path.read_text(encoding="utf-8")
    policy_path.write_text(text.replace("work_done_means: pushed_to_develop", "work_done_means: local_commit"), encoding="utf-8")

    result = check_repo_policy(git_repo).as_dict()

    assert result["ok"] is False
    assert "product_work_done_not_develop" in issue_codes(result)


def test_missing_policy_fails_closed(git_repo: Path) -> None:
    result = check_repo_policy(git_repo).as_dict()

    assert result["ok"] is False
    assert result["status"] == "fail_closed"
    assert "missing_policy" in issue_codes(result)
    assert result["authority"]["external_effects_allowed"] is False


def test_malformed_yaml_fails_closed(git_repo: Path) -> None:
    policy_dir = git_repo / ".hermes"
    policy_dir.mkdir()
    (policy_dir / "repo-policy.yaml").write_text("version: [unterminated\n", encoding="utf-8")

    result = check_repo_policy(git_repo).as_dict()

    assert result["ok"] is False
    assert "malformed_yaml" in issue_codes(result)


def test_unsupported_version_fails_closed(git_repo: Path) -> None:
    run(["git", "branch", "develop"], git_repo)
    install_policy(git_repo, "invalid-unsupported-version.yaml")

    result = check_repo_policy(git_repo).as_dict()

    assert result["ok"] is False
    assert "unsupported_version" in issue_codes(result)


def test_product_missing_develop_fails_closed(git_repo: Path) -> None:
    install_policy(git_repo, "product-develop.yaml")

    result = check_repo_policy(git_repo).as_dict()

    assert result["ok"] is False
    assert "product_develop_not_observed" in issue_codes(result)


def test_runtime_tooling_policy_passes_without_develop_requirement(git_repo: Path) -> None:
    install_policy(git_repo, "runtime-tooling-hermes-agent.yaml")

    result = check_repo_policy(git_repo).as_dict()

    assert result["ok"] is True
    assert result["repo_class"] == "runtime_tooling"
    assert result["authority"]["red_still_requires_explicit_approval"] is True


def test_runtime_tooling_missing_live_queue_fails_closed(git_repo: Path) -> None:
    install_policy(git_repo, "invalid-runtime-missing-queue.yaml")

    result = check_repo_policy(git_repo).as_dict()

    assert result["ok"] is False
    assert "runtime_live_apply_missing_queue" in issue_codes(result)
    assert "runtime_queue_not_required" in issue_codes(result)
    assert "runtime_restart_policy_missing" in issue_codes(result)


def test_agents_conflict_fails_closed(git_repo: Path) -> None:
    run(["git", "branch", "develop"], git_repo)
    install_policy(git_repo, "product-develop.yaml")
    (git_repo / "AGENTS.md").write_text("Always push directly to main.\n", encoding="utf-8")

    result = check_repo_policy(git_repo).as_dict()

    assert result["ok"] is False
    assert "agents_pointer_conflict" in issue_codes(result)


def test_cli_json_exit_codes(git_repo: Path) -> None:
    missing = subprocess.run(
        ["python", str(SCRIPT), str(git_repo), "--json"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert missing.returncode == 2
    assert json.loads(missing.stdout)["status"] == "fail_closed"

    run(["git", "branch", "develop"], git_repo)
    install_policy(git_repo, "product-develop.yaml")
    passing = subprocess.run(
        ["python", str(SCRIPT), str(git_repo), "--json"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert passing.returncode == 0
    assert json.loads(passing.stdout)["status"] == "pass"

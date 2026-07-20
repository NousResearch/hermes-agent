"""End-to-end contracts for the read-only merge coverage checker."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "history_coverage_check.py"


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
    )


def _commit(repo: Path, path: str, content: str, message: str) -> str:
    (repo / path).write_text(content, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _history_repo(tmp_path: Path) -> tuple[Path, str, str, str]:
    repo = tmp_path / "history-repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "coverage@example.invalid")
    _git(repo, "config", "user.name", "Coverage")
    initial = _commit(repo, "README.md", "base\n", "base")

    _git(repo, "checkout", "-b", "source")
    source = _commit(repo, "feature.txt", "source feature\n", "source feature")

    _git(repo, "checkout", "-b", "integration", initial)
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "cherry-pick", source)
    post = _git(repo, "rev-parse", "HEAD").stdout.strip()
    return repo, pre, post, source


def _run_checker(repo: Path, *args: str) -> tuple[int, dict]:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--format",
            "json",
            "--strict",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    return result.returncode, json.loads(result.stdout)


def _write_manifest(path: Path, **overrides: object) -> Path:
    manifest = {
        "version": 1,
        "candidate_commits": [],
        "manual_port_commits": [],
        "candidate_patches": [],
        "policy_tests": [],
        "allowed_target_paths": [],
    }
    manifest.update(overrides)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_exact_cherry_pick_is_reported_as_introduced_coverage(tmp_path: Path):
    repo, pre, post, source = _history_repo(tmp_path)

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--candidate-commit",
        source,
    )

    assert code == 0
    assert report["ok"] is True
    assert report["checks"]["ancestry"]["status"] == "passed"
    assert report["candidates"][0]["status"] == "introduced_exact"


def test_synthetic_merge_result_is_checked_from_advanced_base_not_stale_head(tmp_path: Path):
    repo, initial, _post, source = _history_repo(tmp_path)
    _git(repo, "checkout", "-B", "base", initial)
    _commit(repo, "base-only.txt", "new base\n", "base advance")
    pre = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "merge", "--no-ff", "source", "-m", "synthetic PR merge")
    post = _git(repo, "rev-parse", "HEAD").stdout.strip()

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--candidate-commit",
        source,
    )

    assert code == 0
    assert report["ok"] is True
    assert report["checks"]["ancestry"]["status"] == "passed"
    assert report["candidates"][0]["status"] == "introduced_exact"


def test_applied_then_fully_reverted_commit_fails_even_when_patch_id_appears_in_range(tmp_path: Path):
    repo, pre, post, source = _history_repo(tmp_path)
    _git(repo, "revert", "--no-edit", post)
    post = _git(repo, "rev-parse", "HEAD").stdout.strip()

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--candidate-commit",
        source,
    )

    assert code == 1
    assert report["ok"] is False
    assert report["candidates"][0]["effect_present_at_post"] is False
    assert report["candidates"][0]["status"] == "reverted_in_target"
    assert report["summary"]["reverted_in_target"] == 1


def test_exact_candidate_with_later_same_path_mutation_requires_manual_review(tmp_path: Path):
    repo, pre, post, source = _history_repo(tmp_path)
    _commit(repo, "feature.txt", "different later implementation\n", "later transformation")
    post = _git(repo, "rev-parse", "HEAD").stdout.strip()

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--candidate-commit",
        source,
    )

    assert code == 1
    assert report["candidates"][0]["status"] == "manual_review_required"
    assert report["candidates"][0]["post_source_path_mutations"] == ["feature.txt"]


def test_exact_patch_file_candidate_remains_manual_review_required(tmp_path: Path):
    repo, pre, post, source = _history_repo(tmp_path)
    patch_path = tmp_path / "source.patch"
    patch_path.write_text(_git(repo, "show", "--format=", source).stdout, encoding="utf-8")

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--candidate-patch",
        str(patch_path),
    )

    assert code == 1
    assert report["candidates"][0]["status"] == "manual_review_required"


def test_missing_candidate_is_fail_closed(tmp_path: Path):
    repo, pre, _post, source = _history_repo(tmp_path)
    _git(repo, "checkout", "-B", "missing", pre)
    _commit(repo, "other.txt", "unrelated\n", "unrelated")
    post = _git(repo, "rev-parse", "HEAD").stdout.strip()

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--candidate-commit",
        source,
    )

    assert code == 1
    assert report["ok"] is False
    assert report["candidates"][0]["status"] == "missing"


def test_preexisting_ancestor_is_not_misreported_as_new_merge_coverage(tmp_path: Path):
    repo, _pre, _post, source = _history_repo(tmp_path)
    _git(repo, "checkout", "source")
    pre = source
    post = _commit(repo, "later.txt", "later\n", "later")

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--candidate-commit",
        source,
    )

    assert code == 0
    assert report["ok"] is True
    assert report["candidates"][0]["status"] == "preexisting_ancestry"
    assert report["summary"]["introduced_exact"] == 0
    assert report["summary"]["preexisting"] == 1


def test_added_conflict_markers_fail_without_rejecting_plain_separator_lines(tmp_path: Path):
    repo, pre, post, source = _history_repo(tmp_path)
    _commit(repo, "markers.txt", "======= valid separator\n<<<<<<< conflict\n>>>>>>> branch\n", "markers")
    post = _git(repo, "rev-parse", "HEAD").stdout.strip()

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--candidate-commit",
        source,
    )

    assert code == 1
    assert report["checks"]["conflict_markers"]["status"] == "failed"
    assert report["checks"]["conflict_markers"]["count"] == 2


def test_whitespace_error_fails_diff_check(tmp_path: Path):
    repo, pre, post, source = _history_repo(tmp_path)
    _commit(repo, "whitespace.txt", "trailing space  \n", "whitespace")
    post = _git(repo, "rev-parse", "HEAD").stdout.strip()

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--candidate-commit",
        source,
    )

    assert code == 1
    assert report["checks"]["diff_check"]["status"] == "failed"


def test_transformed_patch_requires_manual_review(tmp_path: Path):
    repo, pre, post, source = _history_repo(tmp_path)
    patch = _git(repo, "show", "--format=", source).stdout
    patch_path = tmp_path / "source.patch"
    patch_path.write_text(patch, encoding="utf-8")
    _git(repo, "checkout", "-B", "manual-port", pre)
    _commit(repo, "feature.txt", "manually ported but different\n", "manual port")
    post = _git(repo, "rev-parse", "HEAD").stdout.strip()

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--candidate-patch",
        str(patch_path),
    )

    assert code == 1
    assert report["candidates"][0]["status"] == "manual_review_required"


def test_explicit_manual_port_commit_requires_review_instead_of_claiming_missing(tmp_path: Path):
    repo, pre, _post, source = _history_repo(tmp_path)
    _git(repo, "checkout", "-B", "manual-port", pre)
    _commit(repo, "feature.txt", "manually ported but different\n", "manual port")
    post = _git(repo, "rev-parse", "HEAD").stdout.strip()

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--manual-port-commit",
        source,
    )

    assert code == 1
    assert report["candidates"][0]["status"] == "manual_review_required"


def test_manual_port_declaration_overrides_exact_cherry_pick_automation(tmp_path: Path):
    repo, pre, post, source = _history_repo(tmp_path)

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--manual-port-commit",
        source,
    )

    assert code == 1
    assert report["candidates"][0]["status"] == "manual_review_required"


def test_strict_paths_fail_closed_for_unexpected_target_changes_and_allow_exact_exceptions(tmp_path: Path):
    repo, pre, post, source = _history_repo(tmp_path)
    _commit(repo, "unrelated.txt", "unrelated\n", "unrelated")
    post = _git(repo, "rev-parse", "HEAD").stdout.strip()

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--candidate-commit",
        source,
        "--strict-paths",
    )

    assert code == 1
    assert report["checks"]["target_scope"]["unexpected_paths"] == ["unrelated.txt"]

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--candidate-commit",
        source,
        "--strict-paths",
        "--allow-target-path",
        "unrelated.txt",
    )

    assert code == 0
    assert report["checks"]["target_scope"]["status"] == "passed"


def test_manifest_supplies_only_versioned_candidate_and_allowlisted_policy_inputs(tmp_path: Path):
    repo, pre, post, source = _history_repo(tmp_path)
    manifest = _write_manifest(
        tmp_path / "recovery-coverage.json",
        candidate_commits=[source],
        policy_tests=[],
    )

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--manifest",
        str(manifest),
    )

    assert code == 0
    assert report["manifest"] == str(manifest.resolve())
    assert report["candidates"][0]["status"] == "introduced_exact"


def test_manifest_manual_port_declaration_overrides_exact_cherry_pick_automation(tmp_path: Path):
    repo, pre, post, source = _history_repo(tmp_path)
    manifest = _write_manifest(tmp_path / "recovery-coverage.json", manual_port_commits=[source])

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--manifest",
        str(manifest),
    )

    assert code == 1
    assert report["candidates"][0]["status"] == "manual_review_required"


def test_manifest_unknown_policy_id_fails_closed_without_executing_a_custom_command(tmp_path: Path):
    repo, pre, post, source = _history_repo(tmp_path)
    manifest = _write_manifest(
        tmp_path / "recovery-coverage.json",
        candidate_commits=[source],
        policy_tests=["not-an-allowed-policy"],
    )

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--manifest",
        str(manifest),
    )

    assert code == 1
    assert report["ok"] is False
    assert report["policy_tests"] == [{"id": "not-an-allowed-policy", "status": "invalid_policy_id"}]


def test_manifest_patch_may_not_escape_its_contract_directory(tmp_path: Path):
    repo, pre, post, _source = _history_repo(tmp_path)
    contract_dir = tmp_path / "contract"
    contract_dir.mkdir()
    manifest = _write_manifest(contract_dir / "recovery-coverage.json", candidate_patches=["../outside.patch"])

    code, report = _run_checker(
        repo,
        "--pre-target",
        pre,
        "--post-target",
        post,
        "--manifest",
        str(manifest),
    )

    assert code == 1
    assert report["ok"] is False
    assert "must stay within the manifest directory" in report["error"]

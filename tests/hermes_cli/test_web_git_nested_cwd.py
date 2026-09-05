"""Regression coverage for Review invoked from a repository subdirectory."""

import subprocess
from pathlib import Path

import pytest

from hermes_cli import web_git


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def nested_repo(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    sub = root / "sub"
    sub.mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    (root / "tracked.txt").write_text("tracked\n")
    (sub / "nested.txt").write_text("nested\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "initial")
    (root / "tracked.txt").write_text("tracked changed\n")
    (sub / "nested.txt").write_text("nested changed\n")
    return root, sub


def test_review_list_and_diff_from_nested_cwd(nested_repo):
    _root, sub = nested_repo

    listed = web_git.review_list(str(sub), "uncommitted", None)

    assert [item["path"] for item in listed["files"]] == ["sub/nested.txt", "tracked.txt"]
    assert "tracked changed" in web_git.review_diff(
        str(sub), "tracked.txt", "uncommitted", None, False
    )


def test_review_stage_and_unstage_from_nested_cwd(nested_repo):
    root, sub = nested_repo

    web_git.review_stage(str(sub), "tracked.txt")
    assert _git(root, "diff", "--cached", "--name-only").strip() == "tracked.txt"

    web_git.review_unstage(str(sub), "tracked.txt")
    assert _git(root, "diff", "--cached", "--name-only").strip() == ""


def test_review_revert_from_nested_cwd(nested_repo):
    root, sub = nested_repo

    web_git.review_revert(str(sub), "tracked.txt")

    assert (root / "tracked.txt").read_text() == "tracked\n"
    assert (sub / "nested.txt").read_text() == "nested changed\n"

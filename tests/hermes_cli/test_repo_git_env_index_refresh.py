"""Read-only git views must not store an index refresh computed with the repository's filters emptied.

A file committed raw before the repository adopted a clean filter differs from HEAD only through that
filter. While it is stat-dirty the user's own ``git status`` lists it; a view that wrote back the
unfiltered refresh would hide it (and the dashboard's ``add -A`` would then skip it)."""

import os
import subprocess
from pathlib import Path

import pytest

_ENV = {"PATH": os.environ["PATH"], "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env={**_ENV, "HOME": str(repo.parent)}).stdout


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "T")
    _git(repo, "config", "user.email", "t@e.invalid")
    (repo / "README").write_text("hello\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    (repo / ".gitattributes").write_text("README filter=up\n")
    _git(repo, "config", "filter.up.clean", "tr a-z A-Z")
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-q", "-m", "attrs")
    os.utime(repo / "README", (1_000_000_000, 1_000_000_000))
    assert "README" in _git(repo, "status", "--porcelain")
    os.utime(repo / "README", (1_000_000_100, 1_000_000_100))
    return repo


def test_dashboard_status(repo):
    from hermes_cli import web_git
    web_git.repo_status(str(repo))
    assert "README" in _git(repo, "status", "--porcelain")


def test_dashboard_review_list(repo):
    from hermes_cli import web_git
    web_git.review_list(str(repo), "working", None)
    assert "README" in _git(repo, "status", "--porcelain")


def test_slash_diff(repo):
    from hermes_agent.tools.working_diff import collect_working_diff
    collect_working_diff(str(repo))
    assert "README" in _git(repo, "status", "--porcelain")


def test_at_diff(repo):
    from agent.context_references import preprocess_context_references
    preprocess_context_references("look @diff", cwd=repo, context_length=100_000)
    assert "README" in _git(repo, "status", "--porcelain")

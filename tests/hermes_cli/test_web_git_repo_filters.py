"""Dashboard read-only git views empty the clean/smudge filter commands a repository names.

Real temporary repositories with real ``git``: ``.gitattributes`` names a filter whose
repository-local command touches a marker file and then behaves as ``cat``, and the tracked files
are stat-dirty. Read views must leave the marker absent and still return their normal content;
mutating operations (staging) keep today's environment, so the filter still runs there.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_agent.hermes_cli import web_git
from hermes_agent.hermes_cli._subprocess_compat import FILTER_DISCOVERY_FAILED

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")

_STAMP = [1_000_000_000]


def _git(repo: Path, *args: str) -> str:
    env = {"HOME": str(repo.parent), "PATH": os.environ["PATH"], "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_CONFIG_GLOBAL": os.devnull}
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True, env=env).stdout


def _make_repo(tmp_path: Path, *, with_filter: bool = True, via_include: bool = False,
               stage: bool = False) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "README").write_text("hello\n")
    (repo / "notes.txt").write_text("alpha\n")
    if with_filter:
        (repo / ".gitattributes").write_text("README filter=probe\n*.txt filter=probe\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    (repo / "notes.txt").write_text("alpha\nbeta\n")
    if stage:
        _git(repo, "add", "notes.txt")
    marker = tmp_path / "filter-ran"
    if with_filter:
        script = tmp_path / "probe-filter.sh"
        script.write_text(f"#!/bin/sh\ntouch '{marker}'\nexec cat\n")
        script.chmod(0o755)
        if via_include:
            (repo / ".git" / "probe-filters.inc").write_text(
                f'[filter "probe"]\n\tclean = {script}\n\tsmudge = {script}\n')
            _git(repo, "config", "include.path", "probe-filters.inc")
        else:
            _git(repo, "config", "filter.probe.clean", str(script))
            _git(repo, "config", "filter.probe.smudge", str(script))
    _stat_dirty(repo, marker)
    return repo, marker


def _stat_dirty(repo: Path, marker: Path) -> None:
    _STAMP[0] += 10
    for name in ("README", "notes.txt"):
        os.utime(repo / name, (_STAMP[0], _STAMP[0]))
    if marker.exists():
        marker.unlink()


def test_repo_status_runs_no_repository_filter(tmp_path):
    repo, marker = _make_repo(tmp_path)

    status = web_git.repo_status(str(repo))

    assert not marker.exists()
    assert status is not None
    assert status["branch"] == "main"
    assert [f["path"] for f in status["files"]] == ["notes.txt"]
    assert (status["added"], status["removed"]) == (1, 0)


@pytest.mark.parametrize("scope", ["working", "lastTurn"])
def test_review_list_runs_no_repository_filter(tmp_path, scope):
    repo, marker = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip() if scope == "lastTurn" else None
    _stat_dirty(repo, marker)

    listing = web_git.review_list(str(repo), scope, base)

    assert not marker.exists()
    assert [(f["path"], f["added"], f["removed"]) for f in listing["files"]] == [("notes.txt", 1, 0)]


@pytest.mark.parametrize("scope", ["working", "lastTurn"])
def test_review_diff_runs_no_repository_filter(tmp_path, scope):
    repo, marker = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip() if scope == "lastTurn" else None
    _stat_dirty(repo, marker)

    diff = web_git.review_diff(str(repo), "notes.txt", scope, base, False)

    assert not marker.exists()
    assert "+beta" in diff


def test_review_diff_of_staged_file_runs_no_repository_filter(tmp_path):
    repo, marker = _make_repo(tmp_path, stage=True)

    diff = web_git.review_diff(str(repo), "notes.txt", "working", None, True)

    assert not marker.exists()
    assert "+beta" in diff


def test_file_diff_vs_head_runs_no_repository_filter(tmp_path):
    repo, marker = _make_repo(tmp_path)

    diff = web_git.file_diff_vs_head(str(repo), "notes.txt")

    assert not marker.exists()
    assert "+beta" in diff


@pytest.mark.parametrize("stage", [False, True])
def test_review_commit_context_runs_no_repository_filter(tmp_path, stage):
    repo, marker = _make_repo(tmp_path, stage=stage)

    context = web_git.review_commit_context(str(repo))

    assert not marker.exists()
    assert "+beta" in context["diff"]
    assert context["recent"] == "init"


def test_review_commit_context_with_path_filter_runs_no_repository_filter(tmp_path):
    repo, marker = _make_repo(tmp_path)

    context = web_git.review_commit_context(str(repo), lambda path: True)

    assert not marker.exists()
    assert "+beta" in context["diff"]


def test_filter_defined_through_include_path_is_neutralized(tmp_path):
    repo, marker = _make_repo(tmp_path, via_include=True)

    status = web_git.repo_status(str(repo))

    assert not marker.exists()
    assert status is not None and [f["path"] for f in status["files"]] == ["notes.txt"]


def test_repository_without_filter_returns_the_same_views(tmp_path):
    repo, _marker = _make_repo(tmp_path, with_filter=False)
    (repo / "new.txt").write_text("one\ntwo\n")

    status = web_git.repo_status(str(repo))
    listing = web_git.review_list(str(repo), "working", None)
    diff = web_git.review_diff(str(repo), "notes.txt", "working", None, False)
    context = web_git.review_commit_context(str(repo))

    assert status is not None
    assert sorted(f["path"] for f in status["files"]) == ["new.txt", "notes.txt"]
    assert (status["added"], status["removed"], status["untracked"]) == (3, 0, 1)
    assert [(f["path"], f["status"], f["added"]) for f in listing["files"]] == [
        ("new.txt", "?", 2), ("notes.txt", "M", 1)]
    assert "-alpha" not in diff and "+beta" in diff
    assert "+beta" in context["diff"] and "#   new.txt" in context["diff"]
    assert context["recent"] == "init"


def test_staging_keeps_the_repository_filter_active_for_mutations(tmp_path):
    repo, marker = _make_repo(tmp_path)

    assert web_git.review_stage(str(repo), "notes.txt") == {"ok": True}

    assert marker.exists()
    assert _git(repo, "diff", "--cached", "--name-only").split() == ["notes.txt"]


def test_unbuildable_repository_environment_refuses_reads_without_spawning_git(tmp_path, monkeypatch):
    repo, marker = _make_repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD").strip()
    _stat_dirty(repo, marker)
    spawned: list[list[str]] = []
    real_run = subprocess.run

    def counting_run(argv, *args, **kwargs):
        spawned.append(list(argv))
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(web_git, "noninteractive_repo_git_env", lambda cwd, base=None: None, raising=False)
    monkeypatch.setattr(web_git.subprocess, "run", counting_run)
    cwd = str(repo)

    assert web_git._git(cwd, ["status", "--porcelain=v2", "-z"]) == (1, "", FILTER_DISCOVERY_FAILED)
    assert web_git.repo_status(cwd) is None
    assert web_git.review_list(cwd, "working", None) == {"files": [], "base": None}
    assert web_git.review_list(cwd, "lastTurn", base) == {"files": [], "base": base}
    assert web_git.review_diff(cwd, "notes.txt", "working", None, False) == ""
    assert web_git.file_diff_vs_head(cwd, "notes.txt") == ""
    assert web_git.review_commit_context(cwd) == {"diff": "", "recent": ""}
    assert not [argv for argv in spawned if argv[:1] == ["git"]]
    assert not marker.exists()

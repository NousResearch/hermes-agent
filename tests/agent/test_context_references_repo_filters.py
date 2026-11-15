"""``@diff`` / ``@staged`` / ``@git:N`` empty the clean/smudge filter commands a repository names.

Real temporary repositories with real ``git``: ``.gitattributes`` names a filter whose
repository-local command touches a marker file and then behaves as ``cat``, and a tracked file
is stat-dirty. Expansion goes through ``preprocess_context_references_async``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agent import context_references
from agent.context_references import preprocess_context_references_async
from hermes_cli._subprocess_compat import FILTER_DISCOVERY_FAILED

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")

_STAMP = [1_000_000_000]


def _git(repo: Path, *args: str) -> None:
    env = {"HOME": str(repo.parent), "PATH": os.environ["PATH"], "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_CONFIG_GLOBAL": os.devnull}
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=env)


def _make_repo(tmp_path: Path, *, with_filter: bool = True, via_include: bool = False) -> tuple[Path, Path]:
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
    (repo / "notes.txt").write_text("alpha\nstaged-line\n")
    _git(repo, "add", "notes.txt")
    _git(repo, "commit", "-q", "-m", "second")
    (repo / "notes.txt").write_text("alpha\nstaged-line\nstaged-two\n")
    _git(repo, "add", "notes.txt")
    (repo / "notes.txt").write_text("alpha\nstaged-line\nstaged-two\nworktree-line\n")
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
    _STAMP[0] += 10
    for name in ("README", "notes.txt"):
        os.utime(repo / name, (_STAMP[0], _STAMP[0]))
    if marker.exists():
        marker.unlink()
    return repo, marker


async def _expand(repo: Path, message: str):
    return await preprocess_context_references_async(message, cwd=repo, context_length=1_000_000)


@pytest.mark.asyncio
async def test_diff_reference_runs_no_repository_filter(tmp_path):
    repo, marker = _make_repo(tmp_path)

    result = await _expand(repo, "review @diff")

    assert not marker.exists()
    assert result.warnings == []
    assert "+worktree-line" in result.message
    assert "git diff" in result.message


@pytest.mark.asyncio
async def test_staged_reference_runs_no_repository_filter(tmp_path):
    repo, marker = _make_repo(tmp_path)

    result = await _expand(repo, "review @staged")

    assert not marker.exists()
    assert result.warnings == []
    assert "+staged-two" in result.message
    assert "+worktree-line" not in result.message


@pytest.mark.asyncio
async def test_git_log_reference_runs_no_repository_filter(tmp_path):
    repo, marker = _make_repo(tmp_path)

    result = await _expand(repo, "review @git:1")

    assert not marker.exists()
    assert result.warnings == []
    assert "+staged-line" in result.message
    assert "second" in result.message


@pytest.mark.asyncio
async def test_filter_defined_through_include_path_is_neutralized(tmp_path):
    repo, marker = _make_repo(tmp_path, via_include=True)

    result = await _expand(repo, "review @diff")

    assert not marker.exists()
    assert "+worktree-line" in result.message


@pytest.mark.asyncio
async def test_repository_without_filter_expands_the_same_content(tmp_path):
    repo, _marker = _make_repo(tmp_path, with_filter=False)

    result = await _expand(repo, "review @diff and @staged")

    assert result.warnings == []
    assert "+worktree-line" in result.message
    assert "+staged-two" in result.message
    assert "🧾 git diff (" in result.message
    assert "🧾 git diff --staged (" in result.message


@pytest.mark.asyncio
async def test_unbuildable_repository_environment_warns_without_spawning_git(tmp_path, monkeypatch):
    repo, marker = _make_repo(tmp_path)
    spawned: list[list[str]] = []
    real_popen = subprocess.Popen

    def counting_popen(argv, *args, **kwargs):
        spawned.append(list(argv))
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(context_references, "noninteractive_repo_git_env", lambda cwd, base=None: None,
                        raising=False)
    monkeypatch.setattr(context_references.subprocess, "Popen", counting_popen)

    result = await _expand(repo, "review @diff")

    assert result.warnings == [f"@diff: {FILTER_DISCOVERY_FAILED}"]
    assert "+worktree-line" not in result.message
    assert not [argv for argv in spawned if "diff" in argv]
    assert not marker.exists()

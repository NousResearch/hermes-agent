"""The e2e-upgrade suites start from the release before HEAD, on every release shape (#150040).

``tests/e2e/core/upgrade/_install_helpers.upgrade_base`` picks the release every upgrade leg starts
from. It runs here against a small temp history rather than this checkout's own tags, so PR CI
covers it: the e2e-upgrade lane runs only on release and dispatch runs.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I

_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "e2e", "GIT_AUTHOR_EMAIL": "e2e@example.invalid",
    "GIT_COMMITTER_NAME": "e2e", "GIT_COMMITTER_EMAIL": "e2e@example.invalid",
}


class _Repo:
    """A git history whose commits and tags carry increasing dates (``--sort=creatordate`` is real)."""

    def __init__(self, path: Path) -> None:
        self.path, self.clock = path, 1_790_000_000
        path.mkdir()
        self.git("init", "-q", "-b", "main")

    def git(self, *args: str) -> str:
        self.clock += 60
        stamp = f"@{self.clock} +0000"
        env = {**_ENV, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
        return subprocess.run(["git", *args], cwd=self.path, env=env, capture_output=True, text=True,
                              check=True).stdout.strip()

    def commit(self, message: str, *tags: str) -> None:
        self.git("commit", "-q", "--allow-empty", "-m", message)
        for tag in tags:
            self.git("tag", "-a", "-m", tag, tag)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Iterator[_Repo]:
    """main: a CalVer release, v0.9.0, v0.10.0 with its candidate, a canary. ``release/0.10`` forks
    at v0.10.0; v0.10.1 is published from a side branch, and v0.10.2 is cut as a merge whose first
    parent is the release branch tip and whose second parent is v0.10.1 (v0.22.2's shape)."""
    for name in ("HERMES_E2E_UPGRADE_BASE", "HERMES_E2E_N1_REF", "CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(name, raising=False)
    r = _Repo(tmp_path / "repo")
    monkeypatch.setattr(H, "WORKTREE", r.path)
    I.upgrade_base.cache_clear()
    r.commit("calver era", "v2026.9.24")
    r.commit("0.9", "v0.9.0")
    r.commit("0.10", "rc.1-v0.10.0", "v0.10.0")
    r.git("branch", "release/0.10")
    r.commit("canary", "v0.10.0+canary.20261106T064428Z")
    r.git("checkout", "-q", "-b", "publish-0.10.1", "release/0.10")
    r.commit("hotfix published from a side branch", "v0.10.1")
    r.git("checkout", "-q", "release/0.10")
    r.commit("release branch moves on")
    r.git("merge", "-q", "--no-ff", "-m", "release v0.10.2", "v0.10.1")
    r.git("tag", "-a", "-m", "v0.10.2", "v0.10.2")
    r.git("checkout", "-q", "main")
    yield r
    I.upgrade_base.cache_clear()


def test_a_release_cut_as_a_merge_starts_from_the_release_on_its_second_parent(repo):
    repo.git("checkout", "-q", "--detach", "v0.10.2")
    assert repo.git("rev-parse", "HEAD^2") == repo.git("rev-parse", "v0.10.1^{commit}")
    assert I.upgrade_base().ref == "v0.10.1"


def test_a_tagged_head_starts_from_the_release_before_it(repo):
    repo.commit("head", "v0.11.0")
    assert I.upgrade_base().ref == "v0.10.0"


def test_no_release_before_head_fails_on_ci_and_skips_locally(tmp_path, monkeypatch):
    for name in ("HERMES_E2E_UPGRADE_BASE", "HERMES_E2E_N1_REF", "CI", "GITHUB_ACTIONS"):
        monkeypatch.delenv(name, raising=False)
    r = _Repo(tmp_path / "tagless")
    r.commit("only commit", "rc.1-v0.1.0")
    r.commit("head")
    monkeypatch.setattr(H, "WORKTREE", r.path)
    I.upgrade_base.cache_clear()
    assert I.upgrade_base() is None
    I.upgrade_base.cache_clear()
    monkeypatch.setenv("CI", "true")
    with pytest.raises(AssertionError, match="no release tag reachable before HEAD"):
        I.upgrade_base()
    I.upgrade_base.cache_clear()

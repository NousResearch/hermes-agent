"""Real-git regression test: upstream sync counts from HEAD, not origin/main.

Covers the maintainer repro on #101127: a fork whose local commits were never
pushed leaves origin/main stale, so counting from origin/main reported a false
"up to date" and the rebase path was never reached. No mocks, no network —
bare local repos only.
"""
import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd_git


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


@pytest.fixture
def stale_mirror_clone(tmp_path, monkeypatch):
    """A checkout with one UNPUSHED local commit + one new upstream commit,
    where origin/main is stale (mirrors the pre-fix blind spot exactly)."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "git-config"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    for name in ("upstream.git", "fork.git"):
        git(tmp_path, "init", "-q", "--bare", "-b", "main", name)
    seed = tmp_path / "seed"
    git(tmp_path, "clone", "-q", str(tmp_path / "upstream.git"), str(seed))
    git(seed, "config", "user.name", "Fixture")
    git(seed, "config", "user.email", "fixture@example.invalid")
    git(seed, "remote", "rename", "origin", "upstream")
    git(seed, "remote", "add", "fork", str(tmp_path / "fork.git"))
    (seed / "base.txt").write_text("base\n", encoding="utf-8")
    git(seed, "add", ".")
    git(seed, "-c", "commit.gpgsign=false", "commit", "-qm", "base")
    git(seed, "push", "-q", "upstream", "main")
    git(seed, "push", "-q", "fork", "main")

    clone = tmp_path / "checkout"
    git(tmp_path, "clone", "-q", str(tmp_path / "fork.git"), str(clone))
    git(clone, "config", "user.name", "Fixture")
    git(clone, "config", "user.email", "fixture@example.invalid")
    git(clone, "remote", "add", "upstream", str(tmp_path / "upstream.git"))
    git(clone, "fetch", "-q", "upstream")
    # Local commit, NEVER pushed: origin/main goes stale from here on.
    (clone / "local.txt").write_text("local\n", encoding="utf-8")
    git(clone, "add", ".")
    git(clone, "-c", "commit.gpgsign=false", "commit", "-qm", "local work")
    # One new upstream commit the checkout has never seen.
    (seed / "upstream.txt").write_text("upstream\n", encoding="utf-8")
    git(seed, "add", ".")
    git(seed, "-c", "commit.gpgsign=false", "commit", "-qm", "upstream work")
    git(seed, "push", "-q", "upstream", "main")
    return clone


def test_unpushed_local_plus_upstream_rebases(stale_mirror_clone):
    clone = stale_mirror_clone
    # Sanity: the blind spot is real — origin/main sees no drift on either side.
    assert git(clone, "rev-list", "--count", "upstream/main..origin/main") == "0"
    assert git(clone, "rev-list", "--count", "origin/main..upstream/main") == "0"

    ok = update_cmd_git._sync_with_upstream_if_needed(
        ["git"], Path(clone), assume_yes=True)
    assert ok is True
    # Local commit survived on top of the new upstream commit; nothing behind.
    assert git(clone, "rev-list", "--count", "HEAD..upstream/main") == "0"
    log = git(clone, "log", "--format=%s", "-2")
    assert "local work" in log and "upstream work" in log


def test_fully_in_sync_reports_up_to_date(stale_mirror_clone, tmp_path):
    clone = stale_mirror_clone
    assert update_cmd_git._sync_with_upstream_if_needed(
        ["git"], Path(clone), assume_yes=True) is True
    # Second run: genuinely nothing to do, still a verified True (not a skip).
    assert update_cmd_git._sync_with_upstream_if_needed(
        ["git"], Path(clone), assume_yes=True) is True
    assert git(clone, "rev-list", "--count", "HEAD..upstream/main") == "0"

from __future__ import annotations

import subprocess
from pathlib import Path

from hermes_cli.worktree_base import resolve_worktree_base


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, filename: str, content: str) -> str:
    (repo / filename).write_text(content, encoding="utf-8")
    _git(repo, "add", filename)
    _git(repo, "commit", "-m", filename)
    return _git(repo, "rev-parse", "HEAD")


def _feature_checkout_with_remote_default(tmp_path: Path) -> tuple[Path, str, str]:
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote))

    repo = tmp_path / "repo"
    _git(tmp_path, "clone", str(remote), str(repo))
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "switch", "-c", "main")
    main_sha = _commit(repo, "main.txt", "main\n")
    _git(repo, "push", "-u", "origin", "main")
    _git(remote, "symbolic-ref", "HEAD", "refs/heads/main")

    _git(repo, "switch", "-c", "feature")
    feature_sha = _commit(repo, "feature.txt", "feature\n")
    _git(repo, "push", "-u", "origin", "feature")
    _git(repo, "remote", "set-head", "origin", "main")
    return repo, main_sha, feature_sha


def test_new_work_resolves_remote_default_not_current_feature_upstream(tmp_path: Path) -> None:
    repo, main_sha, feature_sha = _feature_checkout_with_remote_default(tmp_path)

    base_ref, label = resolve_worktree_base(
        str(repo),
        prefer_current_upstream=False,
    )

    assert base_ref == "origin/main"
    assert "origin/main" in label
    assert _git(repo, "rev-parse", base_ref) == main_sha
    assert _git(repo, "rev-parse", base_ref) != feature_sha


def test_continuation_can_prefer_current_feature_upstream(tmp_path: Path) -> None:
    repo, _main_sha, feature_sha = _feature_checkout_with_remote_default(tmp_path)

    base_ref, _label = resolve_worktree_base(
        str(repo),
        prefer_current_upstream=True,
    )

    assert base_ref == "origin/feature"
    assert _git(repo, "rev-parse", base_ref) == feature_sha


def test_unrelated_fetch_does_not_fake_freshness_for_the_selected_ref(tmp_path: Path) -> None:
    """FETCH_HEAD is repo-wide: fetching a DIFFERENT branch must not make a stale
    origin/main look freshly-fetched just because it shares the file's recent mtime."""
    repo, stale_main_sha, _feature_sha = _feature_checkout_with_remote_default(tmp_path)

    # origin/main advances upstream after our clone/push above...
    other_clone = tmp_path / "other_clone"
    _git(tmp_path, "clone", str(tmp_path / "remote.git"), str(other_clone))
    _git(other_clone, "config", "user.email", "test@example.com")
    _git(other_clone, "config", "user.name", "Test")
    _git(other_clone, "checkout", "main")
    new_main_sha = _commit(other_clone, "main2.txt", "main2\n")
    _git(other_clone, "push", "origin", "main")

    # ...and our local repo's FETCH_HEAD gets a fresh mtime from fetching a
    # DIFFERENT branch (feature), never touching main.
    _git(repo, "fetch", "origin", "feature")
    assert _git(repo, "rev-parse", "origin/main") == stale_main_sha

    base_ref, label = resolve_worktree_base(
        str(repo),
        prefer_current_upstream=False,
    )

    assert base_ref == "origin/main"
    assert _git(repo, "rev-parse", base_ref) == new_main_sha
    # A real network fetch happened rather than trusting the unrelated FETCH_HEAD mtime.
    assert label == "origin/main (fetched)"

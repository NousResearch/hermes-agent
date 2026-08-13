from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _commit(repo: Path, message: str, content: str) -> str:
    (repo / "tracked.txt").write_text(content, encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "hermes@example.invalid")
    return repo


@pytest.mark.parametrize(
    "value",
    [
        "*",
        "release;touch",
        "refs/tags/v2026.7.30",
        "backup/pre-release",
        "v2026.07.30",
        "v2026.7",
        "2026.7.30",
    ],
)
def test_official_release_tag_rejects_non_release_values(value):
    with pytest.raises(ValueError):
        update_cmd._official_release_tag(value)


@pytest.mark.parametrize(
    "value",
    ["v2026.7.30", "v2026.7.7.2", "v2030.12.31"],
)
def test_official_release_tag_accepts_release_shape(value):
    assert update_cmd._official_release_tag(value) == value


def test_fetch_official_release_is_exact_and_does_not_auto_follow_tags(
    monkeypatch, tmp_path
):
    source = _repo(tmp_path)
    first = _commit(source, "first", "first")
    _git(source, "tag", "-a", "v2026.7.1", "-m", "first release", first)
    second = _commit(source, "second", "second")
    _git(source, "tag", "-a", "v2026.7.2", "-m", "second release", second)

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "-b", "main")
    monkeypatch.setattr(update_cmd, "OFFICIAL_REPO_URL", str(source))

    result = update_cmd._fetch_official_release_tag(["git"], checkout, "v2026.7.2")

    assert result.returncode == 0
    assert _git(checkout, "tag", "--list").stdout.splitlines() == ["v2026.7.2"]
    assert update_cmd._resolve_release_commit(["git"], checkout, "v2026.7.2") == second


def test_official_fetch_replaces_counterfeit_local_release_tag(monkeypatch, tmp_path):
    source = _repo(tmp_path)
    official_sha = _commit(source, "official", "official")
    _git(source, "tag", "v2026.8.1", official_sha)

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "-b", "main")
    _git(checkout, "config", "user.name", "Hermes Test")
    _git(checkout, "config", "user.email", "hermes@example.invalid")
    counterfeit_sha = _commit(checkout, "counterfeit", "counterfeit")
    _git(checkout, "tag", "v2026.8.1", counterfeit_sha)
    monkeypatch.setattr(update_cmd, "OFFICIAL_REPO_URL", str(source))

    result = update_cmd._fetch_official_release_tag(["git"], checkout, "v2026.8.1")

    assert result.returncode == 0
    assert (
        update_cmd._resolve_release_commit(["git"], checkout, "v2026.8.1")
        == official_sha
    )


def test_non_commit_release_tag_is_rejected_before_checkout(monkeypatch, tmp_path):
    source = _repo(tmp_path)
    _commit(source, "first", "first")
    tree = _git(source, "rev-parse", "HEAD^{tree}").stdout.strip()
    _git(source, "tag", "-a", "v2026.7.30", "-m", "tree tag", tree)

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "-b", "main")
    monkeypatch.setattr(update_cmd, "OFFICIAL_REPO_URL", str(source))

    fetch = update_cmd._fetch_official_release_tag(["git"], checkout, "v2026.7.30")
    assert fetch.returncode == 0
    assert update_cmd._resolve_release_commit(["git"], checkout, "v2026.7.30") is None


def test_attached_checkout_at_release_commit_still_requires_detach():
    assert update_cmd._release_checkout_required(
        current_branch="main", head_sha="same", release_sha="same"
    )
    assert not update_cmd._release_checkout_required(
        current_branch="HEAD", head_sha="same", release_sha="same"
    )


@pytest.mark.parametrize("start_detached", [False, True])
def test_restore_checkout_identity_preserves_attached_or_detached_start(
    tmp_path, start_detached
):
    repo = _repo(tmp_path)
    start_sha = _commit(repo, "start", "start")
    start_branch = "HEAD" if start_detached else "main"
    if start_detached:
        _git(repo, "checkout", "--detach", start_sha)

    other_sha = _commit(repo, "other", "other") if not start_detached else start_sha
    if start_detached:
        _git(repo, "checkout", "main")
        other_sha = _commit(repo, "other", "other")
    _git(repo, "checkout", "--detach", other_sha)

    assert update_cmd._restore_checkout_identity(["git"], repo, start_branch, start_sha)
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == start_sha
    assert (
        _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == start_branch
    )


def test_failed_release_checkout_restores_dirty_worktree_and_branch(tmp_path):
    repo = _repo(tmp_path)
    start_sha = _commit(repo, "start", "clean")
    (repo / "tracked.txt").write_text("dirty", encoding="utf-8")
    stash_ref = update_cmd._stash_local_changes_if_needed(["git"], repo)
    assert stash_ref is not None
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "clean"

    restored = update_cmd._restore_failed_release_update(
        ["git"], repo, "main", start_sha, stash_ref
    )

    assert restored
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "dirty"
    assert _git(repo, "stash", "list").stdout.strip() == ""


def test_failed_release_checkout_restores_dirty_detached_worktree(tmp_path):
    repo = _repo(tmp_path)
    start_sha = _commit(repo, "start", "clean")
    _commit(repo, "later", "later")
    _git(repo, "checkout", "--detach", start_sha)
    (repo / "tracked.txt").write_text("dirty", encoding="utf-8")
    (repo / "untracked.txt").write_text("untracked", encoding="utf-8")
    stash_ref = update_cmd._stash_local_changes_if_needed(["git"], repo)
    assert stash_ref is not None
    _git(repo, "checkout", "main")

    restored = update_cmd._restore_failed_release_update(
        ["git"], repo, "HEAD", start_sha, stash_ref
    )

    assert restored
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == start_sha
    assert _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "HEAD"
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "dirty"
    assert (repo / "untracked.txt").read_text(encoding="utf-8") == "untracked"
    assert _git(repo, "stash", "list").stdout.strip() == ""

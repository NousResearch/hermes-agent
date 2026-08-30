"""Behavioral regressions for divergent Git updates."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hermes_cli.update_cmd import _recover_diverged_update

GIT = ["git", "-c", "user.name=Hermes Test", "-c", "user.email=test@example.invalid"]
REMOTE_REF = "refs/remotes/origin/main"


def _git(
    repo: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    home = repo.parent / "home"
    home.mkdir(exist_ok=True)
    return subprocess.run(
        GIT + list(args),
        cwd=repo,
        env={**os.environ, "HOME": str(home), "HERMES_HOME": str(home / ".hermes")},
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _commit(repo: Path, path: str, text: str, message: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    base = _commit(repo, "tracked.txt", "base\n", "base")
    _git(repo, "update-ref", "--create-reflog", REMOTE_REF, base)
    return repo, base


def _rewrite_remote(repo: Path, base: str, *, conflict: bool = False) -> str:
    branch = _git(repo, "branch", "--show-current").stdout.strip()
    _git(repo, "checkout", "-b", "rewritten", base)
    if conflict:
        remote = _commit(repo, "tracked.txt", "remote\n", "rewritten conflict")
    else:
        remote = _commit(repo, "replacement.txt", "replacement\n", "rewritten upstream")
    _git(repo, "update-ref", REMOTE_REF, remote)
    _git(repo, "checkout", branch)
    _git(repo, "branch", "-D", "rewritten")
    return remote


def _assert_clean_at(repo: Path, sha: str) -> None:
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == sha
    assert not _git(repo, "status", "--porcelain").stdout
    git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir").stdout.strip())
    assert not (git_dir / "rebase-merge").exists()
    assert not (git_dir / "rebase-apply").exists()


def test_rewritten_upstream_preserves_only_genuine_local_commit(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    discarded = _commit(repo, "obsolete.txt", "obsolete\n", "discarded upstream")
    _git(repo, "update-ref", REMOTE_REF, discarded)
    original = _commit(repo, "local.txt", "local\n", "genuine local commit")
    rewritten = _rewrite_remote(repo, base)

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert ok, detail
    assert _git(repo, "merge-base", "--is-ancestor", rewritten, "HEAD").returncode == 0
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() != original
    assert _git(
        repo, "log", "--format=%s", f"{rewritten}..HEAD"
    ).stdout.splitlines() == ["genuine local commit"]
    assert not (repo / "obsolete.txt").exists()
    assert (repo / "local.txt").read_text(encoding="utf-8") == "local\n"


def test_safe_local_merge_topology_is_preserved(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    _commit(repo, "main.txt", "main\n", "main local")
    _git(repo, "checkout", "-b", "side", base)
    _commit(repo, "side.txt", "side\n", "side local")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "side", "-m", "merge local side")
    remote = _rewrite_remote(repo, base)

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert ok, detail
    assert _git(repo, "merge-base", "--is-ancestor", remote, "HEAD").returncode == 0
    assert (
        _git(repo, "rev-list", "--count", "--merges", f"{remote}..HEAD").stdout.strip()
        == "1"
    )
    assert (repo / "main.txt").read_text(encoding="utf-8") == "main\n"
    assert (repo / "side.txt").read_text(encoding="utf-8") == "side\n"


def test_merge_only_payload_fails_closed(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    _commit(repo, "main.txt", "main\n", "main local")
    _git(repo, "checkout", "-b", "side", base)
    _commit(repo, "side.txt", "side\n", "side local")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "--no-commit", "side")
    (repo / "payload.txt").write_text("must survive\n", encoding="utf-8")
    _git(repo, "add", "payload.txt")
    _git(repo, "commit", "-m", "merge payload")
    original = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _rewrite_remote(repo, base)

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert not ok
    assert "merge" in detail.lower()
    _assert_clean_at(repo, original)
    assert (repo / "payload.txt").read_text(encoding="utf-8") == "must survive\n"


def test_recovery_preserves_safe_local_merge_topology(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    _git(repo, "update-ref", REMOTE_REF, base)
    main = _git(repo, "branch", "--show-current").stdout.strip()
    _commit(repo, "main.txt", "main\n", "main work")
    _git(repo, "checkout", "-b", "topic", base)
    _commit(repo, "topic.txt", "topic\n", "topic work")
    _git(repo, "checkout", main)
    _git(repo, "merge", "--no-ff", "topic", "-m", "safe local merge")
    rewritten = _rewrite_remote(repo, base)

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert ok, detail
    assert _git(repo, "merge-base", "--is-ancestor", rewritten, "HEAD").returncode == 0
    assert _git(repo, "rev-list", "--merges", f"{rewritten}..HEAD").stdout.strip()
    assert (repo / "main.txt").read_text(encoding="utf-8") == "main\n"
    assert (repo / "topic.txt").read_text(encoding="utf-8") == "topic\n"


def test_conflict_aborts_and_restores_original_head(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    original = _commit(repo, "tracked.txt", "local\n", "local conflict")
    _rewrite_remote(repo, base, conflict=True)

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert not ok
    assert detail
    _assert_clean_at(repo, original)
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "local\n"


def test_no_local_commit_resets_to_rewritten_remote(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    discarded = _commit(repo, "obsolete.txt", "obsolete\n", "discarded upstream")
    _git(repo, "update-ref", REMOTE_REF, discarded)
    rewritten = _rewrite_remote(repo, base)

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert ok, detail
    _assert_clean_at(repo, rewritten)
    assert not (repo / "obsolete.txt").exists()


def test_detached_head_preserves_local_commit(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    old_upstream = _commit(repo, "old.txt", "old\n", "old upstream")
    _git(repo, "update-ref", REMOTE_REF, old_upstream)
    local = _commit(repo, "local.txt", "local\n", "detached local")
    rewritten = _rewrite_remote(repo, base)
    _git(repo, "checkout", "--detach", local)

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert ok, detail
    assert not _git(repo, "branch", "--show-current").stdout.strip()
    assert _git(repo, "merge-base", "--is-ancestor", rewritten, "HEAD").returncode == 0
    assert _git(repo, "log", "-1", "--format=%s").stdout.strip() == "detached local"


def test_missing_remote_reflog_fails_closed(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    original = _commit(repo, "local.txt", "local\n", "local")
    _rewrite_remote(repo, base)
    reflog = repo / ".git" / "logs" / "refs" / "remotes" / "origin" / "main"
    reflog.unlink()

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert not ok
    assert "fork point" in detail
    _assert_clean_at(repo, original)

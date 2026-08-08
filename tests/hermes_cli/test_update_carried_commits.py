"""Real-git regressions for preserving carried history on divergent updates."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from hermes_cli.update_cmd import _recover_diverged_update


GIT = ["git", "-c", "user.name=Hermes Test", "-c", "user.email=test@example.invalid"]
REMOTE_REF = "refs/remotes/origin/main"


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
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


def _init_git_repo(repo: Path) -> None:
    """Create a repo whose rebases do not depend on the host's Git identity."""
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "test@example.invalid")


def _write_commit(repo: Path, path: str, text: str, message: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    base = _write_commit(repo, "tracked.txt", "base\n", "base")
    _git(repo, "update-ref", "--create-reflog", REMOTE_REF, base)
    return repo, base


def _advance_remote(repo: Path, base: str, *, conflict: bool = False) -> str:
    local_branch = _git(repo, "branch", "--show-current").stdout.strip()
    _git(repo, "checkout", "-b", "remote-work", base)
    if conflict:
        remote = _write_commit(repo, "tracked.txt", "remote\n", "conflicting upstream")
    else:
        remote = _write_commit(repo, "remote.txt", "upstream\n", "upstream change")
    _git(repo, "update-ref", REMOTE_REF, remote)
    _git(repo, "checkout", local_branch)
    _git(repo, "branch", "-D", "remote-work")
    return remote


def _assert_clean_at(repo: Path, sha: str) -> None:
    assert _git(repo, "rev-parse", "HEAD").stdout.strip() == sha
    assert not _git(repo, "status", "--porcelain").stdout
    git_dir = Path(_git(repo, "rev-parse", "--git-dir").stdout.strip())
    if not git_dir.is_absolute():
        git_dir = repo / git_dir
    assert not (git_dir / "rebase-merge").exists()
    assert not (git_dir / "rebase-apply").exists()


def _fake_git(repo: Path, count_output: bytes) -> tuple[Path, dict[str, str]]:
    """Create an executable git proxy that corrupts only rev-list --count."""
    fake_bin = repo.parent / "fake-bin"
    fake_bin.mkdir()
    count_file = fake_bin / "count-output"
    count_file.write_bytes(count_output)
    wrapper = fake_bin / "git"
    wrapper.write_text(
        """#!/bin/sh
saw_rev_list=false
saw_count=false
for arg do
    [ "$arg" = "rev-list" ] && saw_rev_list=true
    [ "$arg" = "--count" ] && saw_count=true
done
if [ "$saw_rev_list" = true ] && [ "$saw_count" = true ]; then
    cat "$HERMES_FAKE_COUNT"
    exit 0
fi
exec "$HERMES_REAL_GIT" "$@"
""",
        encoding="ascii",
    )
    wrapper.chmod(0o755)
    return wrapper, {
        **os.environ,
        "HERMES_FAKE_COUNT": str(count_file),
        "HERMES_REAL_GIT": shutil.which("git") or "git",
    }


def test_diverged_update_rebases_only_actual_local_commit_after_force_push(
    tmp_path: Path,
) -> None:
    repo, base = _init_repo(tmp_path)
    old_upstream = _write_commit(repo, "obsolete.txt", "obsolete\n", "discarded upstream")
    _git(repo, "update-ref", REMOTE_REF, old_upstream)
    local = _write_commit(repo, "local.txt", "local\n", "actual local commit")

    _git(repo, "checkout", "-b", "rewritten", base)
    rewritten = _write_commit(repo, "replacement.txt", "replacement\n", "rewritten upstream")
    _git(repo, "update-ref", REMOTE_REF, rewritten)
    _git(repo, "checkout", "main")
    _git(repo, "branch", "-D", "rewritten")

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert ok, detail
    assert _git(repo, "merge-base", "--is-ancestor", rewritten, "HEAD").returncode == 0
    subjects = _git(repo, "log", "--format=%s", f"{rewritten}..HEAD").stdout.splitlines()
    assert subjects == ["actual local commit"]
    assert _git(repo, "branch", "--show-current").stdout.strip() == "main"
    assert not (repo / "obsolete.txt").exists()
    assert (repo / "local.txt").read_text(encoding="utf-8") == "local\n"
    assert local != _git(repo, "rev-parse", "HEAD").stdout.strip()


def test_diverged_update_preserves_safe_merge_topology(tmp_path: Path) -> None:
    repo, base = _init_repo(tmp_path)
    _write_commit(repo, "main.txt", "main\n", "main local")
    _git(repo, "checkout", "-b", "local-side", base)
    _write_commit(repo, "side.txt", "side\n", "side local")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "local-side", "-m", "merge local side")
    _git(repo, "branch", "-D", "local-side")
    remote = _advance_remote(repo, base)

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert ok, detail
    assert _git(repo, "merge-base", "--is-ancestor", remote, "HEAD").returncode == 0
    assert _git(repo, "rev-list", "--count", "--merges", f"{remote}..HEAD").stdout.strip() == "1"
    assert _git(repo, "branch", "--show-current").stdout.strip() == "main"
    assert (repo / "main.txt").read_text(encoding="utf-8") == "main\n"
    assert (repo / "side.txt").read_text(encoding="utf-8") == "side\n"


def test_merge_resolution_only_payload_fails_closed_instead_of_losing_content(
    tmp_path: Path,
) -> None:
    repo, base = _init_repo(tmp_path)
    _write_commit(repo, "main.txt", "main\n", "main local")
    _git(repo, "checkout", "-b", "local-side", base)
    _write_commit(repo, "side.txt", "side\n", "side local")
    _git(repo, "checkout", "main")
    _git(repo, "merge", "--no-ff", "--no-commit", "local-side")
    (repo / "merge-payload.txt").write_text("must survive\n", encoding="utf-8")
    _git(repo, "add", "merge-payload.txt")
    _git(repo, "commit", "-m", "merge with resolution-only payload")
    original = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "branch", "-D", "local-side")
    _advance_remote(repo, base)

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert not ok
    assert "merge" in detail.lower()
    _assert_clean_at(repo, original)
    assert (repo / "merge-payload.txt").read_text(encoding="utf-8") == "must survive\n"


def test_diverged_update_aborts_conflict_and_restores_clean_original_head(
    tmp_path: Path,
) -> None:
    repo, base = _init_repo(tmp_path)
    original = _write_commit(repo, "tracked.txt", "local\n", "local carried fix")
    _advance_remote(repo, base, conflict=True)

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert not ok
    assert detail
    _assert_clean_at(repo, original)
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "local\n"


def test_diverged_update_count_failure_fails_closed(monkeypatch, tmp_path: Path) -> None:
    repo, base = _init_repo(tmp_path)
    original = _write_commit(repo, "local.txt", "local\n", "local carried fix")
    _advance_remote(repo, base)
    real_run = subprocess.run

    def fail_count(cmd, **kwargs):
        if "rev-list" in cmd and "--count" in cmd:
            return subprocess.CompletedProcess(cmd, 128, "", "count failed")
        return real_run(cmd, **kwargs)

    with monkeypatch.context() as context:
        context.setattr("hermes_cli.main.subprocess.run", fail_count)
        ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert not ok
    assert "could not determine" in detail
    _assert_clean_at(repo, original)


@pytest.mark.parametrize(
    ("count_output", "expected_local_commits"),
    [
        (b"1\n", 1),
        (b"1", 1),
        (b"0\n", 0),
        (b"0", 0),
        (b"000000001\n", 1),
        (b"000000000\n", 0),
    ],
    ids=[
        "one-terminated",
        "one-unterminated",
        "zero-terminated",
        "zero-unterminated",
        "nine-digit-zero-padded-one",
        "nine-digit-all-zero",
    ],
)
def test_diverged_update_accepts_single_count_with_optional_final_newline(
    count_output: bytes, expected_local_commits: int, monkeypatch, tmp_path: Path
) -> None:
    repo, base = _init_repo(tmp_path)
    original = _write_commit(repo, "local.txt", "local\n", "local carried fix")
    remote = _advance_remote(repo, base)
    wrapper, fake_env = _fake_git(repo, count_output)

    with monkeypatch.context() as context:
        context.setenv("HERMES_FAKE_COUNT", fake_env["HERMES_FAKE_COUNT"])
        context.setenv("HERMES_REAL_GIT", fake_env["HERMES_REAL_GIT"])
        ok, detail = _recover_diverged_update([str(wrapper)], repo, REMOTE_REF)

    assert ok, detail
    if expected_local_commits:
        assert _git(repo, "rev-parse", "HEAD").stdout.strip() != original
        assert _git(repo, "merge-base", "--is-ancestor", remote, "HEAD").returncode == 0
        assert _git(repo, "log", "-1", "--format=%s").stdout.strip() == "local carried fix"
    else:
        _assert_clean_at(repo, remote)
    assert not _git(repo, "status", "--porcelain").stdout


@pytest.mark.parametrize(
    "count_output",
    [
        b"+1\n",
        b"-1\n",
        b" 1 \n",
        b"1x\n",
        b"",
        b"1000000000\n",
        b"9" * 36 + b"\n",
        b"0000000001\n",
        b"0000000000\n",
        b"0" * 4301 + b"1\n",
        b"0" * 4301 + b"0\n",
        b"1\n2\n",
        "\u0661\n".encode(),
    ],
    ids=[
        "positive-sign",
        "negative-sign",
        "padded",
        "suffix",
        "empty",
        "over-bound",
        "oversized",
        "ten-digit-zero-padded-one",
        "ten-digit-all-zero",
        "giant-zero-padded-one",
        "giant-all-zero",
        "multiple-records",
        "non-ascii-digit",
    ],
)
def test_diverged_update_malformed_count_matrix_fails_closed(
    count_output: bytes, monkeypatch, tmp_path: Path
) -> None:
    repo, base = _init_repo(tmp_path)
    original = _write_commit(repo, "local.txt", "local\n", "local carried fix")
    _advance_remote(repo, base)
    wrapper, fake_env = _fake_git(repo, count_output)

    with monkeypatch.context() as context:
        context.setenv("HERMES_FAKE_COUNT", fake_env["HERMES_FAKE_COUNT"])
        context.setenv("HERMES_REAL_GIT", fake_env["HERMES_REAL_GIT"])
        ok, detail = _recover_diverged_update([str(wrapper)], repo, REMOTE_REF)

    assert not ok
    assert "could not determine" in detail
    _assert_clean_at(repo, original)


def test_missing_remote_reflog_fails_closed(tmp_path: Path) -> None:
    repo, base = _init_repo(tmp_path)
    original = _write_commit(repo, "local.txt", "local\n", "local carried fix")
    _advance_remote(repo, base)
    reflog = repo / ".git" / "logs" / "refs" / "remotes" / "origin" / "main"
    reflog.unlink()

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert not ok
    assert "fork point" in detail
    _assert_clean_at(repo, original)


def test_force_push_without_local_commits_resets_to_rewritten_remote(tmp_path: Path) -> None:
    repo, base = _init_repo(tmp_path)
    old_upstream = _write_commit(repo, "obsolete.txt", "obsolete\n", "discarded upstream")
    _git(repo, "update-ref", REMOTE_REF, old_upstream)
    _git(repo, "checkout", "-b", "rewritten", base)
    rewritten = _write_commit(repo, "replacement.txt", "replacement\n", "rewritten upstream")
    _git(repo, "update-ref", REMOTE_REF, rewritten)
    _git(repo, "checkout", "main")
    _git(repo, "branch", "-D", "rewritten")

    ok, detail = _recover_diverged_update(["git"], repo, REMOTE_REF)

    assert ok, detail
    _assert_clean_at(repo, rewritten)
    assert not (repo / "obsolete.txt").exists()

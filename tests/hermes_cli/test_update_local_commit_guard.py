"""The source updater refuses local-only main commits before any update phase starts."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import main, update_cmd


def _git(repo, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _repo_with_local_main_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "hermes-test@example.invalid")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "base")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    (repo / "tracked.txt").write_text("local\n", encoding="utf-8")
    _git(repo, "commit", "-am", "local-only")
    return repo


def test_update_is_refused_before_snapshot_for_local_only_main_commit(tmp_path, monkeypatch, capsys):
    repo = _repo_with_local_main_commit(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(main, "PROJECT_ROOT", repo)

    with pytest.raises(SystemExit) as exc:
        update_cmd._cmd_update_impl(SimpleNamespace(), gateway_mode=False)

    assert exc.value.code == 1
    output = capsys.readouterr().out
    assert "Update blocked" in output
    assert "local-only" in output
    assert "No stash, rebase, reset, or push was attempted" in output
    assert _git(repo, "rev-parse", "HEAD") == head


def test_update_guard_fails_closed_when_origin_main_ref_is_missing(tmp_path):
    from hermes_cli.update_local_commit_guard import LocalUpdateGuardError, local_main_commits

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    with pytest.raises(LocalUpdateGuardError):
        local_main_commits(repo)

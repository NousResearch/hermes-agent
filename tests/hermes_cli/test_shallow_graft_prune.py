"""Bounding .git/shallow growth after depth-1 update checks (#105951).

Every ``git fetch --depth 1`` appends the fetched tip to ``.git/shallow`` as a new
graft and never removes the tip it replaced, so a long-lived shallow installer
checkout accumulates one graft line per update check (57 observed in the wild).
``_prune_stale_shallow_grafts()`` drops grafts no live ref points at;
``hermes update --check`` calls it after its successful depth-1 fetch, bounding the
graft list instead of letting it grow (the passive banner check no longer
git-fetches since #107648).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from hermes_cli.update_cmd import _prune_stale_shallow_grafts

SHA_A = "a" * 40
SHA_B = "b" * 40


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _shallow_lines(repo: Path) -> list:
    return [
        line for line in (repo / ".git" / "shallow").read_text().splitlines() if line
    ]


def _mk_shallow_scenario(tmp_path: Path) -> Path:
    """Depth-1 clone whose origin advanced twice: shallow carries 3 grafts."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@example.com")
    _git(origin, "config", "user.name", "t")
    for i in range(3):
        _git(origin, "commit", "--allow-empty", "-q", "-m", f"c{i}")
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{origin}", str(clone)],
        check=True,
        capture_output=True,
        text=True,
    )
    for i in range(3, 5):
        _git(origin, "commit", "--allow-empty", "-q", "-m", f"c{i}")
        _git(clone, "fetch", "-q", "--depth", "1", "origin", "main")
    return clone


def test_prunes_replaced_tip_keeps_referenced_boundaries(tmp_path):
    clone = _mk_shallow_scenario(tmp_path)
    assert len(_shallow_lines(clone)) == 3  # HEAD graft + two fetched tips

    head_sha = _git(clone, "rev-parse", "HEAD")
    tip_sha = _git(clone, "rev-parse", "origin/main")

    removed = _prune_stale_shallow_grafts(["git"], cwd=str(clone))

    assert removed == 1  # the middle, now-unreferenced tip
    assert set(_shallow_lines(clone)) == {head_sha, tip_sha}
    # Boundaries that survive must still walk cleanly.
    assert _git(clone, "rev-list", "--count", "HEAD") == "1"
    assert _git(clone, "rev-list", "--count", "origin/main") == "1"


def test_prune_is_idempotent_and_noop_outside_shallow_checkouts(tmp_path):
    clone = _mk_shallow_scenario(tmp_path)
    assert _prune_stale_shallow_grafts(["git"], cwd=str(clone)) == 1
    assert (
        _prune_stale_shallow_grafts(["git"], cwd=str(clone)) == 0
    )  # nothing left to drop

    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()
    assert _prune_stale_shallow_grafts(["git"], cwd=str(not_a_repo)) == 0  # no-op


def test_update_check_bounds_grafts_after_fetch(tmp_path, monkeypatch, capsys):
    """`hermes update --check` prunes grafts after its depth-1 fetch and reports the prune."""
    import hermes_cli.update_cmd as update_cmd

    fake_root = SimpleNamespace(PROJECT_ROOT=tmp_path)
    monkeypatch.setattr(update_cmd, "_m", lambda: fake_root)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(
        "hermes_cli.update_contract.evaluate_update_admission", lambda root: None
    )
    monkeypatch.setattr(update_cmd, "_is_shallow_checkout", lambda git_cmd: True)
    monkeypatch.setattr(update_cmd, "_tip_shas", lambda git_cmd, branch: (SHA_A, SHA_B))

    def fake_git_run(git_cmd, args, **kwargs):
        joined = " ".join(args)
        if "get-url" in joined and "upstream" in joined:
            return MagicMock(returncode=1, stdout="", stderr="")  # no upstream remote
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(update_cmd, "_git_run", fake_git_run)
    monkeypatch.setattr(update_cmd, "_base_git_cmd", lambda: ["git"])
    monkeypatch.setattr("hermes_cli.banner._github_compare_behind", lambda *a, **k: 0)
    prune_calls = []
    monkeypatch.setattr(
        update_cmd,
        "_prune_stale_shallow_grafts",
        lambda git_cmd, cwd=None: prune_calls.append(git_cmd) or 2,
    )

    update_cmd._cmd_update_check("main")

    out = capsys.readouterr().out
    assert prune_calls == [["git"]]
    assert "pruned 2 stale shallow graft(s)" in out

"""Stale shallow-graft pruning after depth-1 update checks (#105951).

Every ``git fetch --depth 1`` appends the fetched tip to ``.git/shallow`` as a
new graft and never removes the previous one, so a long-lived shallow installer
checkout accumulates one line per update check (57 observed in the wild). The
stale grafts break ``merge-base`` and push ``hermes update`` into the
orphan-divergence reset path. ``prune_stale_shallow_grafts()`` drops grafts no
live ref points at; ``hermes update --check`` calls it after its successful
depth-1 fetch, clearing the grafts accumulated by past checks (the passive
banner check no longer git-fetches since #107648).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from hermes_cli.gitlock import prune_stale_shallow_grafts


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


def test_prunes_orphaned_grafts_keeps_referenced_boundaries(tmp_path):
    clone = _mk_shallow_scenario(tmp_path)
    assert len(_shallow_lines(clone)) == 3  # HEAD graft + two fetched tips

    head_sha = _git(clone, "rev-parse", "HEAD")
    tip_sha = _git(clone, "rev-parse", "origin/main")

    removed = prune_stale_shallow_grafts(clone)

    assert removed == 1  # the middle, now-unreferenced tip
    assert set(_shallow_lines(clone)) == {head_sha, tip_sha}
    # Boundaries that survive must still walk cleanly.
    assert _git(clone, "rev-list", "--count", "HEAD") == "1"
    assert _git(clone, "rev-list", "--count", "origin/main") == "1"


def test_prune_is_idempotent_and_noop_without_grafts(tmp_path):
    clone = _mk_shallow_scenario(tmp_path)
    assert prune_stale_shallow_grafts(clone) == 1
    assert prune_stale_shallow_grafts(clone) == 0  # nothing left to drop

    empty_dir = tmp_path / "not-a-repo"
    empty_dir.mkdir()
    assert prune_stale_shallow_grafts(empty_dir) == 0  # not a git repo: no-op

    git_no_shallow = tmp_path / "full-clone"
    git_no_shallow.mkdir()
    (git_no_shallow / ".git").mkdir()
    assert prune_stale_shallow_grafts(git_no_shallow) == 0  # no shallow file: no-op


def test_update_check_prunes_and_reports_count(tmp_path, monkeypatch, capsys):
    """`hermes update --check` prunes grafts after its depth-1 fetch and reports the prune."""
    import hermes_cli.update_cmd as update_cmd

    clone = _mk_shallow_scenario(tmp_path)
    assert len(_shallow_lines(clone)) == 3
    head_sha = _git(clone, "rev-parse", "HEAD")
    previous_tip = _git(clone, "rev-parse", "origin/main")
    origin = tmp_path / "origin"
    _git(origin, "commit", "--allow-empty", "-q", "-m", "c5")
    tip_sha = _git(origin, "rev-parse", "HEAD")

    # The check runs git directly, reading main.PROJECT_ROOT through _m().
    monkeypatch.setattr(update_cmd._m(), "PROJECT_ROOT", clone)
    monkeypatch.setattr(
        "hermes_cli.update_contract.evaluate_update_admission", lambda root: None
    )
    # Local fixture commits have no GitHub compare result; keep the check offline.
    monkeypatch.setattr(
        "hermes_cli.source_check._github_compare_behind", lambda *a, **k: None
    )

    update_cmd._cmd_update_check("main")

    out = capsys.readouterr().out
    assert _git(clone, "rev-parse", "HEAD") == head_sha
    assert _git(clone, "rev-parse", "origin/main") == tip_sha != previous_tip
    assert _git(clone, "rev-parse", "FETCH_HEAD") == tip_sha
    assert set(_shallow_lines(clone)) == {head_sha, tip_sha}
    assert _git(clone, "rev-list", "--count", "HEAD") == "1"
    assert _git(clone, "rev-list", "--count", "origin/main") == "1"
    assert "pruned 2 stale shallow graft(s)" in out
    assert "Update available (behind origin/main)." in out

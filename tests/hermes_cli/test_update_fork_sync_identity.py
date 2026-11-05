"""The fork sync acts on ONE upstream commit (review R6 m2).

``_sync_with_upstream_if_needed`` fetches ``refs/remotes/upstream/main`` and fast-forwards to it. The
short name ``upstream/main`` also names a LOCAL branch of that name (git resolves ``refs/heads``
before ``refs/remotes``), so a count by short name and a merge by full ref could each see a
different commit: the sync printed "1 behind" and merged 2 commits.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from hermes_cli.update_cmd_git import UpstreamTargetBroken, _sync_with_upstream_if_needed


def _env(home: Path) -> dict:
    return {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(home),
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          encoding="utf-8", env=_env(cwd)).stdout.strip()


def test_a_local_branch_named_upstream_main_never_changes_what_the_sync_counts_or_merges(
        tmp_path, monkeypatch, capsys):
    for name, value in _env(tmp_path).items():
        if name.startswith("GIT_") or name == "HOME":
            monkeypatch.setenv(name, value)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    commits = []
    for i in range(3):
        (upstream / "f.txt").write_text(f"{i}\n", encoding="utf-8")
        _git(upstream, "add", "f.txt")
        _git(upstream, "commit", "-qm", f"c{i}")
        commits.append(_git(upstream, "rev-parse", "HEAD"))
    origin = tmp_path / "origin.git"
    _git(tmp_path, "clone", "-q", "--bare", str(upstream), str(origin))
    _git(origin, "update-ref", "refs/heads/main", commits[0])
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "remote", "add", "upstream", str(upstream))
    # The shadow: a local branch literally named upstream/main, one commit behind upstream.
    _git(clone, "fetch", "-q", "upstream")
    _git(clone, "branch", "upstream/main", commits[1])

    from hermes_cli import update_cmd_commit
    from hermes_cli.update_cmd_git import _push_synced_fork

    real_arm = update_cmd_commit.arm_tree_move
    armed = []

    def arm(*args, **kwargs):
        armed.append(kwargs["target"])
        return real_arm(*args, **kwargs)

    monkeypatch.setattr(update_cmd_commit, "arm_tree_move", arm)
    assert _sync_with_upstream_if_needed(["git"], clone, assume_yes=True) is True
    out = capsys.readouterr().out
    assert "Fork is 2 commit(s) behind upstream" in out, out
    assert _git(clone, "rev-parse", "HEAD") == commits[2]
    # The tree-move marker names the merged commit: a killed merge is repaired toward it.
    assert armed == [commits[2]]
    # The deferred fork push checks HEAD against the same remote-tracking commit, not the shadow.
    _push_synced_fork(["git"], clone)
    assert _git(origin, "rev-parse", "refs/heads/main") == commits[2]


def _fork_behind_upstream(tmp_path, monkeypatch):
    """A fork clone at c1 (origin/main) whose fetched upstream/main is c2; returns (clone, commits)."""
    for name, value in _env(tmp_path).items():
        if name.startswith("GIT_") or name == "HOME":
            monkeypatch.setenv(name, value)
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    commits = []
    for i in range(3):
        (upstream / "f.txt").write_text(f"{i}\n", encoding="utf-8")
        _git(upstream, "add", "f.txt")
        _git(upstream, "commit", "-qm", f"c{i}")
        commits.append(_git(upstream, "rev-parse", "HEAD"))
    origin = tmp_path / "origin.git"
    _git(tmp_path, "clone", "-q", "--bare", str(upstream), str(origin))
    _git(origin, "update-ref", "refs/heads/main", commits[1])
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "remote", "add", "upstream", str(upstream))
    return clone, commits


def test_an_upstream_sync_whose_marker_cannot_be_written_never_moves_the_tree(tmp_path, monkeypatch):
    """The fork fast-forward is a tree move like the pull: no marker, no merge (F24)."""
    from hermes_cli import update_cmd_commit
    from hermes_cli._early_recovery import interrupted_pull_marker

    clone, commits = _fork_behind_upstream(tmp_path, monkeypatch)
    update_cmd_commit.begin_update_attempt()
    interrupted_pull_marker(clone).mkdir()
    try:
        assert _sync_with_upstream_if_needed(["git"], clone, assume_yes=True) is False
    finally:
        update_cmd_commit.begin_update_attempt()
    assert _git(clone, "rev-parse", "HEAD") == commits[1]
    assert _git(clone, "status", "--porcelain", "--untracked-files=no") == ""


def test_a_failed_upstream_sync_after_the_pull_owes_the_restart_for_the_pulled_commit(tmp_path, monkeypatch):
    """The origin pull moved c0 -> c1 and the fork ff to c2 failed back to c1: the obligation must name
    c1 (dischargeable at HEAD), not the c2 the checkout never reached (F23)."""
    from hermes_cli import update_cmd_commit, update_custody
    from hermes_cli.update_host_obligation import read_host_obligation
    from hermes_cli.venv_sync import completion_pending_path

    clone, commits = _fork_behind_upstream(tmp_path, monkeypatch)
    update_cmd_commit.begin_update_attempt()
    _git(clone, "reset", "-q", "--hard", commits[0])
    update_cmd_commit.record_run_start(["git"], clone)
    update_cmd_commit.arm_commit_obligations(clone, commits[1])
    _git(clone, "reset", "-q", "--hard", commits[1])  # the committed origin pull
    real = update_custody.run_git

    def merge_fails(git_cmd, args, *rest, **kw):
        if args[:1] == ["merge"]:
            raise subprocess.CalledProcessError(128, args, stderr="fatal: Unable to create index.lock")
        return real(git_cmd, args, *rest, **kw)

    monkeypatch.setattr(update_custody, "run_git", merge_fails)
    try:
        assert _sync_with_upstream_if_needed(["git"], clone, assume_yes=True) is False
    finally:
        update_cmd_commit.begin_update_attempt()
    assert _git(clone, "rev-parse", "HEAD") == commits[1]
    assert (read_host_obligation() or {}).get("expected_sha") == commits[1]
    assert completion_pending_path(clone).is_file()


def test_a_custody_refused_upstream_merge_after_the_pull_still_owes_the_pulled_commit(tmp_path, monkeypatch, capsys):
    """A merge refused by custody (CustodyRefused, an OSError) used to escape ``except
    CalledProcessError`` with the obligation naming c2, which the checkout at c1 never holds
    (review C1): it must settle like any failed merge and owe the restart for c1."""
    from hermes_cli import update_cmd_commit, update_custody
    from hermes_cli.update_host_obligation import read_host_obligation

    clone, commits = _fork_behind_upstream(tmp_path, monkeypatch)
    update_cmd_commit.begin_update_attempt()
    _git(clone, "reset", "-q", "--hard", commits[0])
    update_cmd_commit.record_run_start(["git"], clone)
    update_cmd_commit.arm_commit_obligations(clone, commits[1])
    _git(clone, "reset", "-q", "--hard", commits[1])  # the committed origin pull
    real = update_custody.run_git

    def merge_refused(git_cmd, args, *rest, **kw):
        if args[:1] == ["merge"]:
            raise update_custody.CustodyRefused(["git"], "access denied")
        return real(git_cmd, args, *rest, **kw)

    monkeypatch.setattr(update_custody, "run_git", merge_refused)
    try:
        assert _sync_with_upstream_if_needed(["git"], clone, assume_yes=True) is False
    finally:
        update_cmd_commit.begin_update_attempt()
    assert "process job (access denied)" in capsys.readouterr().out
    assert _git(clone, "rev-parse", "HEAD") == commits[1]
    assert (read_host_obligation() or {}).get("expected_sha") == commits[1]


def test_a_failed_upstream_sync_whose_retarget_write_fails_still_owes_the_pulled_commit(tmp_path, monkeypatch):
    """After the origin pull c0 -> c1, the fork ff to c2 armed the debt for c2, failed, and its
    retarget back to c1 hit a writer that now refuses: the debt stayed c2 with HEAD on c1 (review
    O4). The later move owes c1 from the start (every landing of the ff contains it), so a failed
    retarget changes nothing."""
    from hermes_cli import update_cmd_commit, update_cmd_fleet, update_custody
    from hermes_cli.update_host_obligation import read_host_obligation

    clone, commits = _fork_behind_upstream(tmp_path, monkeypatch)
    update_cmd_commit.begin_update_attempt()
    _git(clone, "reset", "-q", "--hard", commits[0])
    update_cmd_commit.record_run_start(["git"], clone)
    update_cmd_commit.arm_commit_obligations(clone, commits[1])
    _git(clone, "reset", "-q", "--hard", commits[1])  # the committed origin pull
    real_write, real_git, writes = update_cmd_fleet._write_fleet_restart_pending_marker, update_custody.run_git, []

    def writer_then_refuses(**kw):
        writes.append(kw.get("expected_sha"))
        return real_write(**kw) if len(writes) == 1 else False  # the store turns unwritable

    def merge_fails(git_cmd, args, *rest, **kw):
        if args[:1] == ["merge"]:
            raise subprocess.CalledProcessError(128, args, stderr="fatal: Unable to create index.lock")
        return real_git(git_cmd, args, *rest, **kw)

    monkeypatch.setattr(update_cmd_fleet, "_write_fleet_restart_pending_marker", writer_then_refuses)
    monkeypatch.setattr(update_custody, "run_git", merge_fails)
    try:
        assert _sync_with_upstream_if_needed(["git"], clone, assume_yes=True) is False
    finally:
        update_cmd_commit.begin_update_attempt()
    assert _git(clone, "rev-parse", "HEAD") == commits[1]
    assert commits[2] not in writes
    assert (read_host_obligation() or {}).get("expected_sha") == commits[1]


def test_a_broken_upstream_target_is_refused_before_the_second_move(tmp_path, monkeypatch, capsys):
    """The fork ff dropped its marker on git's exit 0 and only the caller's later syntax guard
    judged the result, so a kill in between left a broken c2 with no recovery record (review G1).
    The upstream target is compiled before its move, like the origin target's preflight."""
    from hermes_cli import update_cmd_commit
    from hermes_cli._early_recovery import interrupted_pull_marker

    clone, commits = _fork_behind_upstream(tmp_path, monkeypatch)
    upstream = tmp_path / "upstream"
    (upstream / "hermes_constants.py").write_text("def broken(:\n", encoding="utf-8")
    _git(upstream, "add", "hermes_constants.py")
    _git(upstream, "commit", "-qm", "broken c3")
    update_cmd_commit.begin_update_attempt()
    try:
        with pytest.raises(UpstreamTargetBroken) as refused:
            _sync_with_upstream_if_needed(["git"], clone, assume_yes=True)
    finally:
        update_cmd_commit.begin_update_attempt()
    assert refused.value.path == "hermes_constants.py"
    assert _git(clone, "rev-parse", "HEAD") == commits[1]
    assert not interrupted_pull_marker(clone).exists()

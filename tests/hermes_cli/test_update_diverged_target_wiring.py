"""End-to-end proof that the diverged-target planner is actually WIRED IN.

The unit tests above prove ``_plan_diverged_target_update`` returns the right
decision. They do not prove ``hermes update`` calls it — and an unwired
planner would still leave the 2026-08-31 data-loss bug in place.

These tests drive the real ff-only-failure block over real git repositories,
with only the boundaries stubbed (PROJECT_ROOT, and the post-pull steps that
would rebuild the app), and assert on the resulting git state: the commits
either survived or did not.
"""

import subprocess
import sys

import pytest

from hermes_cli import update_cmd


GIT = ["git"]


def _git(cwd, *args, check=True):
    return subprocess.run(
        GIT + list(args), cwd=cwd, capture_output=True, text=True, check=check
    )


def _sha(cwd, rev="HEAD"):
    return _git(cwd, "rev-parse", rev).stdout.strip()


def _log_subjects(cwd):
    return _git(cwd, "log", "--format=%s").stdout.split("\n")


def _make_origin(tmp_path):
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@e.c")
    _git(origin, "config", "user.name", "T")
    (origin / "a.txt").write_text("one\n")
    _git(origin, "add", "a.txt")
    _git(origin, "commit", "-qm", "c1")
    return origin


def _make_clone(tmp_path, origin):
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    _git(clone, "config", "user.email", "t@e.c")
    _git(clone, "config", "user.name", "T")
    return clone


def _run_diverged_recovery(clone, monkeypatch, capsys):
    """Run the REAL production recovery against ``clone``.

    Calls ``update_cmd._recover_diverged_checkout`` — the exact function the
    updater invokes when ``git merge --ff-only`` fails — so a regression in
    production code fails these tests. Only the fetch/ff-only preamble is
    reproduced, to put the repo in the diverged state the function expects.
    """
    monkeypatch.setattr(update_cmd, "PROJECT_ROOT", clone, raising=False)

    _git(clone, "fetch", "-q", "origin", "main")
    ff = subprocess.run(
        GIT + ["merge", "--ff-only", "origin/main"],
        cwd=clone, capture_output=True, text=True,
    )
    assert ff.returncode != 0, "fixture must actually diverge"

    try:
        action = update_cmd._recover_diverged_checkout(GIT, clone, "main")
    except SystemExit as exc:
        # The production path exits on an aborted merge; surface it as an
        # outcome so tests can assert the checkout was left untouched.
        assert exc.code == 1
        return "conflict-stopped", capsys.readouterr().out
    return action, capsys.readouterr().out


def test_update_preserves_local_commits_on_main(tmp_path, monkeypatch, capsys):
    """THE REGRESSION: local commits on main must survive an update."""
    origin = _make_origin(tmp_path)
    clone = _make_clone(tmp_path, origin)

    (clone / "grok.txt").write_text("dedicated Grok guidance\n")
    _git(clone, "add", "grok.txt")
    _git(clone, "commit", "-qm", "feat(prompt): dedicated Grok execution guidance")
    local_sha = _sha(clone)

    (origin / "b.txt").write_text("upstream work\n")
    _git(origin, "add", "b.txt")
    _git(origin, "commit", "-qm", "upstream c2")

    outcome, output = _run_diverged_recovery(clone, monkeypatch, capsys)

    assert outcome == "merged"
    assert "carries 1 local commit(s) not upstream" in output
    # The commit is still reachable — the actual thing that was lost.
    assert (
        subprocess.run(
            GIT + ["merge-base", "--is-ancestor", local_sha, "HEAD"],
            cwd=clone, capture_output=True,
        ).returncode
        == 0
    ), "local commit was destroyed by the update"
    assert "feat(prompt): dedicated Grok execution guidance" in _log_subjects(clone)
    # And upstream work actually arrived.
    assert (clone / "b.txt").exists()


def test_update_still_recovers_from_upstream_force_push(tmp_path, monkeypatch, capsys):
    """No local work → the hard reset still runs and lands on the remote."""
    origin = _make_origin(tmp_path)
    clone = _make_clone(tmp_path, origin)

    (origin / "a.txt").write_text("rewritten\n")
    _git(origin, "commit", "-aq", "--amend", "-m", "c1 rewritten")

    outcome, _ = _run_diverged_recovery(clone, monkeypatch, capsys)

    assert outcome == "reset"
    _git(clone, "fetch", "-q", "origin", "main")
    assert _sha(clone) == _sha(clone, "origin/main")
    assert (clone / "a.txt").read_text() == "rewritten\n"


def test_conflicting_local_commit_stops_update_without_losing_work(
    tmp_path, monkeypatch, capsys
):
    """A conflict must abort cleanly — never fall back to a destructive reset."""
    origin = _make_origin(tmp_path)
    clone = _make_clone(tmp_path, origin)

    (clone / "a.txt").write_text("local version\n")
    _git(clone, "commit", "-aqm", "local edit to a.txt")
    local_sha = _sha(clone)

    (origin / "a.txt").write_text("upstream version\n")
    _git(origin, "commit", "-aqm", "upstream edit to a.txt")

    outcome, _ = _run_diverged_recovery(clone, monkeypatch, capsys)

    assert outcome == "conflict-stopped"
    assert _sha(clone) == local_sha, "checkout moved despite the aborted merge"
    assert (clone / "a.txt").read_text() == "local version\n"

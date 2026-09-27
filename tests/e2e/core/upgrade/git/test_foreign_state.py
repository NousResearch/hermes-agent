"""Concurrent and foreign git state in the install, through a real ``hermes update``.

Users' checkouts carry state other tools left: an ``index.lock`` from a crashed (or a still
running) git, an abandoned interactive rebase, a detached HEAD from a ``git checkout <sha>``,
sometimes with work committed on it. One HEAD install over smart HTTP is shared; each cell seeds
one such state on a clean ``main``, publishes an upstream release and runs ``hermes update --yes``.

The property: the update either heals the state and lands on the release, or refuses with a
message naming the obstacle, leaving HEAD and the tree as they were and claiming nothing false;
in no case does the user's committed work end up reachable only from the reflog.
"""

from __future__ import annotations

import os
import time

import pytest

from tests.e2e.core._pending_fixes import known_failure
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.git import _git_world as G

pytestmark = G.PYTESTMARK


@pytest.fixture(scope="module")
def w(tmp_path_factory):
    with G.world(tmp_path_factory.mktemp("git-foreign"), base=I.head_sha()) as world:
        yield world


def _assert_healthy_success(w: G.World, cp, target: str) -> None:
    assert cp.returncode == 0 and G.SUCCESS in G.output(cp), f"update did not succeed:\n{w.diag(cp)}"
    assert w.head() == target and w.branch() == "main", \
        f"update reported success but HEAD={w.head()[:12]} on {w.branch()}, expected main@{target[:12]}:\n{w.diag(cp)}"
    assert not w.status(), f"update left the tree dirty:\n{w.diag(cp)}"
    version = w.version()
    assert version.returncode == 0 and G.TRACEBACK not in G.output(version), w.diag(version)


def test_crashed_git_index_lock_is_healed(w):
    w.reset_clean()
    lock = w.checkout / ".git" / "index.lock"
    lock.write_text("", encoding="utf-8")
    old = time.time() - 3600  # a git that died an hour ago, not one running now
    os.utime(lock, (old, old))
    target = w.publish("release: e2e foreign stale lock", {"e2e-foreign-stale-lock.txt": "release\n"})

    cp = w.update()

    _assert_healthy_success(w, cp, target)
    assert not lock.exists(), f"stale index.lock survived a successful update:\n{w.diag(cp)}"


def test_live_index_lock_is_refused_truthfully(w):
    w.reset_clean()
    before = w.head()
    lock = w.checkout / ".git" / "index.lock"
    lock.write_text("", encoding="utf-8")  # fresh: another git may be mid-write right now
    w.publish("release: e2e foreign live lock", {"e2e-foreign-live-lock.txt": "release\n"})

    cp = w.update()

    out = G.output(cp)
    assert cp.returncode != 0 and G.SUCCESS not in out, f"update claimed success over a live index.lock:\n{w.diag(cp)}"
    assert "index.lock" in out, f"refusal does not name the lock:\n{w.diag(cp)}"
    assert w.head() == before and not w.status(), f"refusal moved HEAD or dirtied the tree:\n{w.diag(cp)}"
    assert lock.exists(), f"update deleted a lock another git may still hold:\n{w.diag(cp)}"
    with known_failure(r"claims .*history diverged",
                       "gated on #124642: a ff-merge blocked by index.lock is reported as diverged history, "
                       "a rescue ref is written and `git reset --hard` is advised"):
        assert "diverged" not in out and not w.refs("refs/hermes-update-backups"), (
            f"an index.lock refusal claims local history diverged (and writes a rescue ref):\n{w.diag(cp)}")


def test_detached_head_at_an_older_commit_is_brought_back_to_main(w):
    w.reset_clean()
    w.git("checkout", "-q", "--detach", "HEAD~1")
    target = w.publish("release: e2e foreign detached", {"e2e-foreign-detached.txt": "release\n"})

    cp = w.update()

    _assert_healthy_success(w, cp, target)


def test_work_committed_on_a_detached_head_stays_reachable(w):
    w.reset_clean()
    w.git("checkout", "-q", "--detach", "HEAD")
    (w.checkout / "e2e-detached-work.txt").write_text("work on a detached HEAD\n", encoding="utf-8")
    w.git("add", "-A")
    w.git("commit", "-q", "-m", "work committed on a detached HEAD")
    work = w.head()
    target = w.publish("release: e2e foreign detached work", {"e2e-foreign-detached-work.txt": "release\n"})

    cp = w.update()

    assert w.head() == target or cp.returncode != 0, f"update neither landed nor refused:\n{w.diag(cp)}"
    with known_failure(r"reachable only from the reflog",
                       "gated on #124643: switching a detached HEAD to main leaves commits made on it "
                       "reachable only from the reflog, and the output never mentions them"):
        assert w.refs_containing(work) or w.head() == work, (
            f"commit {work[:12]} made on the detached HEAD is reachable only from the reflog after the update:\n"
            f"{w.diag(cp)}")


def test_abandoned_interactive_rebase_is_healed_or_refused(w):
    w.reset_clean()
    (w.checkout / "e2e-rebase-wip.txt").write_text("wip being reworded\n", encoding="utf-8")
    w.git("add", "-A")
    w.git("commit", "-q", "-m", "wip: my commit under interactive rebase")
    env = w.git_env()
    env["GIT_SEQUENCE_EDITOR"] = "sed -i 1s/^pick/edit/"
    I.git("rebase", "-i", "HEAD~1", cwd=w.checkout, env=env)  # stops at "edit": HEAD detached mid-rebase
    assert (w.checkout / ".git" / "rebase-merge").is_dir(), "could not seed a stopped interactive rebase"
    before = w.head()
    target = w.publish("release: e2e foreign rebase", {"e2e-foreign-rebase.txt": "release\n"})

    cp = w.update()

    rebase_left = (w.checkout / ".git" / "rebase-merge").exists()
    if cp.returncode != 0:
        assert "rebase" in G.output(cp).lower(), f"refused without naming the rebase:\n{w.diag(cp)}"
        assert w.head() == before, f"refusal moved HEAD:\n{w.diag(cp)}"
        return
    assert w.head() == target and w.branch() == "main", f"update reported rc=0 but did not land:\n{w.diag(cp)}"
    with known_failure(r"rebase still in progress",
                       "gated on #124644: update switches to main under a stopped rebase and leaves "
                       ".git/rebase-merge, so a later `git rebase --abort` rolls the update back"):
        assert not rebase_left, (
            f"update succeeded but left a rebase still in progress (.git/rebase-merge); `git rebase --abort` "
            f"would now reset main to {before[:12]}:\n{w.diag(cp)}")

"""A retry after a no-verdict exit is told what the previous run left behind.

`detect_crashed_workers` records a clean-exit-without-verdict as a protocol
violation and surfaces the corrective sentence to the retry worker: "if the prior
run already did the work, verify it and report the result via kanban_complete".
Verifying it began with REDISCOVERING it — nothing handed the retry what the last
run had changed, so every attempt re-derived the same state.

Measured on one install: 117 such runs across 49 tasks, 5.0 runs per task, and 4
tasks that never recovered.
"""

from __future__ import annotations

import os
import subprocess
import time

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(kb, "_HERMES_HOME_OVERRIDE", str(tmp_path), raising=False)
    yield


@pytest.fixture
def conn():
    c = kb.connect()
    try:
        yield c
    finally:
        c.close()


def _git(path, *args, env=None):
    return subprocess.run(("git", "-C", str(path)) + args, check=True,
                          capture_output=True, env=env)


def _repo(tmp_path, name="repo"):
    """A repo whose seed commit is genuinely OLD.

    The seed has to predate the run window, or it shows up as "what the previous
    run left behind" — which is what the first version of this fixture did, and
    the empty-workspace tests caught it. Dating it a day back is the difference
    between a window that means something and one that swallows the fixture.
    """
    path = tmp_path / name
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "T")
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(path, "add", "-A")
    old = dict(os.environ,
               GIT_AUTHOR_DATE="2026-08-01T10:00:00",
               GIT_COMMITTER_DATE="2026-08-01T10:00:00")
    _git(path, "commit", "-q", "-m", "seed", env=old)
    return path


VIOLATION_ERROR = ("worker exited cleanly (rc=0) without calling "
                   "kanban_complete or kanban_block — protocol violation.")


def _closed_run(conn, tid, *, error, metadata=None, started_ago=600):
    """A closed run whose window really precedes whatever came after it.

    ``_synthesize_ended_run`` stamps ``started_at == ended_at == now``, so a
    commit made a second earlier would fall outside the ``--since`` window this
    feature reads. Backdating the start is what makes the window mean anything in
    a test that finishes in under a second.
    """
    with kb.write_txn(conn):
        rid = kb._synthesize_ended_run(conn, tid, outcome="crashed",
                                       error=error, metadata=metadata)
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = ?",
                     (int(time.time()) - started_ago, rid))
    return rid


def _task_with_violation(conn, repo, *, commit=None, dirty=False):
    tid = kb.create_task(conn, title="Wire it", workspace_kind="dir",
                         workspace_path=str(repo))
    if commit:
        (repo / commit).write_text("work\n", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "feat: the work the run did not report")
    if dirty:
        (repo / "left-behind.txt").write_text("half done\n", encoding="utf-8")
    _closed_run(conn, tid, error=VIOLATION_ERROR,
                metadata={"protocol_violation": True})
    return tid


def test_the_commit_the_previous_run_did_not_report_is_named(conn, tmp_path):
    repo = _repo(tmp_path)
    tid = _task_with_violation(conn, repo, commit="feature.txt")

    ctx = kb.build_worker_context(conn, tid)

    assert "What the previous run left behind" in ctx
    assert "feat: the work the run did not report" in ctx
    # Named as UNVERIFIED: the previous run never reported it, so the retry has
    # to confirm rather than inherit the claim.
    assert "unverified" in ctx


def test_uncommitted_work_left_behind_is_counted(conn, tmp_path):
    repo = _repo(tmp_path, "repo2")
    tid = _task_with_violation(conn, repo, dirty=True)

    ctx = kb.build_worker_context(conn, tid)

    assert "Uncommitted in" in ctx
    assert "left-behind.txt" in ctx


def test_an_empty_workspace_says_so_instead_of_staying_silent(conn, tmp_path):
    """The useful half of "nothing landed": it tells the retry the work is still
    to do, instead of leaving it to hunt for work that was never there."""
    repo = _repo(tmp_path, "repo3")
    tid = _task_with_violation(conn, repo)

    ctx = kb.build_worker_context(conn, tid)

    assert "left no commit and no uncommitted change" in ctx
    assert "still to do" in ctx


def test_a_normal_failure_gets_no_evidence_block(conn, tmp_path):
    """Only the no-verdict case. A run that reported its own failure already said
    what happened, and a workspace dump would be noise on top of it."""
    repo = _repo(tmp_path, "repo4")
    tid = kb.create_task(conn, title="Wire it", workspace_kind="dir",
                         workspace_path=str(repo))
    _closed_run(conn, tid, error="pid 4242 killed by signal 9")

    ctx = kb.build_worker_context(conn, tid)
    assert "What the previous run left behind" not in ctx
    assert "left no commit" not in ctx


def test_a_scratch_workspace_is_never_probed(conn, tmp_path):
    """`scratch` is not a repo checkout; running git there would be a guess.

    The path is a REAL repo on purpose. The first version of this test used a
    scratch task with no path at all, so it passed on the empty-path check and
    said nothing about the kind — removing the kind guard left it green. Pointing
    a scratch task at a live repo is what makes the kind the only thing standing
    between this feature and a workspace it has no claim to read.
    """
    repo = _repo(tmp_path, "scratch-repo")
    (repo / "later.txt").write_text("after\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feat: something in a scratch path")
    tid = kb.create_task(conn, title="Research something",
                         workspace_kind="scratch", workspace_path=str(repo))
    assert kb.get_task(conn, tid).workspace_kind == "scratch"
    assert kb.get_task(conn, tid).workspace_path == str(repo)
    _closed_run(conn, tid, error=VIOLATION_ERROR,
                metadata={"protocol_violation": True})

    ctx = kb.build_worker_context(conn, tid)
    assert "What the previous run left behind" not in ctx
    assert "something in a scratch path" not in ctx
    assert "left no commit" not in ctx


def test_a_workspace_that_is_not_a_repo_is_silent_not_a_crash(conn, tmp_path):
    plain = tmp_path / "not-a-repo"
    plain.mkdir()
    tid = kb.create_task(conn, title="Wire it", workspace_kind="dir",
                         workspace_path=str(plain))
    _closed_run(conn, tid, error=VIOLATION_ERROR,
                metadata={"protocol_violation": True})

    # The prompt still builds. That is the contract: a prompt that cannot be
    # built is worse than a prompt without this section.
    ctx = kb.build_worker_context(conn, tid)
    assert "Wire it" in ctx
    assert "What the previous run left behind" not in ctx


def test_the_older_error_text_still_counts_as_a_violation(conn, tmp_path):
    """Runs recorded before the durable marker existed carry only the sentence."""
    repo = _repo(tmp_path, "repo5")
    tid = kb.create_task(conn, title="Wire it", workspace_kind="dir",
                         workspace_path=str(repo))
    _closed_run(conn, tid, error=VIOLATION_ERROR)

    ctx = kb.build_worker_context(conn, tid)
    assert "left no commit and no uncommitted change" in ctx

"""Process-group cohesion for worker termination (hazards 1/5/6/7).

Workers spawn with ``start_new_session=True`` so each is the leader of its own
process group. A worker that forks a grandchild (a shell, a tool subprocess)
leaves that grandchild in the SAME group. The deploy-head kill paths signalled
only the leader ``worker_pid``; the grandchild survived, the card reset to
``ready``, and the next tick spawned a DUPLICATE beside the still-running
orphan (dual execution under a single dispatcher).

These tests exercise the fix with REAL process groups (not mocked signals):

* T-C  enforce_max_runtime kills the whole group (grandchild dies) and only
       resets the card once the group is proven dead — otherwise it defers.
* T-D  the shared reclaim killer reaches the grandchild via ``os.killpg``.
* T-E  a crashed leader with a live orphan grandchild is NOT reclaimed beside
       the live process (no duplicate claim); the orphan group is SIGKILLed
       and the card is reclaimed only on a later tick once the group is dead.

POSIX-only: process groups / ``killpg`` don't exist on Windows.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

pytestmark = [
    pytest.mark.skipif(
        os.name == "nt" or not hasattr(os, "killpg"),
        reason="process groups / killpg are POSIX-only",
    ),
    # These tests deliver REAL signals to process groups they spawn
    # themselves (the whole point is to prove killpg reaches a grandchild).
    # The tests only ever signal their own leader/grandchild pids.
    pytest.mark.live_system_guard_bypass,
]


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Real process-group fixtures
# ---------------------------------------------------------------------------

_LEADER_SRC = (
    "import subprocess, sys, time\n"
    # Grandchild in the SAME process group as this leader (no setsid).
    "gc = subprocess.Popen(['sleep', '300'])\n"
    "sys.stdout.write(str(gc.pid) + '\\n')\n"
    "sys.stdout.flush()\n"
    "time.sleep(300)\n"
)


def _spawn_group():
    """Spawn a session-leader worker that forks a grandchild in its group.

    Returns ``(proc, leader_pid, pgid, grandchild_pid)``. The leader and the
    grandchild are both alive on return; ``pgid == leader_pid``.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c", _LEADER_SRC],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    line = proc.stdout.readline().strip()
    gc_pid = int(line)
    pgid = os.getpgid(proc.pid)
    assert pgid == proc.pid, "start_new_session should make the child a group leader"
    # Both must be visibly alive before the test acts.
    _wait_until(lambda: _alive(proc.pid) and _alive(gc_pid), timeout=5)
    return proc, proc.pid, pgid, gc_pid


def _alive(pid):
    """True if pid exists (any state) from this process's perspective."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _wait_until(pred, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def _reap_quiet(pid):
    """Best-effort reap so we don't leak zombies between tests."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, OSError):
        pass


def _kill_quiet(*pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


# ---------------------------------------------------------------------------
# T-C — enforce_max_runtime kills the group, never resets beside a live child
# ---------------------------------------------------------------------------


def test_enforce_max_runtime_kills_whole_group(kanban_home):
    proc, leader, pgid, gc_pid = _spawn_group()
    try:
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="tc", assignee="a")
            claimed = kb.claim_task(conn, tid)
            assert claimed is not None
            past = int(time.time()) - 100
            conn.execute(
                "UPDATE tasks SET worker_pid=?, worker_pgid=?, "
                "max_runtime_seconds=1, started_at=? WHERE id=?",
                (leader, pgid, past, tid),
            )
            conn.execute(
                "UPDATE task_runs SET started_at=? WHERE id=?",
                (past, claimed.current_run_id),
            )
            conn.commit()

            timed_out = kb.enforce_max_runtime(conn)

            assert tid in timed_out, "task past max_runtime should time out"
            task = kb.get_task(conn, tid)
            assert task.status == "ready", (
                "group proven dead → card resets to ready"
            )
        # The killpg reached BOTH the leader and the grandchild.
        assert _wait_until(lambda: not _alive(gc_pid), timeout=6), (
            "grandchild must be dead after group SIGTERM/SIGKILL"
        )
        # Leader is our Popen child — wait() reaps its zombie and confirms exit.
        proc.wait(timeout=6)
    finally:
        _kill_quiet(gc_pid, leader)
        _reap_quiet(leader)


def test_enforce_max_runtime_defers_when_group_survives(kanban_home, monkeypatch):
    """If the group refuses to die, DEFER — never reset the card beside it."""
    proc, leader, pgid, gc_pid = _spawn_group()
    try:
        # Make every signal a no-op so the group cannot actually die; the
        # code must then observe the group still alive and defer.
        monkeypatch.setattr(kb.time, "sleep", lambda *_a, **_k: None)
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="tc-defer", assignee="a")
            claimed = kb.claim_task(conn, tid)
            past = int(time.time()) - 100
            conn.execute(
                "UPDATE tasks SET worker_pid=?, worker_pgid=?, "
                "max_runtime_seconds=1, started_at=? WHERE id=?",
                (leader, pgid, past, tid),
            )
            conn.execute(
                "UPDATE task_runs SET started_at=? WHERE id=?",
                (past, claimed.current_run_id),
            )
            conn.commit()

            timed_out = kb.enforce_max_runtime(
                conn, signal_fn=lambda _t, _s: None,
            )

            assert timed_out == [], "must not report a timeout while deferring"
            task = kb.get_task(conn, tid)
            assert task.status == "running", (
                "worker still alive → hold the claim (no reset, no duplicate)"
            )
            events = [e.kind for e in kb.list_events(conn, tid)]
            assert "reclaim_deferred" in events
    finally:
        _kill_quiet(gc_pid, leader)
        _reap_quiet(leader)


# ---------------------------------------------------------------------------
# T-D — the shared reclaim killer reaches the grandchild via killpg
# ---------------------------------------------------------------------------


def test_terminate_reclaimed_worker_kills_grandchild(kanban_home):
    proc, leader, pgid, gc_pid = _spawn_group()
    try:
        host = kb._claimer_id().split(":", 1)[0]
        claim_lock = f"{host}:reclaim-test"

        info = kb._terminate_reclaimed_worker(leader, claim_lock, pgid=pgid)

        assert info["host_local"] is True
        assert info["group_kill"] is True
        assert info["terminated"] is True, (
            "group-scoped kill must confirm the whole group is gone"
        )
        assert _wait_until(lambda: not _alive(gc_pid), timeout=6), (
            "grandchild must die from the group kill, not survive a pid-only kill"
        )
        proc.wait(timeout=6)
    finally:
        _kill_quiet(gc_pid, leader)
        _reap_quiet(leader)


def test_terminate_reclaimed_worker_pid_fallback_when_pgid_null(kanban_home):
    """Legacy rows (pgid NULL) still terminate via the bare pid."""
    proc, leader, pgid, gc_pid = _spawn_group()
    try:
        host = kb._claimer_id().split(":", 1)[0]
        claim_lock = f"{host}:legacy"

        info = kb._terminate_reclaimed_worker(leader, claim_lock, pgid=None)

        assert info["group_kill"] is False
        assert info["terminated"] is True
        proc.wait(timeout=6)
    finally:
        # pid-only kill leaves the grandchild orphaned — clean it up.
        _kill_quiet(gc_pid, leader)
        _reap_quiet(leader)


# ---------------------------------------------------------------------------
# T-E — crashed leader + live orphan: no duplicate claim beside the live group
# ---------------------------------------------------------------------------


def test_detect_crashed_workers_defers_beside_live_orphan(kanban_home):
    proc, leader, pgid, gc_pid = _spawn_group()
    try:
        # Simulate a leader CRASH: kill only the leader and reap it so it is
        # not a zombie. The grandchild stays alive in the same group.
        os.kill(leader, signal.SIGKILL)
        os.waitpid(leader, 0)
        assert _wait_until(lambda: not _alive(leader), timeout=5)
        assert _alive(gc_pid), "grandchild should outlive the crashed leader"

        with kb.connect() as conn:
            tid = kb.create_task(conn, title="te", assignee="a")
            claimed = kb.claim_task(conn, tid)
            assert claimed is not None
            past = int(time.time()) - 3600  # well past the crash grace window
            conn.execute(
                "UPDATE tasks SET worker_pid=?, worker_pgid=?, started_at=? "
                "WHERE id=?",
                (leader, pgid, past, tid),
            )
            conn.commit()

            # First tick: leader dead, group alive → must NOT reclaim. No
            # duplicate claim can form beside the live orphan.
            crashed = kb.detect_crashed_workers(conn)
            assert tid not in crashed, (
                "must not reclaim (and later respawn) beside a live group member"
            )
            task = kb.get_task(conn, tid)
            assert task.status == "running", "claim held while orphan alive"
            events = [e.kind for e in kb.list_events(conn, tid)]
            assert "crash_orphan_terminated" in events, (
                "the orphan group should be SIGKILLed for cleanup"
            )

            # The orphan gets group-SIGKILLed; once it's gone the NEXT tick
            # reclaims the card as crashed.
            assert _wait_until(lambda: not _alive(gc_pid), timeout=6), (
                "orphan grandchild must be killed by the group SIGKILL"
            )
            crashed2 = kb.detect_crashed_workers(conn)
            assert tid in crashed2, "reclaim once the group is fully dead"
            task = kb.get_task(conn, tid)
            assert task.status == "ready"
    finally:
        _kill_quiet(gc_pid, leader)
        _reap_quiet(leader)


# ---------------------------------------------------------------------------
# HARNESS-FF1 — request_task_review is group-death-proof (native-review residual)
# ---------------------------------------------------------------------------


def test_request_task_review_kills_whole_group_before_release(kanban_home):
    """Review transition with a live grandchild must kill the group first.

    Canary (HARNESS-FF1): supervisor-invoked review-request must not clear
    worker_pid / release the claim beside a live process-group member.
    """
    proc, leader, pgid, gc_pid = _spawn_group()
    try:
        head = "a" * 40
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="ff1-kill", assignee="host-codex")
            claimed = kb.claim_task(conn, tid)
            assert claimed is not None
            host = kb._claimer_id().split(":", 1)[0]
            # Stamp a host-local claim_lock so the group killer acts, and
            # bind the live leader+pgid as the worker identity.
            conn.execute(
                "UPDATE tasks SET worker_pid=?, worker_pgid=?, "
                "claim_lock=? WHERE id=?",
                (leader, pgid, f"{host}:ff1-review", tid),
            )
            conn.commit()

            ok = kb.request_task_review(
                conn,
                tid,
                maker="host-codex",
                checker="grok-reviewer",
                pr="PR #99",
                head=head,
                summary="exact head ready",
                expected_run_id=claimed.current_run_id,
            )
            assert ok is True, "group proven dead → review transition must land"
            task = kb.get_task(conn, tid)
            assert task.status == "review"
            assert task.assignee == "grok-reviewer"
            assert task.worker_pid is None
            events = [e.kind for e in kb.list_events(conn, tid)]
            assert "review_requested" in events
            assert "reclaim_deferred" not in events

        assert _wait_until(lambda: not _alive(gc_pid), timeout=6), (
            "grandchild must die from group kill before slot release"
        )
        proc.wait(timeout=6)
    finally:
        _kill_quiet(gc_pid, leader)
        _reap_quiet(leader)


def test_request_task_review_defers_when_group_survives(kanban_home, monkeypatch):
    """If the group refuses to die, DEFER — never release beside the orphan."""
    proc, leader, pgid, gc_pid = _spawn_group()
    try:
        monkeypatch.setattr(kb.time, "sleep", lambda *_a, **_k: None)
        head = "b" * 40
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="ff1-defer", assignee="host-codex")
            claimed = kb.claim_task(conn, tid)
            assert claimed is not None
            host = kb._claimer_id().split(":", 1)[0]
            claim_lock = f"{host}:ff1-defer"
            conn.execute(
                "UPDATE tasks SET worker_pid=?, worker_pgid=?, "
                "claim_lock=? WHERE id=?",
                (leader, pgid, claim_lock, tid),
            )
            conn.commit()

            ok = kb.request_task_review(
                conn,
                tid,
                maker="host-codex",
                checker="grok-reviewer",
                pr="PR #100",
                head=head,
                summary="should defer",
                expected_run_id=claimed.current_run_id,
                signal_fn=lambda _t, _s: None,  # signals are no-ops
            )
            assert ok is False, "must not report success while deferring"
            task = kb.get_task(conn, tid)
            assert task.status == "running", (
                "worker still alive → hold claim (no review, no duplicate)"
            )
            assert task.assignee == "host-codex"
            assert task.worker_pid == leader
            events = [e.kind for e in kb.list_events(conn, tid)]
            assert "reclaim_deferred" in events
            assert "review_requested" not in events
            # Grandchild still live — proving we did not free the slot beside it.
            assert _alive(gc_pid), "orphan must still be alive under defer"
    finally:
        _kill_quiet(gc_pid, leader)
        _reap_quiet(leader)


def test_request_task_review_no_worker_pid_still_transitions(kanban_home):
    """Happy path with no stamped worker identity still hands off to review."""
    head = "c" * 40
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ff1-nopid", assignee="host-codex")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        ok = kb.request_task_review(
            conn,
            tid,
            maker="host-codex",
            checker="grok-reviewer",
            pr="42",
            head=head,
            summary="no pid",
            expected_run_id=claimed.current_run_id,
        )
        assert ok is True
        task = kb.get_task(conn, tid)
        assert task.status == "review"
        assert task.assignee == "grok-reviewer"

"""Tests: reclaim paths are claim-lock-aware so they can't desync a re-claimed
task (issue #36910).

A stale crash/stale-claim/max-runtime reclaim, computed from a snapshot of an
OLD worker, used to reset ``tasks.status`` back to ``ready`` with only a
``WHERE status='running'`` guard. If the task had since been reclaimed AND
re-claimed by a NEW worker (new run, new claim_lock, live pid), that stale
UPDATE clobbered the live task: ``tasks.status='ready'`` while the new
``task_runs.status='running'`` and the worker kept executing — the board showed
the task in the Ready lane and the dispatcher could treat live work as
available. The reset is now gated on the snapshot's ``claim_lock`` (and pid),
so it only fires when the task is still owned by the worker the reclaim was
computed for.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect() as c:
        yield c


def test_stale_crash_reset_rejected_for_reclaimed_task(conn):
    """A reset carrying an OLD worker's claim_lock must NOT clobber a task
    that has since been re-claimed by a new worker."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="desync", assignee="w")

    # Worker A claims, then dies.
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kbd._set_worker_pid(conn, tid, dead.pid)
    old = conn.execute(
        "SELECT claim_lock, worker_pid FROM tasks WHERE id=?", (tid,)
    ).fetchone()

    # Reclaim + re-claim by worker B (alive).
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, current_run_id=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()
    kb.claim_task(conn, tid, claimer=f"{host}:B")
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        kbd._set_worker_pid(conn, tid, sleeper.pid)

        # The stale reset for worker A — same shape as the guarded UPDATE in
        # detect_crashed_workers — must reject (rowcount 0) because B owns it.
        cur = conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL "
            "WHERE id=? AND status='running' AND worker_pid=? AND claim_lock IS ?",
            (tid, old["worker_pid"], old["claim_lock"]),
        )
        conn.commit()
        assert cur.rowcount == 0, "stale reclaim wrongly clobbered the re-claimed task"

        final = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        assert final["status"] == "running"
        assert final["claim_lock"] == f"{host}:B"
    finally:
        sleeper.terminate()


def test_genuine_crash_still_reclaims(conn):
    """When the claim_lock still matches the dead worker, the crash reclaim
    fires normally — the guard must not break the legitimate path."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="legit", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kbd._set_worker_pid(conn, tid, dead.pid)
    # Rewind started_at so the launch grace window doesn't skip the check.
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.execute(
        "UPDATE task_runs SET started_at = started_at - 9999 WHERE task_id=?", (tid,)
    )
    conn.commit()
    kbd._record_worker_exit(dead.pid, 1 << 8)  # nonzero exit → crash

    crashed = kbd.detect_crashed_workers(conn)
    assert tid in crashed
    final = conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
    assert final["status"] in ("ready", "blocked", "todo")


def _dead_pid() -> int:
    """A pid that has already exited, portable across platforms (no ``true``)."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_account_crashes_rejects_task_reclaimed_between_reclaim_and_accounting(conn):
    """``_reclaim_dead_workers`` commits its reclaim txn, then
    ``_account_crashes`` runs failure accounting in a SEPARATE txn. A new
    worker that claims the task in between must not have its run's task row
    clobbered by the old crash's stale error/status — but the crash must
    still count toward the unified consecutive-failures budget."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="race", assignee="w")

    kb.claim_task(conn, tid, claimer=f"{host}:A")
    dead_pid = _dead_pid()
    kbd._set_worker_pid(conn, tid, dead_pid)
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.execute("UPDATE task_runs SET started_at = started_at - 9999 WHERE task_id=?", (tid,))
    conn.commit()
    kbd._record_worker_exit(dead_pid, 1 << 8)  # nonzero exit → crash

    sweep = kbd._reclaim_dead_workers(conn)
    assert tid in sweep.crashed

    # A new worker claims the task before the crash is accounted for, and is
    # still running when accounting runs.
    kb.claim_task(conn, tid, claimer=f"{host}:B")

    kbd._account_crashes(conn, sweep.crash_details)

    final = conn.execute(
        "SELECT last_failure_error, consecutive_failures, status FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert final["status"] == "running"
    assert final["last_failure_error"] is None, (
        "old run's crash accounting must not clobber the new run's task row"
    )
    assert final["consecutive_failures"] == 1, (
        "the old crash must still count toward the unified failure budget"
    )


def test_account_crashes_still_counts_after_replacement_run_already_finished(conn):
    """ABA case: the replacement run claimed between reclaim and accounting has
    ALSO already finished (so ``current_run_id`` is NULL again) by the time
    late accounting runs. A nullity check on ``current_run_id`` would treat
    this as safe to overwrite; accounting must instead detect the later run
    via ``task_runs`` and still only bump the counter."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="aba", assignee="w")

    kb.claim_task(conn, tid, claimer=f"{host}:A")
    dead_pid = _dead_pid()
    kbd._set_worker_pid(conn, tid, dead_pid)
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.execute("UPDATE task_runs SET started_at = started_at - 9999 WHERE task_id=?", (tid,))
    conn.commit()
    kbd._record_worker_exit(dead_pid, 1 << 8)  # nonzero exit → crash

    sweep = kbd._reclaim_dead_workers(conn)
    assert tid in sweep.crashed

    # A new worker claims the task and finishes it before the old crash is
    # accounted for — ``current_run_id`` is NULL again, same as right after
    # the original reclaim.
    kb.claim_task(conn, tid, claimer=f"{host}:B")
    kb.complete_task(conn, tid, summary="done")

    kbd._account_crashes(conn, sweep.crash_details)

    final = conn.execute(
        "SELECT last_failure_error, consecutive_failures, status FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert final["status"] == "done", "late accounting must not reopen the completed task"
    assert final["last_failure_error"] is None
    assert final["consecutive_failures"] == 1


def test_record_task_failure_timeout_still_running_counts_without_clobbering(conn):
    """Same CAS protection, exercised via the ``timed_out`` outcome that
    ``enforce_max_runtime`` passes to ``_record_task_failure``."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="timeout-race", assignee="w")

    kb.claim_task(conn, tid, claimer=f"{host}:A")
    old_run_id = kb._end_run(
        conn, tid, outcome="timed_out", status="timed_out", error="elapsed 99s > limit 1s"
    )
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()

    # A new worker claims the task before the timeout is accounted for, and
    # is still running when accounting runs.
    kb.claim_task(conn, tid, claimer=f"{host}:B")

    kbd._record_task_failure(
        conn, tid, error="elapsed 99s > limit 1s", outcome="timed_out",
        release_claim=False, end_run=False, reclaimed_run_id=old_run_id,
    )

    final = conn.execute(
        "SELECT last_failure_error, consecutive_failures, status FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert final["status"] == "running"
    assert final["last_failure_error"] is None
    assert final["consecutive_failures"] == 1


def test_record_task_failure_timeout_still_counts_after_replacement_run_already_finished(conn):
    """ABA case for the timeout path: the replacement run has also already
    finished by the time late accounting runs."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="timeout-aba", assignee="w")

    kb.claim_task(conn, tid, claimer=f"{host}:A")
    old_run_id = kb._end_run(
        conn, tid, outcome="timed_out", status="timed_out", error="elapsed 99s > limit 1s"
    )
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()

    kb.claim_task(conn, tid, claimer=f"{host}:B")
    kb.complete_task(conn, tid, summary="done")

    kbd._record_task_failure(
        conn, tid, error="elapsed 99s > limit 1s", outcome="timed_out",
        release_claim=False, end_run=False, reclaimed_run_id=old_run_id,
    )

    final = conn.execute(
        "SELECT last_failure_error, consecutive_failures, status FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    assert final["status"] == "done"
    assert final["last_failure_error"] is None
    assert final["consecutive_failures"] == 1

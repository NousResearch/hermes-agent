"""Kanban lifecycle process-safety regressions.

Two production incidents, one coherent lifecycle contract: a task must not
be handed to a new worker while the *previous* decision about it is still
binding.

**Bug A — the breaker decision was not binding.**
``t_27263082`` recorded ``gave_up`` after two spawn failures with payload
``effective_limit=2 / limit_source=dispatcher``. A later ``recompute_ready``
call resolved a *different* effective limit (its own default / a differently
configured caller), decided the task was still under budget, promoted it back
to ``ready``, and the dispatcher retried it. The limit that actually tripped
the breaker must be persisted and honoured by every later caller — restart,
config change, or default mismatch included. The only legitimate exits stay
explicit: ``kanban_unblock``, operator reclaim/promote, or a success.

**Bug B — the requeue decision outran the OS.**
``t_80a3542a`` run 141 hit its iteration/runtime budget. The dispatcher
signalled the worker PID, did not confirm the process group was gone (PID
98725 was still alive, reparented to PID 1, with live Claude children),
flipped the task to ``ready`` anyway, and immediately claimed run 152 with a
second worker (PID 57286) against the same task and workspace: two live
writers. A timeout/reclaim must either bound the termination of the whole
process group or leave a durable fence that keeps the task nonclaimable
until that group is provably gone.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def no_sleep(monkeypatch):
    """Collapse the termination grace polling so tests stay fast."""
    monkeypatch.setattr(kb.time, "sleep", lambda *_a, **_k: None)


@pytest.fixture
def live_worker():
    """A real, session-leading child process (pgid == pid), reaped on exit."""
    procs = []

    def _spawn():
        proc = subprocess.Popen(  # noqa: S603 - fixed argv
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        procs.append(proc)
        return proc

    yield _spawn

    for proc in procs:
        try:
            # getattr: a test may have removed killpg to exercise the
            # no-process-groups (Windows-shaped) degradation path.
            killpg = getattr(os, "killpg", None)
            if killpg is not None:
                killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                os.kill(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, OSError, AttributeError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            pass


@pytest.fixture
def orphaned_group():
    """A process group whose LEADER has exited but whose child is still alive.

    This is the ``t_80a3542a`` shape: the worker dies, its Claude children
    reparent to PID 1, keep the worker's process group, and keep writing.
    Returns ``(leader_pid, pgid, child_pid)`` with the leader already reaped.
    """
    state = {}

    def _spawn():
        leader = subprocess.Popen(  # noqa: S603 - fixed argv
            [
                sys.executable, "-c",
                "import os, subprocess, sys, time\n"
                "child = subprocess.Popen([sys.executable, '-c',"
                " 'import time; time.sleep(120)'])\n"
                "print(os.getpid(), os.getpgid(0), child.pid, flush=True)\n",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        line = leader.stdout.readline().split()
        leader.wait(timeout=10)  # leader exits and is reaped: truly gone
        leader_pid, pgid, child_pid = (int(x) for x in line)
        state["pgid"] = pgid
        state["child"] = child_pid
        return leader_pid, pgid, child_pid

    yield _spawn

    if state:
        try:
            os.killpg(state["pgid"], signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


def _kill_and_reap(pid):
    """SIGKILL ``pid`` and wait until it is really gone (not just a zombie)."""
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            return
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _events(conn, task_id, kind=None):
    sql = "SELECT kind, payload FROM task_events WHERE task_id = ?"
    args = [task_id]
    if kind is not None:
        sql += " AND kind = ?"
        args.append(kind)
    sql += " ORDER BY id"
    return [
        (r["kind"], json.loads(r["payload"]) if r["payload"] else {})
        for r in conn.execute(sql, args).fetchall()
    ]


def _trip_breaker(conn, task_id, *, failure_limit, attempts=None):
    """Drive the real spawn-failure path until the breaker trips."""
    attempts = attempts if attempts is not None else failure_limit
    tripped = False
    for i in range(attempts):
        assert kb.claim_task(conn, task_id) is not None, f"claim {i} failed"
        tripped = kb._record_task_failure(
            conn, task_id,
            error="spawn failed: profile not found",
            outcome="spawn_failed",
            failure_limit=failure_limit,
            release_claim=True,
            end_run=True,
        )
    return tripped


def _make_running_with_worker(conn, task_id, pid, *, pgid=None, elapsed=600):
    """Claim ``task_id`` and attach a worker pid whose runtime cap is blown."""
    claimed = kb.claim_task(conn, task_id)
    assert claimed is not None
    kb._set_worker_pid(conn, task_id, int(pid), pgid=pgid)
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET max_runtime_seconds = 1, started_at = ? WHERE id = ?",
            (now - elapsed, task_id),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = ?",
            (now - elapsed, kb.get_task(conn, task_id).current_run_id),
        )
    return kb.get_task(conn, task_id)


# ===========================================================================
# Bug A — the recorded breaker decision binds every later caller
# ===========================================================================


def test_gave_up_records_the_effective_limit_that_tripped(kanban_home):
    """The trip must persist *which* limit it decided against."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="breaker", assignee="a")
        assert _trip_breaker(conn, tid, failure_limit=2) is True

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures == 2

        gave_up = _events(conn, tid, "gave_up")
        assert len(gave_up) == 1
        assert gave_up[0][1]["effective_limit"] == 2
        assert gave_up[0][1]["limit_source"] == "dispatcher"

        # And it is readable from the task row, not just the event log, so
        # every promotion caller can honour it without replaying events.
        assert kb.recorded_breaker_limit(conn, tid) == 2


def test_recompute_with_a_lenient_caller_limit_cannot_repromote(kanban_home):
    """The exact t_27263082 bug.

    Breaker tripped at effective_limit=2 (limit_source=dispatcher). A later
    ``recompute_ready`` from a caller carrying a larger limit must not
    promote the task: 2 failures already exhausted the limit the breaker
    actually decided against.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t_27263082 shape", assignee="a")
        _trip_breaker(conn, tid, failure_limit=2)

        for caller_limit in (None, 3, 4, 10):
            promoted = (
                kb.recompute_ready(conn)
                if caller_limit is None
                else kb.recompute_ready(conn, failure_limit=caller_limit)
            )
            assert promoted == 0, f"caller_limit={caller_limit} re-promoted"
            assert kb.get_task(conn, tid).status == "blocked"


def test_recorded_limit_survives_restart_and_dispatcher_config_drift(
    kanban_home,
):
    """Gateway restart + a raised ``kanban.failure_limit`` must not resurrect
    a task the breaker already gave up on."""
    spawned = []

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="restart", assignee="a")
        _trip_breaker(conn, tid, failure_limit=1, attempts=1)
        assert kb.get_task(conn, tid).status == "blocked"

    # Fresh process/connection, dispatcher now configured far more leniently.
    with kb.connect() as conn:
        res = kb.dispatch_once(
            conn,
            spawn_fn=lambda *a, **k: spawned.append(a) or 4321,
            failure_limit=9,
        )
        assert res.promoted == 0, "gave_up task promoted after restart"
        assert spawned == [], "dispatcher respawned a gave_up task"
        assert kb.get_task(conn, tid).status == "blocked"


def test_task_max_retries_trip_is_also_binding(kanban_home):
    """A per-task ``max_retries`` trip (limit_source=task) is just as sticky
    when a later caller passes a larger dispatcher limit."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="task-limit", assignee="a")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET max_retries = 1 WHERE id = ?", (tid,),
            )
        _trip_breaker(conn, tid, failure_limit=5, attempts=1)
        assert kb.get_task(conn, tid).status == "blocked"
        assert _events(conn, tid, "gave_up")[0][1]["limit_source"] == "task"

        assert kb.recompute_ready(conn, failure_limit=9) == 0
        assert kb.get_task(conn, tid).status == "blocked"


def test_explicit_unblock_clears_the_recorded_breaker_decision(kanban_home):
    """``kanban_unblock`` is the operator's deliberate fresh start: the
    recorded trip must not survive it."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="unblock me", assignee="a")
        _trip_breaker(conn, tid, failure_limit=2)
        assert kb.get_task(conn, tid).status == "blocked"

        assert kb.unblock_task(conn, tid) is True
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert kb.recorded_breaker_limit(conn, tid) is None

        # A later blocked-with-failures state resolves against the caller's
        # limit again (the #35072 contract), because nothing is recorded.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='blocked', consecutive_failures=1 "
                "WHERE id = ?",
                (tid,),
            )
        assert kb.recompute_ready(conn, failure_limit=5) == 1
        assert kb.get_task(conn, tid).status == "ready"


def test_legacy_gave_up_rows_are_backfilled_on_migration(kanban_home):
    """Boards written before this fix carry the decision only in the event
    payload. Migration must lift it onto the task row so the historical
    ``t_27263082`` shape is fenced too."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="legacy", assignee="a")
        _trip_breaker(conn, tid, failure_limit=2)
        # Simulate a row written by the pre-fix code: event payload only.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET breaker_limit = NULL WHERE id = ?", (tid,),
            )

    with kb.connect() as conn:  # re-open → migration/backfill runs
        assert kb.recorded_breaker_limit(conn, tid) == 2
        assert kb.recompute_ready(conn, failure_limit=9) == 0
        assert kb.get_task(conn, tid).status == "blocked"


def test_parent_gated_todo_promotion_still_works(kanban_home):
    """The breaker fence must not touch ordinary dependency promotion."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="a")
        child = kb.create_task(conn, title="child", assignee="a", parents=[parent])
        assert kb.get_task(conn, child).status == "todo"

        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, result="done")
        # complete_task recomputes inline; the child is promoted there.
        assert kb.get_task(conn, child).status == "ready"

        # A blocked-on-dependency child (no breaker trip on record) is still
        # promoted by a later recompute once its parent is done.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='blocked' WHERE id = ?", (child,),
            )
        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, child).status == "ready"


def test_worker_initiated_block_stays_sticky(kanban_home):
    """#28712 contract is untouched: an explicit worker block never
    auto-promotes, with or without a recorded breaker limit."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="review please", assignee="a")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid,
            reason="review-required: verify ACL change",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        for _ in range(3):
            assert kb.recompute_ready(conn, failure_limit=9) == 0
            assert kb.get_task(conn, tid).status == "blocked"


def test_completed_task_is_terminal(kanban_home):
    """A done task is never re-promoted or re-claimed, even with a stale
    ``gave_up`` in its history."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="finished", assignee="a")
        _trip_breaker(conn, tid, failure_limit=2)
        assert kb.unblock_task(conn, tid) is True
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="shipped")
        assert kb.get_task(conn, tid).status == "done"

        assert kb.recompute_ready(conn, failure_limit=9) == 0
        assert kb.claim_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == "done"


# ===========================================================================
# Bug B — no replacement worker while the old process group can still write
# ===========================================================================


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_timeout_does_not_requeue_while_worker_group_is_alive(
    kanban_home, live_worker, no_sleep, monkeypatch,
):
    """The exact t_80a3542a bug: budget exhausted, worker still alive.

    ``enforce_max_runtime`` must not flip the task to ``ready`` (which is
    what let the dispatcher claim run 152 beside live run 141). The task
    stays ``running``, the run stays open, and the hold is visible.
    """
    proc = live_worker()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t_80a3542a shape", assignee="a")
        task = _make_running_with_worker(conn, tid, proc.pid)
        run_id = task.current_run_id

        # signal_fn/killpg_fn are no-ops: the worker ignores our signals,
        # exactly like a process wedged in an uninterruptible syscall.
        timed_out = kb.enforce_max_runtime(
            conn,
            signal_fn=lambda *a, **k: None,
            killpg_fn=lambda *a, **k: None,
        )

        assert timed_out == []
        task = kb.get_task(conn, tid)
        assert task.status == "running", "requeued while worker still alive"
        assert task.current_run_id == run_id, "run finalized under a live worker"
        assert task.consecutive_failures == 0
        assert [k for k, _ in _events(conn, tid)].count("timed_out") == 0
        deferred = _events(conn, tid, "reclaim_deferred")
        assert deferred, "the hold must be recorded for operators"
        assert deferred[-1][1]["prev_pid"] == proc.pid

        # Same through a full dispatcher tick, with the worker still
        # ignoring signals (a D-state / throttled worker): no requeue, no
        # second spawn — this is the tick that produced run 152.
        spawned = []
        with monkeypatch.context() as m:
            m.setattr(kb.os, "kill", lambda *a, **k: None)
            m.setattr(kb.os, "killpg", lambda *a, **k: None)
            kb.dispatch_once(conn, spawn_fn=lambda *a, **k: spawned.append(a) or 1)
        assert spawned == []
        assert kb.get_task(conn, tid).status == "running"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_timeout_termination_targets_the_whole_process_group(
    kanban_home, live_worker, no_sleep,
):
    """Signalling only the worker PID leaves its Claude children alive (they
    reparent to PID 1 and keep writing). Terminate the group."""
    proc = live_worker()
    pgid = os.getpgid(proc.pid)
    calls = []

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="group kill", assignee="a")
        _make_running_with_worker(conn, tid, proc.pid)
        assert kb.get_task(conn, tid).worker_pgid == pgid

        kb.enforce_max_runtime(
            conn,
            signal_fn=lambda *a, **k: None,
            killpg_fn=lambda *a, **k: calls.append(a),
        )

    assert calls, "no process-group signal was sent"
    assert {c[0] for c in calls} == {pgid}
    assert signal.SIGTERM in {c[1] for c in calls}
    assert signal.SIGKILL in {c[1] for c in calls}, "no bounded escalation"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_timeout_requeues_once_the_group_is_gone(kanban_home, live_worker, no_sleep):
    """Normal path preserved: when termination is confirmed, the task is
    requeued exactly once with a ``timed_out`` run and a counted failure."""
    proc = live_worker()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="clean timeout", assignee="a")
        task = _make_running_with_worker(conn, tid, proc.pid)
        run_id = task.current_run_id

        timed_out = kb.enforce_max_runtime(conn)  # real signals: group dies

        assert timed_out == [tid]
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.current_run_id is None
        assert task.consecutive_failures == 1
        run = kb.get_run(conn, run_id)
        assert run.outcome == "timed_out"
        assert [k for k, _ in _events(conn, tid)].count("timed_out") == 1

        # And nothing fences the (now dead) worker: the retry can claim.
        assert kb.claim_task(conn, tid) is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_claim_is_fenced_until_the_prior_process_group_exits(
    kanban_home, live_worker, no_sleep,
):
    """Durable nonclaimable reconciliation: a task that reached ``ready``
    while its previous worker group was still alive must not be claimed —
    across restarts, since the fence lives on the row."""
    proc = live_worker()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="fenced", assignee="a")
        task = _make_running_with_worker(conn, tid, proc.pid)
        run_id = task.current_run_id
        # A path outside our control (crash reaper, manual SQL, an older
        # gateway) requeued the task while the group was still alive.
        kb.record_worker_fence(
            conn, tid,
            pid=proc.pid,
            pgid=os.getpgid(proc.pid),
            run_id=run_id,
            claim_lock=task.claim_lock,
            reason="requeued_worker_alive",
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='ready', claim_lock=NULL, "
                "claim_expires=NULL, worker_pid=NULL, current_run_id=NULL "
                "WHERE id = ?",
                (tid,),
            )

    with kb.connect() as conn:  # restart
        assert kb.claim_task(conn, tid) is None, "claimed beside a live worker"
        rejected = _events(conn, tid, "claim_rejected")
        assert rejected and rejected[-1][1]["reason"] == "prior_worker_alive"
        assert rejected[-1][1]["fence_run_id"] == run_id
        assert kb.get_task(conn, tid).status == "ready"

        spawned = []
        kb.dispatch_once(conn, spawn_fn=lambda *a, **k: spawned.append(a) or 1)
        assert spawned == [], "dispatcher spawned a duplicate writer"

        # Once the group is gone the fence clears itself and work resumes.
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=5)
        assert kb.claim_task(conn, tid) is not None
        assert kb.get_task(conn, tid).status == "running"
        assert kb.worker_fence(conn, tid) is None


@pytest.mark.live_system_guard_bypass  # signals only our own orphaned group
@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_crashed_worker_with_live_children_is_fenced(
    kanban_home, orphaned_group, no_sleep,
):
    """The worker PID is gone but the children it forked are not.

    They reparent to PID 1, keep the worker's process group, and keep
    writing to the workspace — the ``t_80a3542a`` shape. The crash reaper
    may requeue the task, but it must not become claimable. Real processes
    throughout: a group-leading worker that has exited and been reaped, with
    a live child still holding its group.
    """
    leader_pid, pgid, _child_pid = orphaned_group()

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="orphaned children", assignee="a")
        claimed = kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, leader_pid, pgid=pgid)
        run_id = kb.get_task(conn, tid).current_run_id
        assert kb.get_task(conn, tid).worker_pgid == pgid
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?",
                (int(time.time()) - 3600, tid),
            )

        crashed = kb.detect_crashed_workers(conn)
        assert tid in crashed

        fence = kb.worker_fence(conn, tid)
        assert fence is not None, "requeued with live children and no fence"
        assert fence["pgid"] == pgid
        assert fence["run_id"] == run_id
        assert fence["claim_lock"] == claimed.claim_lock

        assert kb.claim_task(conn, tid) is None
        rejected = _events(conn, tid, "claim_rejected")
        assert rejected[-1][1]["reason"] == "prior_worker_alive"


def test_run_finalization_is_exactly_once(kanban_home):
    """A run may be finalized once. A second finalization (racing reaper vs
    worker completion) must not reopen it, rewrite its outcome, or emit a
    second terminal row."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="one finalize", assignee="a")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id

        with kb.write_txn(conn):
            first = kb._end_run(conn, tid, outcome="timed_out", status="timed_out")
        assert first == run_id

        with kb.write_txn(conn):
            second = kb._end_run(conn, tid, outcome="completed", status="done")
        assert second is None, "second finalization must be a no-op"

        run = kb.get_run(conn, run_id)
        assert run.outcome == "timed_out"
        assert run.ended_at is not None
        open_runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ? AND ended_at IS NULL",
            (tid,),
        ).fetchone()[0]
        assert open_runs == 0


# ===========================================================================
# Review round 2 — every reclaim path fences, and the fence is identity-safe
# ===========================================================================


@pytest.mark.live_system_guard_bypass  # signals only our own orphaned group
@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_stale_running_reclaim_fences_orphaned_children(
    kanban_home, orphaned_group, no_sleep,
):
    """``detect_stale_running`` is a third requeue path and must fence too.

    Dead leader, live child holding the worker's process group: the
    heartbeat reaper may requeue the task, but a replacement worker must
    not be claimable until those children are gone.
    """
    leader_pid, pgid, child_pid = orphaned_group()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="stale with children", assignee="a")
        claimed = kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, leader_pid, pgid=pgid)
        run_id = kb.get_task(conn, tid).current_run_id
        old = int(time.time()) - 100_000
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ?, last_heartbeat_at = NULL "
                "WHERE id = ?",
                (old, tid),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE id = ?", (old, run_id),
            )

        assert kb.detect_stale_running(conn, stale_timeout_seconds=60) == [tid]

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.worker_pgid is None, "stale worker_pgid residue"
        fence = kb.worker_fence(conn, tid)
        assert fence is not None, "requeued with live children and no fence"
        assert fence["pgid"] == pgid
        assert fence["run_id"] == run_id
        assert fence["claim_lock"] == claimed.claim_lock

        assert kb.claim_task(conn, tid) is None
        assert _events(conn, tid, "claim_rejected")[-1][1]["reason"] == (
            "prior_worker_alive"
        )

        _kill_and_reap(child_pid)
        assert kb.claim_task(conn, tid) is not None
        assert kb.worker_fence(conn, tid) is None


@pytest.mark.live_system_guard_bypass  # signals only our own orphaned group
@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_release_stale_claims_fences_orphaned_children(
    kanban_home, orphaned_group, no_sleep,
):
    """Same contract on the TTL-expiry path, exercised end to end."""
    leader_pid, pgid, child_pid = orphaned_group()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ttl with children", assignee="a")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, leader_pid, pgid=pgid)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_expires = ? WHERE id = ?",
                (int(time.time()) - 60, tid),
            )

        assert kb.release_stale_claims(conn) == 1
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.worker_pgid is None
        assert kb.worker_fence(conn, tid) is not None
        assert kb.claim_task(conn, tid) is None

        _kill_and_reap(child_pid)
        assert kb.claim_task(conn, tid) is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_foreign_host_claim_is_never_fenced_from_local_pids(
    kanban_home, live_worker, no_sleep,
):
    """A claim held by ANOTHER host says nothing about local PIDs.

    Its pid/pgid numbers are meaningful only on that host; a local process
    that happens to share the number must not fence the task forever — we
    have no way to ever observe the remote worker exiting.
    """
    proc = live_worker()  # local process colliding with the remote pid number
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="foreign claim", assignee="a")
        kb.claim_task(conn, tid)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_lock = ?, claim_expires = ?, "
                "worker_pid = ?, worker_pgid = ?, worker_identity = NULL "
                "WHERE id = ?",
                (
                    "some-other-host:4242", int(time.time()) - 60,
                    proc.pid, os.getpgid(proc.pid), tid,
                ),
            )

        assert kb.release_stale_claims(conn) == 1
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.worker_fence(conn, tid) is None, (
            "fenced a foreign-host claim from a local pid collision"
        )
        assert kb.claim_task(conn, tid) is not None
        assert proc.poll() is None, "signalled a process we do not own"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_fence_does_not_survive_pid_reuse(kanban_home, live_worker, no_sleep):
    """A fence pins process *identity*, not a recycled number.

    The recorded worker is long gone; the OS handed its pid to something
    unrelated. Holding the task hostage to a stranger's lifetime would strand
    it forever.
    """
    proc = live_worker()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="reused pid", assignee="a")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        kb.record_worker_fence(
            conn, tid,
            pid=proc.pid,
            pgid=os.getpgid(proc.pid),
            run_id=1,
            claim_lock=kb._claimer_id(),
            reason="unit",
            identity="definitely-not-this-process",
        )

        assert kb.claim_task(conn, tid) is not None, "fenced by a recycled pid"
        assert kb.worker_fence(conn, tid) is None
        cleared = _events(conn, tid, "worker_fence_cleared")
        assert cleared and cleared[-1][1]["cleared_reason"] == "worker_identity_changed"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_fence_held_by_children_only_is_bounded(
    kanban_home, orphaned_group, no_sleep,
):
    """A fence with no verifiable leader is bounded in time.

    Children can hold a process group for a very long time (or a recycled
    pgid can look like they do). While the leader itself is verifiably
    alive the fence never expires, but a group-only hold gives up after
    ``WORKER_FENCE_MAX_SECONDS`` rather than stranding the task.
    """
    _leader_pid, pgid, _child = orphaned_group()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="bounded fence", assignee="a")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        kb.record_worker_fence(
            conn, tid,
            pid=_leader_pid, pgid=pgid, run_id=1,
            claim_lock=kb._claimer_id(), reason="unit",
            identity="leader-is-gone",
        )
        # Fresh fence: children hold it.
        assert kb.claim_task(conn, tid) is None

        # Age it past the bound.
        fence = kb.worker_fence(conn, tid)
        fence["since"] = int(time.time()) - kb.WORKER_FENCE_MAX_SECONDS - 1
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_fence = ? WHERE id = ?",
                (json.dumps(fence), tid),
            )
        assert kb.claim_task(conn, tid) is not None
        expired = _events(conn, tid, "worker_fence_expired")
        assert expired, "expiry must be auditable"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_operator_recovery_clears_a_fence(kanban_home, live_worker, no_sleep):
    """Explicit operator actions are the escape hatch for a stuck fence."""
    for action in ("unblock", "promote"):
        proc = live_worker()
        with kb.connect() as conn:
            tid = kb.create_task(conn, title=f"escape-{action}", assignee="a")
            kb.record_worker_fence(
                conn, tid,
                pid=proc.pid, pgid=os.getpgid(proc.pid), run_id=1,
                claim_lock=kb._claimer_id(), reason="unit",
                identity=kb._process_identity(proc.pid),
            )
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,),
                )
            # Still fenced by a genuinely live worker...
            assert kb.worker_fence(conn, tid) is not None

            if action == "unblock":
                assert kb.unblock_task(conn, tid) is True
            else:
                ok, _err = kb.promote_task(conn, tid, actor="operator")
                assert ok

            assert kb.worker_fence(conn, tid) is None, (
                f"{action} left the task fenced with no operator escape"
            )
            assert kb.claim_task(conn, tid) is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_group_signal_requires_verified_leader_identity(
    kanban_home, live_worker, no_sleep,
):
    """Never blast a process group we cannot prove is still ours.

    PIDs and PGIDs are recycled. Signalling a group whose leader identity no
    longer matches (or was never recorded) would kill an unrelated group.
    """
    proc = live_worker()
    pgid = os.getpgid(proc.pid)

    # Identity matches → the group is ours, group signalling is allowed.
    group_calls = []
    kb._terminate_reclaimed_worker(
        proc.pid, kb._claimer_id(),
        signal_fn=lambda *a: None,
        killpg_fn=lambda *a: group_calls.append(a),
        pgid=pgid,
        identity=kb._process_identity(proc.pid),
    )
    assert group_calls and group_calls[0][0] == pgid

    # Identity mismatch (pid was recycled) → no group signal at all.
    stale_calls = []
    pid_calls = []
    info = kb._terminate_reclaimed_worker(
        proc.pid, kb._claimer_id(),
        signal_fn=lambda *a: pid_calls.append(a),
        killpg_fn=lambda *a: stale_calls.append(a),
        pgid=pgid,
        identity="not-the-recorded-process",
    )
    assert stale_calls == [], "blasted a recycled process group"
    assert info["terminated"] is True, "a recycled pid means our worker is gone"
    assert info["still_alive"] is False

    # No recorded identity (legacy row) → fail closed on the group too.
    legacy_calls = []
    kb._terminate_reclaimed_worker(
        proc.pid, kb._claimer_id(),
        signal_fn=lambda *a: None,
        killpg_fn=lambda *a: legacy_calls.append(a),
        pgid=pgid,
        identity=None,
    )
    assert legacy_calls == [], "group-signalled an unverifiable process group"


def test_force_trip_gave_up_is_binding_below_the_limit(kanban_home):
    """A recorded trip binds on its own, not on a counter comparison.

    ``force_trip`` (protocol-violation streak, systemic-error trip) records
    ``gave_up`` with ``failures`` BELOW ``effective_limit``. Comparing the
    counter against the limit let those tasks auto-promote on the very next
    recompute — the give-up decision has to bind until something explicitly
    resets it.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="force tripped", assignee="a")
        kb.claim_task(conn, tid)
        assert kb._record_task_failure(
            conn, tid,
            error="clean exit without terminal tool call",
            outcome="crashed",
            failure_limit=5,
            force_trip=True,
            release_claim=True,
            end_run=True,
        ) is True

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures < 5, "precondition: below the limit"
        assert kb.recorded_breaker_limit(conn, tid) is not None

        for caller_limit in (None, 5, 99):
            promoted = (
                kb.recompute_ready(conn) if caller_limit is None
                else kb.recompute_ready(conn, failure_limit=caller_limit)
            )
            assert promoted == 0, f"force-tripped task promoted ({caller_limit})"
            assert kb.get_task(conn, tid).status == "blocked"

        # Explicit operator recovery still works.
        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.recorded_breaker_limit(conn, tid) is None


def test_breaker_limit_is_not_stamped_on_a_terminal_race(kanban_home):
    """The trip stamp rides the status-guarded transition.

    If the worker completes the task while the dispatcher is accounting a
    failure, the guarded UPDATE matches nothing — and the breaker stamp must
    not land either, or a finished task carries a binding give-up decision.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="racy completion", assignee="a")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="done before the reaper looked")
        assert kb.get_task(conn, tid).status == "done"

        kb._record_task_failure(
            conn, tid,
            error="reaper thought this crashed",
            outcome="crashed",
            failure_limit=1,
            release_claim=False,
            end_run=False,
        )

        task = kb.get_task(conn, tid)
        assert task.status == "done", "terminal task reopened"
        assert task.breaker_limit is None, "stamped a completed task"
        assert kb.recompute_ready(conn) == 0


def test_terminal_transitions_clear_worker_process_state(kanban_home, live_worker):
    """No stale pgid residue: whatever clears worker_pid clears worker_pgid."""
    proc = live_worker()
    with kb.connect() as conn:
        for finisher in ("complete", "block"):
            tid = kb.create_task(conn, title=f"residue-{finisher}", assignee="a")
            kb.claim_task(conn, tid)
            kb._set_worker_pid(conn, tid, proc.pid)
            assert kb.get_task(conn, tid).worker_pgid is not None
            run_id = kb.get_task(conn, tid).current_run_id
            if finisher == "complete":
                kb.complete_task(conn, tid, result="ok")
            else:
                kb.block_task(
                    conn, tid, reason="review-required: x",
                    expected_run_id=run_id,
                )
            task = kb.get_task(conn, tid)
            assert task.worker_pid is None
            assert task.worker_pgid is None, f"{finisher} left a stale pgid"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def _fence_via_stale_reclaim(conn, title, leader_pid, pgid):
    """Produce a real fenced-``ready`` row through the production path.

    ``detect_stale_running`` requeues the task and NULLs worker_pid /
    worker_pgid / worker_identity, so the fence payload is the only record
    of which process group is still writing. Anything that later re-probes
    the worker has to read it from there.
    """
    tid = kb.create_task(conn, title=title, assignee="a")
    kb.claim_task(conn, tid)
    kb._set_worker_pid(conn, tid, leader_pid, pgid=pgid)
    old = int(time.time()) - 100_000
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET started_at = ?, last_heartbeat_at = NULL "
            "WHERE id = ?",
            (old, tid),
        )
        conn.execute(
            "UPDATE task_runs SET started_at = ? WHERE id = ?",
            (old, kb.get_task(conn, tid).current_run_id),
        )
    assert kb.detect_stale_running(conn, stale_timeout_seconds=60) == [tid]
    task = kb.get_task(conn, tid)
    assert task.status == "ready"
    assert (task.worker_pid, task.worker_pgid, task.worker_identity) == (
        None, None, None,
    ), "fenced-ready rows carry the worker only in the fence payload"
    assert kb.worker_fence(conn, tid) is not None
    return tid


@pytest.mark.live_system_guard_bypass  # signals only our own orphaned group
@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_operator_reclaim_keeps_a_fence_over_a_live_orphan_group(
    kanban_home, orphaned_group, no_sleep,
):
    """Reclaim is an operator escape, not an override of physics.

    The fenced ``ready`` row has NULL worker columns by construction, so a
    reclaim that probes those columns probes nothing, finds "no worker", and
    hands the task to a second writer beside the live orphan group. The
    fence payload is the source of truth for that probe.
    """
    leader_pid, pgid, child_pid = orphaned_group()
    with kb.connect() as conn:
        tid = _fence_via_stale_reclaim(conn, "reclaim vs orphans", leader_pid, pgid)
        assert kb.claim_task(conn, tid) is None

        kb.reclaim_task(conn, tid, reason="operator wants it back")

        assert kb.worker_fence(conn, tid) is not None, (
            "reclaim released a task next to a live orphan group"
        )
        assert kb.claim_task(conn, tid) is None
        spawned = []
        kb.dispatch_once(conn, spawn_fn=lambda *a, **k: spawned.append(a) or 1)
        assert spawned == []

        # Once the group really is gone, the same call releases it.
        _kill_and_reap(child_pid)
        kb.reclaim_task(conn, tid, reason="operator wants it back")
        assert kb.worker_fence(conn, tid) is None
        assert kb.claim_task(conn, tid) is not None


@pytest.mark.live_system_guard_bypass  # signals only our own orphaned group
@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_reclaim_opens_no_claim_window(kanban_home, orphaned_group, no_sleep):
    """The fence must never be momentarily absent mid-reclaim.

    A dispatcher tick that lands between "clear the fence" and "decide
    whether to re-record it" claims the task — the very duplicate-writer
    state the fence exists to prevent. Probe first, then commit one
    decision: this test claims from inside the probe.
    """
    leader_pid, pgid, _child = orphaned_group()
    mid_reclaim_claims = []
    real_terminate = kb._terminate_reclaimed_worker

    with kb.connect() as conn:
        tid = _fence_via_stale_reclaim(conn, "no window", leader_pid, pgid)

        def _claim_during_probe(*args, **kwargs):
            # Stands in for a concurrent dispatcher tick.
            mid_reclaim_claims.append(kb.claim_task(conn, tid))
            return real_terminate(*args, **kwargs)

        try:
            kb._terminate_reclaimed_worker = _claim_during_probe
            kb.reclaim_task(conn, tid, reason="operator")
        finally:
            kb._terminate_reclaimed_worker = real_terminate

        assert mid_reclaim_claims, "probe never ran — test would prove nothing"
        assert all(c is None for c in mid_reclaim_claims), (
            "a dispatcher could claim mid-reclaim"
        )
        assert kb.worker_fence(conn, tid) is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_operator_reclaim_releases_a_confirmed_dead_fence(
    kanban_home, live_worker, no_sleep,
):
    """The other half: nothing alive behind the fence → reclaim releases."""
    proc = live_worker()
    pid, pgid = proc.pid, os.getpgid(proc.pid)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="dead fence", assignee="a")
        kb.record_worker_fence(
            conn, tid,
            pid=pid, pgid=pgid, run_id=1,
            claim_lock=kb._claimer_id(), reason="unit",
            identity=kb._process_identity(pid),
        )
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL "
                "WHERE id = ?", (tid,),
            )
        assert kb.claim_task(conn, tid) is None  # fenced by a live worker

        _kill_and_reap(pid)
        assert kb.reclaim_task(conn, tid, reason="operator says go") is True
        assert kb.worker_fence(conn, tid) is None
        assert kb.claim_task(conn, tid) is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_identity_null_fence_is_bounded_even_with_a_live_pid(
    kanban_home, live_worker, no_sleep,
):
    """Only a POSITIVE identity match earns an unbounded hold.

    A fence with no identity token (legacy row, or a platform without a
    probe) cannot tell "our worker" from "the process that inherited its
    number". Treating a live PID as proof made the TTL unreachable and let a
    recycled pid strand the task forever.
    """
    proc = live_worker()
    pid, pgid = proc.pid, os.getpgid(proc.pid)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="unverifiable fence", assignee="a")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        kb.record_worker_fence(
            conn, tid,
            pid=pid, pgid=pgid, run_id=1,
            claim_lock=kb._claimer_id(), reason="unit",
            identity=None,
        )
        # Fresh: the evidence is weak but recent — still held.
        assert kb.claim_task(conn, tid) is None

        fence = kb.worker_fence(conn, tid)
        fence["since"] = int(time.time()) - kb.WORKER_FENCE_MAX_SECONDS - 1
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_fence = ? WHERE id = ?",
                (json.dumps(fence), tid),
            )

        assert kb.claim_task(conn, tid) is not None, (
            "identity-NULL fence never reached its bound"
        )
        assert _events(conn, tid, "worker_fence_expired")


def test_completion_clears_a_worker_fence(kanban_home, live_worker):
    """A completed task is terminal: no fence rides along into ``done``."""
    proc = live_worker()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="completed with fence", assignee="a")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, proc.pid)
        kb.record_worker_fence(
            conn, tid,
            pid=proc.pid, pgid=os.getpgid(proc.pid), run_id=1,
            claim_lock=kb._claimer_id(), reason="unit",
            identity=kb._process_identity(proc.pid),
        )
        kb.complete_task(conn, tid, result="shipped")

        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert kb.worker_fence(conn, tid) is None, "fence survived completion"


def test_degrades_safely_without_process_group_primitives(
    kanban_home, live_worker, no_sleep, monkeypatch,
):
    """Windows (and any platform without groups/identity) must fail closed.

    No ``os.killpg`` and no identity probe means: no group signalling, no
    group-based liveness, and the pre-existing PID-only behaviour — never a
    wider blast radius and never a fence we cannot resolve.
    """
    proc = live_worker()
    monkeypatch.delattr(kb.os, "killpg", raising=False)
    monkeypatch.setattr(kb, "_process_identity", lambda pid: None)
    # No identity probe at all — the documented boundary where a bare PID
    # remains the only handle we have ever had on a worker.
    monkeypatch.setattr(kb, "_identity_probe_supported", lambda: False)

    assert kb._process_group_alive(os.getpgid(proc.pid)) is False

    sent = []
    info = kb._terminate_reclaimed_worker(
        proc.pid, kb._claimer_id(),
        signal_fn=lambda *a: sent.append(a),
        pgid=os.getpgid(proc.pid),
    )
    assert info["group_signaled"] is False
    assert {s[0] for s in sent} == {proc.pid}, "signalled beyond the pid"

    # A fence recorded on this platform still resolves: the pid probe alone
    # decides, exactly as the pre-group behaviour did.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="no groups here", assignee="a")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        kb.record_worker_fence(
            conn, tid,
            pid=proc.pid, pgid=os.getpgid(proc.pid), run_id=1,
            claim_lock=kb._claimer_id(), reason="unit",
        )
        assert kb.claim_task(conn, tid) is None  # live pid holds it
        _kill_and_reap(proc.pid)
        assert kb.claim_task(conn, tid) is not None


# ===========================================================================
# Review round 3 — the fence must be part of the claim CAS, and no OS probe
# may run while the board's write lock is held
# ===========================================================================


@pytest.mark.live_system_guard_bypass  # signals only our own orphaned group
@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_claim_loses_to_a_fence_committed_after_its_gate(
    kanban_home, orphaned_group, no_sleep,
):
    """The fence check and the claim must be ONE atomic decision.

    Real ordering, two connections: the dispatcher reads an unfenced task
    and starts claiming; before its CAS lands, the reaper commits
    ``ready + worker_fence`` for a worker whose children are still writing.
    A CAS that only guards ``status`` and ``claim_lock`` still wins — and the
    task is claimed beside the live orphan group.
    """
    leader_pid, pgid, child_pid = orphaned_group()
    with kb.connect() as claimer_conn, kb.connect() as reaper_conn:
        tid = kb.create_task(conn := claimer_conn, title="cas race", assignee="a")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, leader_pid, pgid=pgid)
        run_id = kb.get_task(conn, tid).current_run_id
        lock = kb.get_task(conn, tid).claim_lock

        real_gate = kb._fence_blocks_claim

        def _gate_then_reaper_commits(gate_conn, gate_tid):
            verdict = real_gate(gate_conn, gate_tid)
            # The reaper tick lands here — on its own connection, fully
            # committed — while this claimer is between gate and CAS.
            kb.record_worker_fence(
                reaper_conn, gate_tid,
                pid=leader_pid, pgid=pgid, run_id=run_id,
                claim_lock=lock, reason="requeued_worker_alive",
                identity=None,
            )
            with kb.write_txn(reaper_conn):
                reaper_conn.execute(
                    "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, "
                    "worker_pgid = NULL, worker_identity = NULL, "
                    "current_run_id = NULL WHERE id = ?",
                    (gate_tid,),
                )
            return verdict

        try:
            kb._fence_blocks_claim = _gate_then_reaper_commits
            claimed = kb.claim_task(conn, tid)
        finally:
            kb._fence_blocks_claim = real_gate

        assert claimed is None, "claimed beside a live orphan group"
        assert kb.worker_fence(conn, tid) is not None
        assert kb.get_task(conn, tid).status == "ready"
        assert kb._process_group_alive(pgid), "precondition: orphans still live"

        # And the fence still resolves normally once the group is gone.
        _kill_and_reap(child_pid)
        assert kb.claim_task(conn, tid) is not None


def test_review_claim_is_fenced_too(kanban_home, live_worker, no_sleep):
    """``review -> running`` is a claim as well, and needs the same gate."""
    proc = live_worker()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="review me", assignee="a")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
        kb.record_worker_fence(
            conn, tid,
            pid=proc.pid, pgid=os.getpgid(proc.pid), run_id=1,
            claim_lock=kb._claimer_id(), reason="unit",
            identity=kb._process_identity(proc.pid),
        )

        assert kb.claim_review_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == "review"

        _kill_and_reap(proc.pid)
        assert kb.claim_review_task(conn, tid) is not None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_crash_detection_probes_outside_the_write_lock(
    kanban_home, live_worker, no_sleep,
):
    """``ps``/``/proc`` scans must not run while the board write lock is held.

    ``detect_crashed_workers`` opens a BEGIN IMMEDIATE over the whole board;
    a process-group scan inside it blocks every other writer (workers
    completing, heartbeats, other boards' ticks) for the duration of a
    subprocess call.
    """
    proc = live_worker()
    probed_in_txn = []
    real_group_alive = kb._process_group_alive
    real_leader_alive = kb._worker_leader_alive

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="probe placement", assignee="a")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, proc.pid, pgid=os.getpgid(proc.pid))
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ? WHERE id = ?",
                (int(time.time()) - 3600, tid),
            )

        def _watch(fn):
            def _wrapped(*a, **k):
                probed_in_txn.append(bool(conn.in_transaction))
                return fn(*a, **k)
            return _wrapped

        try:
            kb._process_group_alive = _watch(real_group_alive)
            kb._worker_leader_alive = _watch(real_leader_alive)
            kb.detect_crashed_workers(conn)
        finally:
            kb._process_group_alive = real_group_alive
            kb._worker_leader_alive = real_leader_alive

    assert probed_in_txn, "no probe ran — the test would prove nothing"
    assert not any(probed_in_txn), (
        "OS probe ran while the board write transaction was held"
    )


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_crash_detection_is_not_fooled_by_a_recycled_pid(
    kanban_home, live_worker, no_sleep,
):
    """A live PID that is no longer our worker must not suppress the crash.

    ``_pid_alive`` alone says "still running" and the task sits in
    ``running`` forever with nothing behind it.
    """
    proc = live_worker()  # unrelated process now holding the recorded number
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="recycled pid", assignee="a")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, proc.pid, pgid=os.getpgid(proc.pid))
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ?, worker_identity = ? "
                "WHERE id = ?",
                (int(time.time()) - 3600, "some-older-process", tid),
            )

        assert kb.detect_crashed_workers(conn) == [tid]
        assert kb.get_task(conn, tid).status in ("ready", "blocked")
        assert proc.poll() is None, "signalled a process that is not ours"


def test_below_threshold_failure_does_not_stamp_a_completed_task(kanban_home):
    """N4: the counter branch needs the same status guard as the trip branch.

    A worker that completes while the reaper is accounting a (below-limit)
    failure must not come out of it with a failure counted against a task
    that is already ``done``.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="racy below-threshold", assignee="a")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, result="finished first")

        assert kb._record_task_failure(
            conn, tid,
            error="reaper thought this timed out",
            outcome="timed_out",
            failure_limit=5,          # well below the trip threshold
            release_claim=False,
            end_run=False,
        ) is False

        task = kb.get_task(conn, tid)
        assert task.status == "done"
        assert task.consecutive_failures == 0, "counted a failure on a done task"
        assert task.last_failure_error is None


def test_archive_clears_a_worker_fence(kanban_home, live_worker):
    """N5: archiving is terminal — no fence rides along."""
    proc = live_worker()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="archived with fence", assignee="a")
        kb.record_worker_fence(
            conn, tid,
            pid=proc.pid, pgid=os.getpgid(proc.pid), run_id=1,
            claim_lock=kb._claimer_id(), reason="unit",
            identity=kb._process_identity(proc.pid),
        )
        assert kb.archive_task(conn, tid) is True
        assert kb.worker_fence(conn, tid) is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_unverifiable_pid_widens_to_nothing_beyond_itself(
    kanban_home, live_worker, no_sleep,
):
    """Legacy rows carry a PID and no identity. On a platform that CAN
    identify processes, that number is not evidence of *which* process is
    there — so the blast radius stays exactly that one number. No process
    group (the widest radius we have, and the one that needs proof), and no
    unbounded fence.

    Signalling the pid itself is still right: it is the only handle we ever
    had on that worker, and declining leaves a live writer to be carried by
    a fence that expires. A recorded identity that MISMATCHES is the case
    where we signal nothing at all — see
    ``test_group_signal_requires_verified_leader_identity``.
    """
    proc = live_worker()
    pgid = os.getpgid(proc.pid)

    sent = []
    group_calls = []
    info = kb._terminate_reclaimed_worker(
        proc.pid, kb._claimer_id(),
        signal_fn=lambda *a: sent.append(a),
        killpg_fn=lambda *a: group_calls.append(a),
        pgid=pgid,
        identity=None,
    )
    assert group_calls == [], "blasted a group we cannot prove is ours"
    assert info["group_signaled"] is False
    assert {s[0] for s in sent} == {proc.pid}, "signalled beyond the pid"
    assert info["still_alive"] is True
    assert info["leader_alive"] is True, (
        "a live worker we could not kill must hold, not fall to a bounded fence"
    )
    assert proc.poll() is None


# ===========================================================================
# Review round 4 — clearing a fence is a compare-and-swap, and a pgid is
# only recorded when it identifies the worker's own group
# ===========================================================================


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_releasing_a_fence_cannot_delete_a_newer_one(
    kanban_home, live_worker, no_sleep,
):
    """Clearing must CAS against the exact fence that was evaluated.

    The claim gate decides "this fence is releasable" against a snapshot,
    then writes. If a reaper records a NEW fence in between — a different
    worker, still alive — a blind ``worker_fence = NULL`` by task id deletes
    it, and the task is claimable beside that live worker.
    """
    replacement = live_worker()
    with kb.connect() as conn, kb.connect() as reaper_conn:
        tid = kb.create_task(conn, title="fence swap", assignee="a")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        # F1: describes a worker that is already gone, so it is releasable.
        kb.record_worker_fence(
            conn, tid,
            pid=999_999_000, pgid=999_999_000, run_id=1,
            claim_lock=kb._claimer_id(), reason="f1",
            identity="long-gone",
        )
        f1 = kb.worker_fence(conn, tid)

        real_reason = kb._fence_release_reason

        def _release_then_swap(fence, now):
            verdict = real_reason(fence, now)
            if fence.get("reason") == "f1":
                # A reaper fences the task again — for a worker that IS alive.
                kb.record_worker_fence(
                    reaper_conn, tid,
                    pid=replacement.pid,
                    pgid=os.getpgid(replacement.pid),
                    run_id=2,
                    claim_lock=kb._claimer_id(), reason="f2",
                    identity=kb._process_identity(replacement.pid),
                )
            return verdict

        try:
            kb._fence_release_reason = _release_then_swap
            claimed = kb.claim_task(conn, tid)
        finally:
            kb._fence_release_reason = real_reason

        surviving = kb.worker_fence(conn, tid)
        assert surviving is not None, "a stale release deleted the newer fence"
        assert surviving["reason"] == "f2"
        assert surviving != f1
        assert claimed is None, "claimed beside the newly fenced live worker"
        assert kb.claim_task(conn, tid) is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_pgid_is_recorded_only_for_a_group_leading_worker(kanban_home, no_sleep):
    """A pgid is worth persisting only when it IS the worker's own group.

    Production spawns with ``start_new_session=True``, so the worker leads
    its group and pgid == pid. Anything else (a worker started without a new
    session, an inherited group) means the recorded id names the *gateway's*
    group — signalling it would take down the dispatcher and every sibling
    worker. Store nothing rather than that.
    """
    shared = subprocess.Popen(  # noqa: S603 - fixed argv
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )  # deliberately NO start_new_session: shares our process group
    try:
        assert os.getpgid(shared.pid) != shared.pid, "precondition: not a leader"
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="shared group", assignee="a")
            kb.claim_task(conn, tid)
            kb._set_worker_pid(conn, tid, shared.pid)

            task = kb.get_task(conn, tid)
            assert task.worker_pid == shared.pid
            assert task.worker_pgid is None, (
                "persisted the gateway's own process group id"
            )

            # And nothing downstream can target that shared group.
            group_calls = []
            kb._terminate_reclaimed_worker(
                task.worker_pid, task.claim_lock,
                signal_fn=lambda *a: None,
                killpg_fn=lambda *a: group_calls.append(a),
                pgid=task.worker_pgid,
                identity=task.worker_identity,
            )
            assert group_calls == [], "signalled a shared process group"
    finally:
        try:
            os.kill(shared.pid, signal.SIGKILL)
            shared.wait(timeout=5)
        except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
            pass


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_session_leading_worker_still_records_its_pgid(kanban_home, live_worker):
    """The production shape is unaffected: a session leader keeps its pgid."""
    proc = live_worker()
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="normal spawn", assignee="a")
        kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, proc.pid)
        task = kb.get_task(conn, tid)
        assert task.worker_pgid == os.getpgid(proc.pid) == proc.pid
        assert task.worker_identity is not None


# ===========================================================================
# Review round 5 — a refused reclaim must mutate nothing, and group
# leadership is never assumed
# ===========================================================================


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_reclaim_refused_by_a_newer_fence_mutates_nothing(
    kanban_home, live_worker, no_sleep,
):
    """A reclaim that cannot complete must not half-complete.

    The fence is read before the OS probe and the writes happen after it. If
    a reaper records a different fence in between, bailing out *after* the
    status/claim/worker columns have already been reset commits a partial
    reclaim: the task is released, no ``reclaimed`` event is emitted, the run
    is never finalized, and ``current_run_id`` points at a run row that stays
    open forever. Either the whole reclaim lands or none of it does.
    """
    replacement = live_worker()
    with kb.connect() as conn, kb.connect() as reaper_conn:
        tid = kb.create_task(conn, title="atomic reclaim", assignee="a")
        claimed = kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 999_999_001)
        before = kb.get_task(conn, tid)
        run_id = before.current_run_id
        # F1: a fence for a worker that is long gone (releasable).
        kb.record_worker_fence(
            conn, tid,
            pid=999_999_000, pgid=999_999_000, run_id=run_id,
            claim_lock=claimed.claim_lock, reason="f1", identity="long-gone",
        )

        real_terminate = kb._terminate_reclaimed_worker

        def _swap_fence_mid_probe(*args, **kwargs):
            # A reaper tick lands between reclaim's pre-read and its writes,
            # fencing the task for a worker that IS alive.
            kb.record_worker_fence(
                reaper_conn, tid,
                pid=replacement.pid, pgid=os.getpgid(replacement.pid),
                run_id=run_id, claim_lock=claimed.claim_lock, reason="f2",
                identity=kb._process_identity(replacement.pid),
            )
            return real_terminate(*args, **kwargs)

        try:
            kb._terminate_reclaimed_worker = _swap_fence_mid_probe
            result = kb.reclaim_task(conn, tid, reason="operator")
        finally:
            kb._terminate_reclaimed_worker = real_terminate

        assert result is False
        after = kb.get_task(conn, tid)
        assert after.status == before.status == "running", "partial reclaim"
        assert after.claim_lock == before.claim_lock
        assert after.worker_pid == before.worker_pid
        assert after.current_run_id == run_id, "stranded the run pointer"
        run = kb.get_run(conn, run_id)
        assert run.ended_at is None and run.outcome is None, "run left half-closed"
        assert _events(conn, tid, "reclaimed") == [], (
            "claimed a reclaim happened when nothing did"
        )

        surviving = kb.worker_fence(conn, tid)
        assert surviving is not None and surviving["reason"] == "f2"
        assert kb.claim_task(conn, tid) is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_unknown_process_group_is_never_invented(kanban_home, no_sleep, monkeypatch):
    """If we cannot ask the OS for the group, we do not guess one.

    Synthesizing ``pgid = pid`` on a failed ``getpgid`` passes the
    "worker leads its group" check without any proof that it does — the
    exact assumption that check exists to stop.
    """
    def _boom(_pid):
        raise ProcessLookupError("gone before we looked")

    monkeypatch.setattr(kb.os, "getpgid", _boom)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="unknowable group", assignee="a")
        claimed = kb.claim_task(conn, tid)
        kb._set_worker_pid(conn, tid, 999_999_002)

        task = kb.get_task(conn, tid)
        assert task.worker_pid == 999_999_002
        assert task.worker_pgid is None, "invented a process group"

        group_calls = []
        kb._terminate_reclaimed_worker(
            task.worker_pid, claimed.claim_lock,
            signal_fn=lambda *a: None,
            killpg_fn=lambda *a: group_calls.append(a),
            pgid=task.worker_pgid,
            identity=task.worker_identity,
        )
        assert group_calls == []


# ===========================================================================
# Review round 5 — an identity-less LIVE worker must be killed, not fenced
#
# Failing closed on the *group* (never blast a pgid we cannot prove is ours)
# had been over-applied to the *pid*: a legacy or transiently-unidentified
# row whose worker was still running got no signal at all, only a fence —
# and a fence with no identity token is bounded by WORKER_FENCE_MAX_SECONDS.
# The live worker therefore aged out of its own fence and a replacement
# claim landed beside it: exactly the ``t_80a3542a`` duplicate-writer state.
# ===========================================================================


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_identityless_live_worker_is_terminated_not_left_to_age_out(
    kanban_home, live_worker, no_sleep,
):
    """No identity token is not a reason to leave a live worker running.

    A bare PID we recorded ourselves is weak evidence of *which* process is
    there, but the pre-existing behaviour — signal the pid — stays the right
    call while the number is still alive: the alternative is a fence that
    expires under a worker that never stopped writing. The blast radius stays
    the single pid; the process GROUP still requires a proven identity.
    """
    proc = live_worker()
    pgid = os.getpgid(proc.pid)

    group_calls = []
    info = kb._terminate_reclaimed_worker(
        proc.pid, kb._claimer_id(),
        killpg_fn=lambda *a: group_calls.append(a),
        pgid=pgid,
        identity=None,
    )

    assert info["may_signal"] is True, "declined to signal a live worker"
    assert info["termination_attempted"] is True
    assert group_calls == [], "blasted a group we cannot prove is ours"
    assert info["group_signaled"] is False
    assert info["terminated"] is True
    assert info["still_alive"] is False
    proc.wait(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_identityless_survivor_holds_the_claim_instead_of_fencing(
    kanban_home, live_worker, no_sleep,
):
    """A live identity-less leader that shrugs off the kill is a HOLD.

    ``_worker_survived_termination`` is what keeps the claim (and so keeps
    the dispatcher from spawning a second writer). Reporting an unsignalled,
    unverifiable-but-live leader as "not alive" routed this state onto the
    bounded fence path, which resolves itself after an hour regardless of
    the process. Holding is correct and self-correcting: the next tick
    retries the kill.
    """
    proc = live_worker()
    sent = []
    info = kb._terminate_reclaimed_worker(
        proc.pid, kb._claimer_id(),
        signal_fn=lambda *a: sent.append(a),
        pgid=os.getpgid(proc.pid),
        identity=None,
    )

    assert {s[0] for s in sent} == {proc.pid}, "signalled beyond the pid"
    assert info["sigkill"] is True, "never escalated past SIGTERM"
    assert info["leader_alive"] is True
    assert info["still_alive"] is True
    assert info["terminated"] is False
    assert kb._worker_survived_termination(info) is True, (
        "a live worker we could not kill must hold its claim, not age out"
    )
    assert proc.poll() is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")
def test_operator_reclaim_pins_a_live_identityless_leader(
    kanban_home, live_worker, no_sleep,
):
    """The operator path fences rather than holds — so the fence must bind.

    ``reclaim_task`` honours the operator's release even for a worker that
    survived termination, and records a fence to stop the duplicate writer.
    With no identity on the row that fence was weakly evidenced and expired
    after ``WORKER_FENCE_MAX_SECONDS`` while the worker was still running.
    Pinning the identity observed at termination time makes the hold provable
    (so it cannot age out under a live worker) and still self-releasing (a
    recycled pid reads as ``worker_identity_changed``).
    """
    proc = live_worker()
    pgid = os.getpgid(proc.pid)
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="legacy live worker", assignee="a")
        kb.claim_task(conn, tid)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_pid = ?, worker_pgid = ?, "
                "worker_identity = NULL WHERE id = ?",
                (proc.pid, pgid, tid),
            )

        # Signals swallowed: the worker outlives the operator's reclaim.
        assert kb.reclaim_task(
            conn, tid, reason="operator", signal_fn=lambda *a: None,
        ) is True

        fence = kb.worker_fence(conn, tid)
        assert fence is not None, "released a live worker with no fence"
        assert fence["identity"] == kb._process_identity(proc.pid), (
            "fenced a live leader on nothing but a recyclable number"
        )

        # Age it far past the weak-evidence bound: a PROVEN live leader is
        # not on a clock.
        fence["since"] = int(time.time()) - kb.WORKER_FENCE_MAX_SECONDS * 10
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_fence = ? WHERE id = ?",
                (json.dumps(fence), tid),
            )
        assert kb._fence_release_reason(fence, int(time.time())) is None
        assert kb.claim_task(conn, tid) is None, (
            "a live worker aged out of its own fence: duplicate writers"
        )

        # And it still resolves on its own once the worker is really gone.
        _kill_and_reap(proc.pid)
        assert kb.claim_task(conn, tid) is not None

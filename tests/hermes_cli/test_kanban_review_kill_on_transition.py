"""Kill-at-terminal-review-transition + bound-pid reap targeting.

Incident 2026-11-16 (PR#51 dual-reviewer collision, card t_58b75f7e): a reviewer
worker's run was ended by ANOTHER execution's ``changes_requested`` transition;
the reaper SIGTERM'd the spawn parent, reported ``terminated: true``, and the
BOUND worker session survived 3.5 more minutes and squash-merged the PR 41 s
after its review authority ended.

Locked invariants:
* D1 — reap/kill target EVERY process identity recorded for the attempt (run-row
  spawn pair + ``spawned``/``worker_bound`` events), with SIGKILL escalation, and
  ``terminated: true`` only after a /proc re-check finds every target gone.
* D2 — a terminal review transition recorded by a FOREIGN session over a LIVE
  review worker terminates that worker immediately (within the transition call),
  not at reap-grace + next tick. The worker's OWN verdict keeps the untouched
  graceful path.
* D3 — review transition payloads carry the recording session's identity
  (``actor``), and a foreign close over a live review execution records a
  ``review_claim_collision`` event: the double-execution gets a mechanical trace.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_RUN_ID", "HERMES_KANBAN_BOARD", "HERMES_SESSION_ID",
                "HERMES_DELEGATED_CHILD_CONTEXT", "HERMES_KANBAN_TASK"):
        monkeypatch.delenv(var, raising=False)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _sleeper():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    return proc


def _events(conn, task_id, kind):
    return conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind=?", (task_id, kind),
    ).fetchall()


def _review_run(conn, *, assignee="reviewer"):
    """Card taken through implementer -> review -> dispatcher review-claim.
    Returns (task_id, review_run_id). The claim uses a host claimer so the
    attempt is host-local (the reaper/kill only ever touch host-local pids)."""
    tid = kb.create_task(conn, title="review fan-in", assignee=assignee)
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    run_id = kb._current_run_id(conn, tid)
    assert kb.request_review(conn, tid, summary="implementer handoff",
                             reviewer=assignee, expected_run_id=run_id) is True
    review = kb.claim_review_task(conn, tid, claimer=kb._claimer_id())
    assert review is not None
    review_run = kb._current_run_id(conn, tid)
    assert review_run is not None
    return tid, int(review_run)


def _bind_worker_session(conn, tid, run_id, bound_proc):
    """Record the ``worker_bound`` identity of the attempt: the interpreter the
    owner session adopted beside the dispatcher's spawn handle."""
    fingerprint = kbd._process_fingerprint(bound_proc.pid)
    assert fingerprint is not None
    with kb.write_txn(conn):
        kb._append_event(
            conn, tid, "worker_bound",
            {"pid": bound_proc.pid, "claim_lock": None, "started_at": fingerprint},
            run_id=run_id,
        )


# ---------------------------------------------------------------------------
# D1 — reap targets the bound session, not just the spawn parent
# ---------------------------------------------------------------------------

def test_reap_kills_bound_worker_when_spawn_parent_is_already_gone(conn, monkeypatch):
    """PR#51's survivor shape: the spawn handle is dead (the old reaper was
    'successful'), the bound session pid lives — the sweep must find it in the
    worker_bound event, kill it, and only then resolve the row."""
    dead_parent = _sleeper()
    survivor = _sleeper()
    try:
        tid, run_id = _review_run(conn)
        kbd._set_worker_pid(conn, tid, dead_parent.pid, run_id=run_id)
        _bind_worker_session(conn, tid, run_id, survivor)
        # The completion is the run's OWN session recording its verdict (graceful
        # path): the transition must NOT pre-empt the kill — the reaper owns it here.
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
        assert kb.complete_task(conn, tid, result="approved", expected_run_id=run_id) is True
        monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
        dead_parent.kill()
        dead_parent.wait(timeout=10)
        conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?",
                     (kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS, run_id))

        assert kbd.reap_terminal_workers(conn) == [tid]

        # terminated:true now means the BOUND pid is actually gone, not that one
        # os.kill call to the spawn parent did not raise.
        assert survivor.wait(timeout=10) is not None
        row = conn.execute("SELECT worker_pid, worker_reaped FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert row["worker_pid"] is None and row["worker_reaped"] is not None
        (event,) = _events(conn, tid, "terminal_worker_reaped")
        import json
        payload = json.loads(event[0])
        assert survivor.pid in payload["targets"]
        assert payload["terminated"] is True
    finally:
        for p in (dead_parent, survivor):
            if p.poll() is None:
                p.kill()
                p.wait()


def test_reap_resolves_evidence_less_rows_and_never_rescans_them(conn):
    """A closed run with no live process anywhere (including a run whose spawn
    died before a pid was ever recorded) is marked resolved once; the sweep does
    not rescan board history every tick."""
    tid, run_id = _review_run(conn)
    assert kb.complete_task(conn, tid, result="approved", expected_run_id=run_id) is True
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?",
                 (kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS, run_id))

    assert kbd.reap_terminal_workers(conn) == []  # nothing alive: resolved, not "reaped"
    resolved = conn.execute("SELECT worker_reaped FROM task_runs WHERE id=?", (run_id,)).fetchone()["worker_reaped"]
    assert resolved is not None
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?",
                 (kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS, run_id))
    assert kbd.reap_terminal_workers(conn) == []  # second tick: row no longer even selected


# ---------------------------------------------------------------------------
# D2 + D3 — transition-triggered kill + attribution + collision witness
# ---------------------------------------------------------------------------

def test_foreign_changes_requested_terminates_live_review_worker_immediately(conn):
    """The incident repro, at the kernel boundary: a review transition recorded
    by a session OTHER than the live review worker must leave that worker dead
    when the transition call returns — before any next action, not one
    tick + grace later. The transition records WHO ended the execution and the
    double-execution gets a ``review_claim_collision`` event."""
    import json
    survivor = _sleeper()
    try:
        tid, run_id = _review_run(conn)
        _bind_worker_session(conn, tid, run_id, survivor)

        ok, _ = kb.request_changes(conn, tid, reason="found issues", expected_run_id=run_id)
        assert ok is True

        # Dead within the transition — the worker's next command (the merge) cannot run.
        assert survivor.wait(timeout=10) is not None

        (transition,) = _events(conn, tid, "changes_requested")
        actor = json.loads(transition[0])["actor"]
        assert actor["pid"] == os.getpid()  # recording session identity (D3)
        collisions = _events(conn, tid, "review_claim_collision")
        assert len(collisions) == 1
        collision = json.loads(collisions[0][0])
        assert collision["ending_run_id"] == run_id
        assert [r["run_id"] for r in collision["live_runs"]] == [run_id]
        assert survivor.pid in collision["live_runs"][0]["pids"] or collision["live_runs"][0]["pids"] == []
    finally:
        if survivor.poll() is None:
            survivor.kill()
            survivor.wait()


def test_worker_self_transition_keeps_the_graceful_path(conn, monkeypatch):
    """The worker recording its OWN verdict is the healthy path: no immediate
    kill (grace reap as before) and no collision event."""
    own = _sleeper()
    try:
        tid, run_id = _review_run(conn)
        _bind_worker_session(conn, tid, run_id, own)
        # The acting process IS the attempt: the run's identity is in its env.
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

        ok, _ = kb.request_changes(conn, tid, reason="verdict: rework", expected_run_id=run_id)
        assert ok is True

        assert own.poll() is None  # untouched; the grace reap owns the final kill
        assert _events(conn, tid, "review_claim_collision") == []
    finally:
        if own.poll() is None:
            own.kill()
            own.wait()
        else:
            own.wait()


def test_review_claim_over_open_sibling_run_leaves_mechanical_trace(conn):
    """Case (b) (Pepper, t_ea9f278a): a review claim landing while another run
    on the same card is still open records ONE collision event — witness only,
    the claim itself succeeds (arbitration belongs to the process layer)."""
    import json
    tid, run_id = _review_run(conn)
    ok, _ = kb.request_changes(conn, tid, reason="verdict from elsewhere", expected_run_id=run_id)
    assert ok is True
    # The lost-run shape (PR#51's board state): the card is parked in `review`
    # while the previous execution's row never closed — the identity was steered
    # elsewhere and the board kept the run open.
    conn.execute("UPDATE task_runs SET ended_at = NULL, outcome = NULL WHERE id = ?", (run_id,))
    conn.execute("UPDATE tasks SET status = 'review', current_run_id = NULL WHERE id = ?", (tid,))
    before = _events(conn, tid, "review_claim_collision")

    again = kb.claim_review_task(conn, tid, claimer=kb._claimer_id())
    assert again is not None  # no policy change: the claim succeeds
    after = _events(conn, tid, "review_claim_collision")
    assert len(after) == len(before) + 1
    collision = json.loads(after[-1][0])
    assert collision["kind"] == "claim"
    assert [r["run_id"] for r in collision["live_runs"]] == [run_id]


def test_clean_review_handoff_records_no_collision_at_claim(conn):
    """The healthy handoff (verdict recorded, run closed, card in review) must
    stay silent: the witness only fires over an OPEN sibling run."""
    tid, run_id = _review_run(conn)
    ok, _ = kb.request_changes(conn, tid, reason="verdict: rework", expected_run_id=run_id)
    assert ok is True
    # Card returns to the implementer on changes_requested; drive it back to review
    # with a second clean handoff so the review-claim path is proven silent.
    impl = kb.claim_task(conn, tid, claimer=kb._claimer_id())
    assert impl is not None
    run2 = kb._current_run_id(conn, tid)
    assert kb.request_review(conn, tid, summary="second handoff",
                             reviewer="reviewer", expected_run_id=run2) is True
    before = _events(conn, tid, "review_claim_collision")
    assert kb.claim_review_task(conn, tid, claimer=kb._claimer_id()) is not None
    assert _events(conn, tid, "review_claim_collision") == before  # no new event


def test_transition_kill_failure_is_recorded_not_silent(conn, monkeypatch):
    """If the immediate kill cannot deliver, the gap is booked (``worker_kill_
    deferred``) instead of pretending the worker is down — the reap tick retries.
    The seam is patched on the dispatch module AS RESOLVED IN sys.modules at call
    time: transition code imports it lazily, so patching the collection-time
    reference alone can miss when an earlier fixture swapped the module graph."""
    import json
    survivor = _sleeper()
    live_kbd = sys.modules.get("hermes_cli.kanban_db_dispatch", kbd)
    real_terminate = live_kbd._terminate_reclaimed_worker

    def failing_terminate(pid, claim_lock, **kw):
        return {"prev_pid": pid, "host_local": True, "termination_attempted": True,
                "terminated": False, "sigkill": False}

    try:
        tid, run_id = _review_run(conn)
        _bind_worker_session(conn, tid, run_id, survivor)
        monkeypatch.setattr(live_kbd, "_terminate_reclaimed_worker", failing_terminate)

        ok, _ = kb.request_changes(conn, tid, reason="found issues", expected_run_id=run_id)
        assert ok is True
        assert survivor.poll() is None  # we failed to kill it: the deferral must be visible

        (deferred,) = _events(conn, tid, "worker_kill_deferred")
        payload = json.loads(deferred[0])
        assert payload["run_id"] == run_id
        assert survivor.pid in payload["targets"]
        monkeypatch.setattr(live_kbd, "_terminate_reclaimed_worker", real_terminate)
        # The next reap tick closes the gap (grace already passed by the test clock? No —
        # ended just now: force past the grace window, exactly as a stuck row ages).
        conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?",
                     (live_kbd.TERMINAL_WORKER_REAP_GRACE_SECONDS, run_id))
        assert live_kbd.reap_terminal_workers(conn) == [tid]
        assert survivor.wait(timeout=10) is not None
    finally:
        if survivor.poll() is None:
            survivor.kill()
            survivor.wait()

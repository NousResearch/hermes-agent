"""Clean exits with no terminal kanban call: parked with state, never a crash.

Board evidence for the change: of the last 40 crashed runs on board ``vx``, 21
carried *"worker exited cleanly (rc=0) without calling kanban_complete or
kanban_block"*. One card (``t_9036debb``) accumulated nine of them; every one of
those runs had heartbeated for minutes first, and two had already left a
checkpoint naming a commit and an opened PR. Re-running such a card from scratch
throws that state away — which is how a single card reaches nine crashed runs —
and asking the *next* worker to work out whether the previous one finished is a
human-shaped judgement sitting inside a retry loop.

So a clean exit that left observable state is PARKED (``blocked``,
``block_kind='transient'``) carrying that state, and is never ``done``; a clean
exit that recorded nothing observable keeps the bounded retry ladder, unchanged.
"""

from __future__ import annotations

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
    # Pre-date-the-grace-window semantics, as in test_kanban_core_functionality.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claim(conn, tid, *, heartbeats=0, notes=False):
    """Claim ``tid`` and open a real run, optionally with heartbeats on it."""
    import hermes_cli.kanban_db as _kb
    from hermes_cli import kanban_db_dispatch as _kbd

    host_prefix = _kb._claimer_id().split(":", 1)[0]
    claimed = _kb.claim_task(conn, tid, claimer=f"{host_prefix}:mock")
    assert claimed is not None, "task was not claimable"
    run_id = _kb._current_run_id(conn, tid)
    for i in range(heartbeats):
        assert _kbd.heartbeat_worker(
            conn, tid, note=(f"checkpoint step {i}" if notes else None),
            expected_run_id=run_id,
        )
    return run_id


def _reap_clean_exit(conn, tid, fake_pid):
    """Record ``rc=0`` for the task's dead worker and run one reaper pass.

    Returns ``(crashed_ids, parked_ids)``. Resolves both modules fresh and reads
    the park side channel off the SAME dispatch function object it called: a
    full-suite run can reload the module, and comparing against a module-level
    import would read ``_last_unreported_parked`` off a stale function object and
    report "nothing parked" even though the park happened. The same reasoning
    applies to the exit registry / liveness patch below (recording the exit into
    one module object while reaping through another makes ``_classify_worker_exit``
    return ``unknown``, silently turning a clean exit into a plain crash).
    """
    import hermes_cli.kanban_db as _kb
    from hermes_cli import kanban_db_dispatch as _kbd

    _kbd._set_worker_pid(conn, tid, fake_pid)
    _kbd._record_worker_exit(fake_pid, 0)  # os.W_EXITCODE(0, 0) == 0 on POSIX
    original_alive = _kb._pid_alive
    _kb._pid_alive = lambda p: False
    try:
        crashed = _kbd.detect_crashed_workers(conn)
    finally:
        _kb._pid_alive = original_alive
    return crashed, list(getattr(_kbd.detect_crashed_workers, "_last_unreported_parked", []))


def _events(conn, tid, kind):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


def _latest_run(conn, tid):
    runs = kb.list_runs(conn, tid)
    return runs[-1]


def test_clean_exit_with_checkpoint_parks_card_instead_of_re_running(kanban_home):
    """Acceptance 1 + 2: the outcome asserted is park-with-state, by code path.

    A run that exits ``rc=0`` with no terminal call, having left a checkpoint
    comment and heartbeats behind, must end in a visible anomaly carrying that
    state — not a bare crash, not a re-run from scratch, and never ``done``.
    """
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="te7 settings modal", assignee="worker")
        run_id = _claim(conn, tid, heartbeats=3, notes=True)
        checkpoint_id = kb.add_comment(
            conn, tid, author="worker",
            body="**checkpoint (run %s)** — PR opened: "
                 "https://github.com/The-ValueExchange/website-frontend/pull/839 "
                 "(base development)" % run_id,
        )

        crashed, parked = _reap_clean_exit(conn, tid, 991100)

        task = kb.get_task(conn, tid)
        # (1) visible anomaly, never done, never re-run from scratch
        assert task.status == "blocked", f"expected parked blocked, got {task.status}"
        assert task.status != "done"
        assert task.block_kind == "transient"
        assert tid not in crashed, "a parked unreported exit is not a crash"
        assert parked == [tid]

        # (2) the run is ``unreported`` — not a plain crash, not completed
        run = _latest_run(conn, tid)
        assert run.outcome == kbd.UNREPORTED_RUN_OUTCOME
        assert run.outcome not in ("crashed", "completed")

        # (3) the park carries the run's own last-known state, in the reason,
        #     the typed event, the routing event and a comment for the next worker
        assert "pull/839" in (task.last_failure_error or "")
        assert "3 heartbeat(s)" in (task.last_failure_error or "")
        unreported = _events(conn, tid, "unreported_exit")
        assert len(unreported) == 1
        payload = unreported[0].payload or {}
        assert payload.get("parked") is True and payload.get("salvaged") is True
        state = payload.get("state") or {}
        assert state.get("run_id") == run_id
        assert [c["comment_id"] for c in state["checkpoints"]] == [checkpoint_id]
        assert state["heartbeats"] == 3
        blocked = _events(conn, tid, "blocked")
        assert blocked, "the park must emit the routing block event"
        dispatcher_comments = [
            c for c in kb.list_comments(conn, tid) if c.author == "dispatcher"
        ]
        assert dispatcher_comments, "the state must be readable by the next worker"
        assert "PARKED" in dispatcher_comments[-1].body
        assert "pull/839" in dispatcher_comments[-1].body
        # A park survives the promote pass: it is not quietly un-parked.
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "blocked"
    finally:
        conn.close()


def test_liveness_alone_is_enough_state_to_park(kanban_home):
    """A heartbeat is run-bound proof the worker was alive: do not re-run blind."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="liveness only", assignee="worker")
        _claim(conn, tid, heartbeats=2)
        _reap_clean_exit(conn, tid, 991101)
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_kind == "transient"
        payload = (_events(conn, tid, "unreported_exit")[0].payload or {})
        assert (payload.get("state") or {}).get("heartbeats") == 2
        assert (payload.get("state") or {}).get("checkpoints") == []
    finally:
        conn.close()


def test_clean_exit_with_nothing_observable_keeps_the_bounded_retry(kanban_home):
    """The retry ceiling is NOT loosened: an empty exit still retries, bounded.

    Runs that recorded nothing at all (no checkpoint, no heartbeat) are the one
    case with nothing to salvage. They keep the violation-only streak limit, and
    their outcome is ``unreported`` rather than a plain crash.
    """
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="empty exit", assignee="worker")
        for i, pid in enumerate((991110, 991111)):
            _claim(conn, tid)
            _reap_clean_exit(conn, tid, pid)
            task = kb.get_task(conn, tid)
            assert task.status == "ready", (
                f"empty exit {i + 1} must still retry, got {task.status}"
            )
            assert task.consecutive_failures == 0, (
                "an unreported exit must not tick the unified failure counter"
            )
            assert _latest_run(conn, tid).outcome == kbd.UNREPORTED_RUN_OUTCOME
        # Third consecutive empty exit: the unchanged streak bound parks it.
        _claim(conn, tid)
        _reap_clean_exit(conn, tid, 991112)
        assert kb.get_task(conn, tid).status == "blocked"
        gave_up = _events(conn, tid, "gave_up")
        assert len(gave_up) == 1
        assert (gave_up[0].payload or {}).get("protocol_violations") == \
            kbd._PROTOCOL_VIOLATION_FAILURE_LIMIT
    finally:
        conn.close()


def test_evidence_of_other_lanes_is_not_this_task_s_state(kanban_home):
    """Evidence must be bound to THIS task and THIS run.

    Another task's checkpoint in the same wall-clock window, and this task's own
    comment from before the run started, are not evidence that this run did work
    — counting them would park a card on another lane's activity.
    """
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="mislabelled", assignee="worker")
        other = kb.create_task(conn, title="someone else", assignee="worker")
        kb.add_comment(conn, other, author="worker", body="checkpoint: other lane")
        kb._insert_comment(conn, tid, "worker", "checkpoint from a previous run", 1)
        run_id = _claim(conn, tid)
        assert run_id is not None

        state = kbd._run_observable_state(
            conn, tid, run_id=run_id,
            started_at=kb.list_runs(conn, tid)[-1].started_at,
        )
        assert state["checkpoints"] == [], (
            "comments on another task, or from before the run, are not this run's state"
        )
        assert kbd._state_has_observable_work(state) is False

        _reap_clean_exit(conn, tid, 991120)
        task = kb.get_task(conn, tid)
        assert task.status == "ready", "with no state of its own, retry is correct"
        assert not _events(conn, tid, "unreported_exit")[0].payload.get("parked")
    finally:
        conn.close()

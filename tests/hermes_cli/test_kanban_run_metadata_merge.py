"""Invariant: a run's ``task_runs.metadata`` accumulates; it is never replaced wholesale.

``task_runs.metadata`` is the only home for a run's provenance and check results —
``worker_session_id``, ``verdict``, ``changed_files``, ``ac_status`` — and different
writers learn those keys at different times. ``hermes kanban edit --metadata`` replaced
the column with just its own payload, so backfilling a completed task's result silently
destroyed the verdict and session provenance already recorded on that run.

``_end_run`` replaced it too. The worker stamps ``worker_session_id`` on its first
heartbeat, because that is the earliest moment where both the session id and the run row
exist — the dispatcher cannot do it, since the session does not exist at spawn time.
Every dispatcher-side ending then writes its own payload: ``timed_out``, ``stale``,
``reclaimed``, ``crashed``, ``gave_up``. While that was a replace, each of those erased
the stamp — on exactly the runs that never got a polite close and most need the
post-mortem. Three callers pass no metadata at all, so they nulled the column outright.
"""

from __future__ import annotations

import os
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
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _completed_run(conn, *, metadata: dict) -> tuple[str, int]:
    """A task completed by its worker, carrying the run metadata that completion recorded."""
    tid = kb.create_task(conn, title="work", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    kbd._set_worker_pid(conn, tid, os.getpid())
    run_id = kb._current_run_id(conn, tid)
    assert kb.complete_task(conn, tid, result="done", metadata=metadata,
                            expected_run_id=run_id) is True
    return tid, run_id


def _metadata(conn, run_id: int) -> dict:
    row = conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    return kb._json_dict(row["metadata"])


def test_edit_preserves_run_metadata_it_does_not_carry(conn):
    """``hermes kanban edit --metadata`` must not destroy the run's recorded verdict."""
    tid, run_id = _completed_run(
        conn, metadata={"worker_session_id": "sess-abc", "verdict": "pass", "ac_status": "met"})

    assert kb.edit_completed_task_result(
        conn, tid, result="corrected result", metadata={"published_pr": "http://x/1"}) is True

    meta = _metadata(conn, run_id)
    assert meta["published_pr"] == "http://x/1"
    assert meta["worker_session_id"] == "sess-abc"
    assert meta["verdict"] == "pass"
    assert meta["ac_status"] == "met"


def test_edit_overwrites_the_keys_it_does_carry(conn):
    """Accumulating is not the same as freezing: an edit still corrects its own keys."""
    tid, run_id = _completed_run(conn, metadata={"verdict": "pass", "worker_session_id": "sess-abc"})

    assert kb.edit_completed_task_result(
        conn, tid, result="corrected", metadata={"verdict": "fail"}) is True

    meta = _metadata(conn, run_id)
    assert meta["verdict"] == "fail"
    assert meta["worker_session_id"] == "sess-abc"


def test_edit_without_metadata_leaves_the_column_alone(conn):
    """A result-only edit is not a reason to touch provenance."""
    tid, run_id = _completed_run(conn, metadata={"verdict": "pass"})

    assert kb.edit_completed_task_result(conn, tid, result="corrected") is True

    assert _metadata(conn, run_id) == {"verdict": "pass"}
    row = conn.execute("SELECT result FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["result"] == "corrected"


def test_merge_direction_is_the_callers_choice(conn):
    """``incoming_wins`` is what lets a periodic writer avoid churning a row it re-visits."""
    tid, run_id = _completed_run(conn, metadata={"shared": "original"})

    with kb.write_txn(conn):
        kb._merge_run_metadata(conn, run_id, {"shared": "loser"}, incoming_wins=False)
    assert _metadata(conn, run_id)["shared"] == "original"

    with kb.write_txn(conn):
        kb._merge_run_metadata(conn, run_id, {"shared": "winner"}, incoming_wins=True)
    assert _metadata(conn, run_id)["shared"] == "winner"


# ---------------------------------------------------------------------------
# _end_run: the close must not erase what the run already recorded
# ---------------------------------------------------------------------------

def _running_run(conn) -> tuple[str, int]:
    """A claimed, running task standing in for a dispatcher-spawned worker."""
    tid = kb.create_task(conn, title="work", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    kbd._set_worker_pid(conn, tid, os.getpid())
    return tid, kb._current_run_id(conn, tid)


@pytest.mark.parametrize("outcome", ["timed_out", "stale", "reclaimed", "crashed"])
def test_dispatcher_close_retains_worker_session_id(conn, outcome):
    """The regression the merge exists to prevent: a run that heartbeats and is then
    closed by the dispatcher keeps the session link the worker stamped."""
    tid, run_id = _running_run(conn)
    assert kbd.heartbeat_worker(
        conn, tid, expected_run_id=run_id, metadata={"worker_session_id": "sess-abc"}) is True

    with kb.write_txn(conn):
        kb._end_run(
            conn, tid, outcome=outcome, status=outcome, error="worker died",
            metadata={"pid": 4242, "claimer": "host:1"},
        )

    meta = _metadata(conn, run_id)
    assert meta["worker_session_id"] == "sess-abc"
    # The closer's own payload still lands.
    assert meta["pid"] == 4242 and meta["claimer"] == "host:1"


def test_heartbeat_stamp_is_first_write_wins(conn):
    """Repeated beats from a long-running worker must not churn the row."""
    tid, run_id = _running_run(conn)
    kbd.heartbeat_worker(conn, tid, expected_run_id=run_id, metadata={"worker_session_id": "first"})
    kbd.heartbeat_worker(conn, tid, expected_run_id=run_id, metadata={"worker_session_id": "second"})
    kbd.heartbeat_worker(conn, tid, expected_run_id=run_id, metadata={"later_key": "kept"})

    meta = _metadata(conn, run_id)
    assert meta["worker_session_id"] == "first"
    assert meta["later_key"] == "kept"


def test_close_without_metadata_does_not_erase(conn):
    """``changes_requested`` / ``archived`` / ancestor-reopen pass no metadata; under
    replace semantics they nulled the column."""
    tid, run_id = _running_run(conn)
    kbd.heartbeat_worker(conn, tid, expected_run_id=run_id,
                         metadata={"worker_session_id": "sess-abc"})

    with kb.write_txn(conn):
        kb._end_run(conn, tid, outcome="reclaimed", status="todo", summary="ancestor reopened")

    assert _metadata(conn, run_id)["worker_session_id"] == "sess-abc"


def test_closer_wins_on_key_collision(conn):
    """Merge direction is asymmetric on purpose: the last write describes the ending."""
    tid, run_id = _running_run(conn)
    kbd.heartbeat_worker(conn, tid, expected_run_id=run_id, metadata={"shared": "heartbeat"})

    with kb.write_txn(conn):
        kb._end_run(conn, tid, outcome="crashed", status="crashed", metadata={"shared": "closer"})

    assert _metadata(conn, run_id)["shared"] == "closer"


def test_end_run_does_not_touch_metadata_when_cas_fails(conn):
    """``WHERE ended_at IS NULL`` is the idempotence CAS; a second close writes nothing."""
    tid, run_id = _running_run(conn)
    with kb.write_txn(conn):
        kb._end_run(conn, tid, outcome="crashed", status="crashed", metadata={"first": "close"})
    # current_run_id is cleared, so re-close the same row directly.
    conn.execute("UPDATE tasks SET current_run_id = ? WHERE id = ?", (run_id, tid))
    with kb.write_txn(conn):
        kb._end_run(conn, tid, outcome="timed_out", status="timed_out",
                    metadata={"second": "close"})

    assert _metadata(conn, run_id) == {"first": "close"}


def test_heartbeat_without_metadata_leaves_row_alone(conn):
    """A locally-driven worker that bypassed the dispatcher stamps nothing."""
    tid, run_id = _running_run(conn)
    assert kbd.heartbeat_worker(conn, tid, expected_run_id=run_id) is True
    assert _metadata(conn, run_id) == {}

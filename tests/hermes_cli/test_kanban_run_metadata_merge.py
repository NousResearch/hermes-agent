"""Invariant: a run's ``task_runs.metadata`` accumulates; it is never replaced wholesale.

``task_runs.metadata`` is the only home for a run's provenance and check results —
``worker_session_id``, ``verdict``, ``changed_files``, ``ac_status`` — and different
writers learn those keys at different times. ``hermes kanban edit --metadata`` replaced
the column with just its own payload, so backfilling a completed task's result silently
destroyed the verdict and session provenance already recorded on that run.
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

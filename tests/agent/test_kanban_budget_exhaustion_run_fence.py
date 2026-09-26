"""A kanban worker's budget-exhaustion failure is fenced to the worker's own run.

A worker that already completed its card, or handed it off to review, can still exhaust its
iteration budget while writing the final summary. ``_record_kanban_budget_exhausted`` then fired
against whatever run was *current*: on a ``done`` card it appended a spurious ``timed_out`` event
("will retry", ``retry_status: ready``); on a card a reviewer had already claimed it closed the
reviewer's live run as ``timed_out``, released the reviewer's claim, moved the card back to
``review`` and charged the breaker.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agent.turn_finalizer import _record_kanban_budget_exhausted
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

_LOG = logging.getLogger(__name__)


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kbc.connect()
    yield conn, monkeypatch
    conn.close()


def _as_worker(monkeypatch, task_id, run_id):
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    _record_kanban_budget_exhausted(task_id, 90, 90, _LOG)


def _snapshot(conn, task_id):
    task = conn.execute(
        "SELECT status, current_run_id, claim_lock, consecutive_failures FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    runs = conn.execute(
        "SELECT id, status, outcome, ended_at FROM task_runs WHERE task_id = ? ORDER BY id", (task_id,),
    ).fetchall()
    events = conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (task_id,),
    ).fetchall()
    return tuple(task), [tuple(r) for r in runs], [e[0] for e in events]


def test_worker_outliving_its_run_records_nothing(board):
    conn, mp = board
    done = kb.create_task(conn, title="done", assignee="default")
    done_run = kb.claim_task(conn, done).current_run_id
    assert kb.complete_task(conn, done, summary="ok", expected_run_id=done_run)

    handed = kb.create_task(conn, title="handed off", assignee="default")
    impl_run = kb.claim_task(conn, handed).current_run_id
    assert kb.request_review(conn, handed, summary="impl", reviewer="reviewer", expected_run_id=impl_run)
    reviewer_run = kb.claim_review_task(conn, handed).current_run_id
    assert reviewer_run != impl_run

    before = {t: _snapshot(conn, t) for t in (done, handed)}
    _as_worker(mp, done, done_run)
    _as_worker(mp, handed, impl_run)

    assert {t: _snapshot(conn, t) for t in (done, handed)} == before
    status, current, claim, _ = before[handed][0]
    assert (status, current) == ("running", reviewer_run) and claim


def test_worker_exhausting_its_live_run_still_records_the_failure(board):
    conn, mp = board
    tid = kb.create_task(conn, title="live", assignee="default")
    run_id = kb.claim_task(conn, tid).current_run_id

    _as_worker(mp, tid, run_id)

    (status, current, claim, failures), runs, events = _snapshot(conn, tid)
    assert (status, current, claim, failures) == ("ready", None, None, 1)
    assert runs[-1][:3] == (run_id, "timed_out", "timed_out") and runs[-1][3] is not None
    assert events[-1] == "timed_out"

"""Run-record bookkeeping: ``task_runs.status`` vs ``task_runs.outcome``.

``_synthesize_ended_run`` writes the row for a terminal transition on a task
that was never claimed — ``hermes kanban complete <ready-task>``, or the
dashboard marking a ready task done. It used to pass ``outcome`` for both
columns, which is only correct where the two names happen to coincide. They
diverge for the two transitions that matter most:

    complete        -> status='done'    outcome='completed'
    request_review  -> status='review'  outcome='review_requested'

With the columns conflated those rows landed as ``status='completed'`` and
``status='review_requested'``, values no consumer filters on, so every query
selecting ``status='done'`` skipped them without erroring.
"""
from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path):
    db = tmp_path / "kanban.db"
    kb.init_db(db)
    conn = kb.connect(db)
    yield conn
    conn.close()


def _run_rows(conn, task_id):
    return [
        (r["status"], r["outcome"])
        for r in conn.execute(
            "SELECT status, outcome FROM task_runs WHERE task_id = ?", (task_id,)
        )
    ]


def test_complete_without_a_claimed_run_writes_status_done(board):
    tid = kb.create_task(board, title="never claimed")
    kb.complete_task(board, tid, summary="closed straight from ready")
    board.commit()

    assert _run_rows(board, tid) == [("done", "completed")]


def test_request_review_without_a_claimed_run_writes_status_review(board):
    tid = kb.create_task(board, title="never claimed")
    kb.request_review(board, tid, summary="straight to review")
    board.commit()

    assert _run_rows(board, tid) == [("review", "review_requested")]


def test_synthesized_run_is_visible_to_a_status_done_filter(board):
    """The regression this guards against was silent: the row existed, so
    nothing errored, but a consumer filtering on the documented terminal
    status could not see it."""
    tid = kb.create_task(board, title="never claimed")
    kb.complete_task(board, tid, summary="s")
    board.commit()

    found = board.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ? AND status = 'done'",
        (tid,),
    ).fetchone()[0]
    assert found == 1


def test_synthesized_status_values_are_in_the_declared_domain(board):
    """Whatever the transitions write must be a member of the frozenset that
    now defines the column, otherwise the domain has drifted again."""
    for title, fn in (
        ("c", kb.complete_task),
        ("r", kb.request_review),
    ):
        tid = kb.create_task(board, title=title)
        fn(board, tid, summary="s")
        board.commit()
        for status, _outcome in _run_rows(board, tid):
            assert status in kb.TASK_RUN_STATUSES, status


def test_declared_domain_covers_the_statuses_the_writers_use(board):
    """``TASK_RUN_STATUSES`` replaced a CREATE TABLE comment that had drifted
    from the code — it listed values nothing writes and omitted values that
    are written. Pin the two that the synthesize path depends on."""
    assert "done" in kb.TASK_RUN_STATUSES
    assert "review" in kb.TASK_RUN_STATUSES
    # The pre-fix values were the outcome names; they are not row states.
    assert "completed" not in kb.TASK_RUN_STATUSES
    assert "review_requested" not in kb.TASK_RUN_STATUSES


def test_unknown_status_warns_but_does_not_raise(caplog):
    """The write-time check is deliberately non-fatal: a bad status is a
    bookkeeping problem, and raising here would take down a worker that is
    otherwise finishing correctly."""
    import logging

    with caplog.at_level(logging.WARNING):
        kb._validate_run_status("not-a-real-status", "test")

    assert any(
        "TASK_RUN_STATUSES" in r.getMessage() for r in caplog.records
    ), caplog.text


def test_known_status_does_not_warn(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        kb._validate_run_status("done", "test")

    assert not [
        r for r in caplog.records if "TASK_RUN_STATUSES" in r.getMessage()
    ]

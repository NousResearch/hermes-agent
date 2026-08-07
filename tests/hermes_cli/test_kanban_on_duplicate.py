"""``--on-duplicate comment`` makes a deduped card show how bad things are.

Deduping is right; silence is not the same thing. A card that says "failed
at 15:18" reads like a one-off even when the job has failed every ten minutes
since, because nothing on the board separates a blip from an ongoing outage.

The throttle exists so the cure is not the disease: a ten-minute job must not
leave 144 comments a day. It bounds *lines*, never the count — the running
total stays exact whether it was written down once or forty-seven times.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


def _comments(conn, task_id):
    return conn.execute(
        "SELECT body, created_at FROM task_comments WHERE task_id = ? "
        "ORDER BY id",
        (task_id,),
    ).fetchall()


def _recurrence_lines(conn, task_id):
    return [c for c in _comments(conn, task_id)
            if c["body"].startswith(kb.RECURRENCE_COMMENT_PREFIX)]


def _file(conn, **kw):
    kw.setdefault("title", "nightly import failed")
    kw.setdefault("idempotency_key", "import:nightly")
    kw.setdefault("on_duplicate", "comment")
    kw.setdefault("recurrence_throttle_seconds", 0)
    return kb.create_task(conn, **kw)


# --- default behaviour is unchanged -----------------------------------------


def test_return_is_the_default_and_stays_silent(board):
    first = kb.create_task(board, title="x", idempotency_key="k")
    second = kb.create_task(board, title="x", idempotency_key="k")
    assert second == first
    assert _comments(board, first) == []


def test_first_call_never_comments(board):
    """Nothing recurred yet — the task was just created."""
    tid = _file(board)
    assert _recurrence_lines(board, tid) == []
    assert kb.count_recurrences(board, tid) == 0


# --- counting ---------------------------------------------------------------


def test_duplicate_records_an_occurrence(board):
    tid = _file(board)
    assert _file(board) == tid
    assert kb.count_recurrences(board, tid) == 1


def test_count_climbs_with_each_duplicate(board):
    tid = _file(board)
    for _ in range(5):
        _file(board)
    assert kb.count_recurrences(board, tid) == 5


def test_unthrottled_leaves_one_line_per_occurrence(board):
    tid = _file(board)
    for _ in range(3):
        _file(board)
    lines = _recurrence_lines(board, tid)
    assert len(lines) == 3
    assert "occurrence #3" in lines[-1]["body"]


# --- throttling bounds lines, never the count -------------------------------


def test_throttle_coalesces_into_one_line(board):
    tid = _file(board, recurrence_throttle_seconds=3600)
    for _ in range(10):
        _file(board, recurrence_throttle_seconds=3600)

    lines = _recurrence_lines(board, tid)
    assert len(lines) == 1, "a job on a timer should not spam the card"
    assert "occurrence #10" in lines[0]["body"]
    assert kb.count_recurrences(board, tid) == 10


def test_throttle_window_is_anchored_not_refreshed(board):
    """The window must expire, or the card stops looking alive.

    If each update pushed `created_at` forward, a frequent job would hold one
    line open forever and you could never tell from the timestamps whether it
    was still happening or had stopped days ago.
    """
    tid = _file(board, recurrence_throttle_seconds=3600)
    _file(board, recurrence_throttle_seconds=3600)
    anchored = _recurrence_lines(board, tid)[0]["created_at"]

    _file(board, recurrence_throttle_seconds=3600)
    lines = _recurrence_lines(board, tid)
    assert len(lines) == 1
    assert lines[0]["created_at"] == anchored


def test_new_line_once_the_window_expires(board):
    tid = _file(board, recurrence_throttle_seconds=3600)
    _file(board, recurrence_throttle_seconds=3600)

    # Backdate the open line so its window has elapsed.
    with kb.write_txn(board):
        board.execute(
            "UPDATE task_comments SET created_at = ? WHERE task_id = ?",
            (int(time.time()) - 7200, tid),
        )

    _file(board, recurrence_throttle_seconds=3600)
    lines = _recurrence_lines(board, tid)
    assert len(lines) == 2, "the window should expire and start a fresh line"
    assert "occurrence #2" in lines[-1]["body"]
    assert kb.count_recurrences(board, tid) == 2


# --- interaction with dedupe_scope ------------------------------------------


def test_a_regression_gets_its_own_card_not_a_comment(board):
    """With scope=open, a closed task is not a duplicate — it is a new fault.

    The recurrence count belongs to the card that is actually open, so the
    fresh card starts from zero rather than inheriting the old one's history.
    """
    first = _file(board, dedupe_scope="open")
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET status='done' WHERE id=?", (first,))

    second = _file(board, dedupe_scope="open")
    assert second != first
    assert kb.count_recurrences(board, second) == 0


# --- guardrails --------------------------------------------------------------


def test_unknown_on_duplicate_is_rejected(board):
    kb.create_task(board, title="x", idempotency_key="k")
    with pytest.raises(ValueError, match="on_duplicate"):
        kb.create_task(board, title="x", idempotency_key="k", on_duplicate="shout")


def test_commenting_never_breaks_the_caller(board, monkeypatch):
    """Dedup is the quiet path; bookkeeping must not turn it into a failure."""
    tid = _file(board)

    def boom(*_a, **_kw):
        raise ValueError("comment subsystem is having a day")

    monkeypatch.setattr(kb, "record_recurrence", boom)
    assert _file(board) == tid

"""``complete_task`` must store a summary-only handoff in ``tasks.result``.

Regression test for the 129-card "empty work results" class: workers finish
with ``summary`` (the kanban_complete schema marks ``result`` legacy), but the
tasks-row write stored ONLY ``result`` — leaving ``tasks.result`` NULL on every
summary-only completion even though ``task_runs.summary`` carried the handoff.
The done card must carry its handoff in BOTH places.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _claimed(conn) -> str:
    tid = kb.create_task(conn, title="demo", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    return tid


def test_summary_only_completion_populates_tasks_result(conn):
    """A summary-only completion must NOT leave tasks.result NULL."""
    tid = _claimed(conn)
    assert kb.complete_task(conn, tid, summary="shipped the fix, verified 5/5") is True
    row = conn.execute("SELECT result FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["result"] == "shipped the fix, verified 5/5", (
        "summary-only completion left tasks.result NULL — the 129-card bug"
    )


def test_explicit_result_stored_verbatim(conn):
    """An explicit ``result`` is stored unchanged (legacy path intact)."""
    tid = _claimed(conn)
    assert kb.complete_task(conn, tid, result="explicit result") is True
    row = conn.execute("SELECT result FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["result"] == "explicit result"


def test_result_beats_summary_when_both_given(conn):
    """When both are supplied, the explicit ``result`` wins on the tasks row."""
    tid = _claimed(conn)
    assert kb.complete_task(conn, tid, result="result wins",
                            summary="summary loses") is True
    row = conn.execute("SELECT result FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["result"] == "result wins"


def test_completed_event_carries_handoff(conn):
    """The 'completed' event payload must carry the handoff, not a NULL result."""
    tid = _claimed(conn)
    assert kb.complete_task(conn, tid, summary="handoff for the event") is True
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'completed'"
        " ORDER BY id", (tid,)).fetchall()
    assert rows, "no completed event row"
    assert "handoff for the event" in rows[-1]["payload"]


def test_empty_summary_still_rejected(conn):
    """The empty-completion guard must still fire on a whitespace summary."""
    tid = _claimed(conn)
    with pytest.raises(kb.EmptyCompletionError):
        kb.complete_task(conn, tid, summary="   ")
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["status"] == "running"

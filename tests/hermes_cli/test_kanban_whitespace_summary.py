"""Regression: completing/editing a task with a whitespace-only summary or
result must not crash.

``complete_task`` and ``edit_completed_task_result`` build a one-line event
preview via ``text.strip().splitlines()[0]`` guarded only by the *truthiness*
of the raw text. A non-empty but whitespace-only string (``"   "``, a bare
newline) is truthy, so the guard passes, but ``.strip().splitlines()`` is
``[]`` and ``[0]`` raises ``IndexError`` inside the write transaction. LLM
workers and ``hermes kanban complete <id> --summary "\\n"`` reach this on the
normal success path.
"""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


@pytest.mark.parametrize("summary", ["   ", "\n\n", "\t"])
def test_complete_task_whitespace_summary_does_not_crash(kanban_home, summary):
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.complete_task(conn, tid, summary=summary) is True


@pytest.mark.parametrize("result", ["   ", "\n"])
def test_edit_completed_result_whitespace_does_not_crash(kanban_home, result):
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.complete_task(conn, tid, result="real") is True
        assert kb.edit_completed_task_result(conn, tid, result=result) is True

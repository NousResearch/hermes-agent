"""Per-task max_turns is stored on the card and passed as hermes --max-turns."""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def test_create_with_max_turns_puts_flag_on_worker_argv(board, monkeypatch):
    monkeypatch.setattr(dispatch, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(dispatch, "_resolve_worker_cli_toolsets", lambda home: None)
    tid = kb.create_task(board, title="plan qa", assignee="qa-verifier", max_turns=40)
    task = kb.get_task(board, tid)
    assert task.max_turns == 40
    argv = dispatch._worker_argv(task, "qa-verifier", None)
    assert "--max-turns" in argv
    assert argv[argv.index("--max-turns") + 1] == "40"


def test_create_without_max_turns_does_not_invent_a_cap(board, monkeypatch):
    monkeypatch.setattr(dispatch, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(dispatch, "_resolve_worker_cli_toolsets", lambda home: None)
    tid = kb.create_task(board, title="code qa", assignee="qa-verifier")
    task = kb.get_task(board, tid)
    assert task.max_turns is None
    argv = dispatch._worker_argv(task, "qa-verifier", None)
    assert "--max-turns" not in argv

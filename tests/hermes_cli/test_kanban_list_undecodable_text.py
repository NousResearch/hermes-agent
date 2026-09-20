"""Invariant: one task row with an undecodable text field cannot take down the
board listing (issue #111743).

A ``body`` stored as TEXT holding invalid UTF-8 made sqlite3 abort the whole
``fetchall`` ("Could not decode to UTF-8 column 'body'"), so ``hermes kanban list``
failed for every card until the row was deleted by hand; a BLOB-typed body came
back as ``bytes`` and crashed ``--json``. Board connections now decode lossily
(U+FFFD) and the ``from_row`` constructors (task, comment, event, run) coerce
BLOB cells the same way, so ``show --json`` survives a BLOB comment too.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return db_path


def test_list_survives_undecodable_and_blob_text_cells(board):
    with kbc.connect() as conn:
        good = kb.create_task(conn, title="good card", assignee="coder")
        conn.execute(
            "INSERT INTO tasks (id, title, body, assignee, status, priority, created_by, created_at) "
            "VALUES ('t_text_bad', 'text invalid', CAST(X'FFFEABCD00' AS TEXT), 'coder', 'ready', 0, 'user', 1000)"
        )
        conn.execute(
            "INSERT INTO tasks (id, title, body, assignee, status, priority, created_by, created_at) "
            "VALUES ('t_blob_bad', 'blob body', X'FFFEABCD00', 'coder', 'ready', 0, 'user', 1000)"
        )

    with kbc.connect() as conn:
        tasks = {t.id: t for t in kb.list_tasks(conn, status="ready")}
        assert {good, "t_text_bad", "t_blob_bad"} <= set(tasks)
        for tid in ("t_text_bad", "t_blob_bad"):
            body = tasks[tid].body
            assert isinstance(body, str) and "\ufffd" in body
            json.dumps(dataclasses.asdict(tasks[tid]))  # the --json path
        assert kb.get_task(conn, "t_text_bad").title == "text invalid"


def test_show_json_survives_blob_cells_in_sibling_tables(board):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="card", assignee="coder")
        conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, 'user', X'FFFEABCD00', 1000)", (tid,),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'note', X'FFFEABCD00', 1000)", (tid,),
        )

    with kbc.connect() as conn:
        comments = kb.list_comments(conn, tid)
        assert isinstance(comments[-1].body, str) and "\ufffd" in comments[-1].body
        events = kb.list_events(conn, tid)
        json.dumps([dataclasses.asdict(c) for c in comments] + [dataclasses.asdict(e) for e in events])

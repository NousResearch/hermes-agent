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
    kb.init_db()
    with kbc.connect() as connection:
        yield connection


def _blocked(conn):
    task_id = kb.create_task(conn, title="wait", assignee="w")
    task = kb.claim_task(conn, task_id)
    assert kb.block_task(
        conn,
        task_id,
        reason="wait",
        kind="dependency",
        expected_run_id=task.current_run_id,
    )
    conn.execute("UPDATE task_runs SET ended_at = ended_at - 5 WHERE task_id = ?", (task_id,))
    conn.commit()
    return task_id


def test_parentless_dependency_outcome_holds_after_promotion(conn):
    task_id = _blocked(conn)

    kb.recompute_ready(conn)

    assert kbd.check_respawn_guard(conn, task_id) == "blocked_outcome"


def test_comment_releases_parentless_blocked_outcome(conn):
    task_id = _blocked(conn)

    kb.add_comment(conn, task_id, author="op", body="new input")

    assert kbd.check_respawn_guard(conn, task_id) is None

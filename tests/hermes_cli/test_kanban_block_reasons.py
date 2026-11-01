"""block_task refusal reasons let stale workers understand lifecycle fences (#133358)."""
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def worker_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, int]:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="block-reason", assignee="test-worker")
        kb.claim_task(conn, task_id)
        run_id = kb.get_task(conn, task_id).current_run_id
    assert run_id is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return task_id, run_id


def test_block_task_with_reason_identifies_stale_run(worker_task):
    task_id, run_id = worker_task
    with kbc.connect() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready', current_run_id = NULL WHERE id = ?",
                (task_id,),
            )

        ok, reason = kb.block_task(
            conn, task_id, reason="stale attempt", expected_run_id=run_id, with_reason=True,
        )

        assert ok is False
        assert "stale run" in reason
        assert str(run_id) in reason
        assert "none" in reason
        assert kb.get_task(conn, task_id).status == "ready"


def test_block_task_reason_mode_preserves_bool_default(worker_task):
    task_id, _ = worker_task
    with kbc.connect() as conn:
        assert kb.block_task(conn, task_id, reason="waiting") is True
        assert kb.block_task(conn, task_id, reason="again") is False


def test_kanban_block_reports_stale_run_reason(worker_task):
    import json

    from tools import kanban_tools as kt

    task_id, run_id = worker_task
    with kbc.connect() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready', current_run_id = NULL WHERE id = ?",
                (task_id,),
            )

    out = json.loads(kt._handle_block({"reason": "stale worker block"}))

    assert "error" in out
    assert "stale run" in out["error"]
    assert str(run_id) in out["error"]
    assert "unknown id" not in out["error"]
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "ready"


def test_kanban_cli_block_reports_stale_run_reason(worker_task, capsys):
    import argparse

    from hermes_cli import kanban as kanban_cli

    task_id, run_id = worker_task
    with kbc.connect() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready', current_run_id = NULL WHERE id = ?",
                (task_id,),
            )

    args = argparse.Namespace(task_id=task_id, ids=None, reason=["stale", "worker"], kind=None)
    assert kanban_cli._cmd_block(args) == 1
    err = capsys.readouterr().err

    assert "stale run" in err
    assert str(run_id) in err
    assert "cannot block" in err
    assert "unknown id" not in err
    with kbc.connect() as conn:
        assert kb.get_task(conn, task_id).status == "ready"

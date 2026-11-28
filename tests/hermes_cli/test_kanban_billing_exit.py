"""Verified billing failures park implementer and reviewer cards until operator recovery."""

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.turn_exit import turn_exit_code


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kbd, "_recent_worker_exits", {})
    kb.init_db()
    return home


@pytest.mark.parametrize("lane", ["ready", "review"])
def test_confirmed_billing_exit_blocks_and_survives_reopen(kanban_home, lane):
    result = {"failed": True, "failure_reason": "billing", "billing_unverified": False}
    code = turn_exit_code(result, kanban_worker=True)
    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="billing wall", assignee="a", max_retries=5)
        claimed = kb.claim_task(conn, tid, claimer=f"{host}:w0")
        assert claimed is not None
        if lane == "review":
            assert kb.request_review(
                conn, tid, summary="done", reviewer="r", expected_run_id=claimed.current_run_id,
            )
            claimed = kb.claim_review_task(conn, tid, claimer=f"{host}:r0")
            assert claimed is not None
        pid = 71001
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
        conn.commit()
        kbd._record_worker_exit(pid, kbd._wait_status_from_returncode(code))
        assert tid in kbd.detect_crashed_workers(conn)
        task = kb.get_task(conn, tid)
        assert task is not None and task.status == "blocked"
        assert task.consecutive_failures == 1
        assert "billing" in (task.last_failure_error or "")
        run = conn.execute("SELECT outcome, metadata FROM task_runs WHERE id=?", (claimed.current_run_id,)).fetchone()
        assert run["outcome"] == "crashed"
        assert json.loads(run["metadata"])["exit_kind"] == "terminal_provider"
        payload = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='gave_up'", (tid,),
        ).fetchone()[0]
        assert json.loads(payload)["terminal_provider"] is True
        kb.recompute_ready(conn, failure_limit=10)
        assert kb.get_task(conn, tid).status == "blocked"
    with kbc.connect() as conn:
        kb.recompute_ready(conn, failure_limit=10)
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.claim_task(conn, tid) is None
        assert kb.claim_review_task(conn, tid) is None

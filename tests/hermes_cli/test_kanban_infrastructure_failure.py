"""Regression coverage for Kanban local infrastructure failure exit isolation."""

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _exited_status(code: int) -> int:
    """Raw wait-status for a WIFEXITED child with the given exit code."""
    return code << 8


def test_infrastructure_exit_requeues_without_counting_failure_and_uses_cooldown(
    kanban_home, monkeypatch,
):
    import hermes_cli.kanban_db as _kb
    from hermes_cli import kanban_db_dispatch as _kbd

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")

    with kbc.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="persistence failure", assignee="a", max_retries=1)
        for i in range(3):
            pid = 72000 + i
            kb.claim_task(conn, tid, claimer=f"{host}:infra{i}")
            conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
            conn.commit()
            _kbd._record_worker_exit(pid, _exited_status(_kb.KANBAN_INFRASTRUCTURE_EXIT_CODE))

            assert tid not in kbd.detect_crashed_workers(conn)
            assert tid in getattr(_kbd.detect_crashed_workers, "_last_infrastructure_failed", [])
            task = kb.get_task(conn, tid)
            assert task.status == ("blocked" if i == 2 else "ready")
            if i == 2:
                assert task.block_kind == "needs_input"
            assert task.consecutive_failures == 0
            assert task.last_failure_error and "local persistence" in task.last_failure_error

            run = conn.execute(
                "SELECT outcome, metadata, ended_at FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (tid,),
            ).fetchone()
            assert run["outcome"] == "infrastructure_failure"
            run_metadata = json.loads(run["metadata"])
            assert run_metadata["infrastructure"] is True
            if i == 2:
                assert run_metadata["retry_status"] == "blocked"
                assert run_metadata["manual_review_required"] is True
            monkeypatch.setattr(_kb.time, "time", lambda: run["ended_at"] + 100)
            assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"


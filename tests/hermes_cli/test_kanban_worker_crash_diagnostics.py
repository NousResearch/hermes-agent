"""Crash diagnostics for dispatcher-spawned Kanban workers."""

from __future__ import annotations

import json
import time
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


def _claim_with_dead_worker(conn, task_id: str, pid: int):
    host = kb._claimer_id().split(":", 1)[0]
    task = kb.claim_task(conn, task_id, claimer=f"{host}:worker")
    assert task is not None and task.current_run_id is not None
    kb._set_worker_pid(conn, task_id, pid)
    conn.execute(
        "UPDATE tasks SET started_at = ? WHERE id = ?",
        (int(time.time()) - 60, task_id),
    )
    conn.commit()
    return task


def test_nonzero_exit_event_has_bounded_redacted_current_run_tail(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kb, "_classify_worker_exit", lambda _pid: ("nonzero_exit", 1))

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="startup crash", assignee="worker")
        task = _claim_with_dead_worker(conn, task_id, 43210)

        log_path = kb.worker_logs_dir() / f"{task_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("previous successful run\n", encoding="utf-8")
        kb._append_worker_run_marker(log_path, task.current_run_id)
        secret = "ghp_abcdefghijklmnopqrstuvwxyz"
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write("x" * (kb.WORKER_DIAGNOSTIC_TAIL_BYTES + 200))
            stream.write(f"\nAuthorization: Bearer {secret}\nstartup exploded\n")

        assert kb.detect_crashed_workers(conn) == [task_id]
        event = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'crashed' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        payload = json.loads(event["payload"])

    tail = payload["worker_log_tail"]
    assert tail.endswith("startup exploded")
    assert "previous successful run" not in tail
    assert secret not in tail
    assert len(tail.encode("utf-8")) <= kb.WORKER_DIAGNOSTIC_TAIL_BYTES


def test_spawn_failure_explicitly_reports_no_worker_output(kanban_home: Path) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="cannot spawn", assignee="worker")
        task = _claim_with_dead_worker(conn, task_id, 43211)
        log_path = kb.worker_logs_dir() / f"{task_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("previous attempt output\n", encoding="utf-8")
        kb._append_worker_run_marker(log_path, task.current_run_id)

        assert not kb._record_spawn_failure(
            conn, task_id, "executable missing", failure_limit=2,
        )
        event = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'spawn_failed' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        payload = json.loads(event["payload"])

    assert payload["error"] == "executable missing"
    assert payload["worker_log_tail"] is None


def test_worker_tail_remains_byte_bounded_for_invalid_utf8(kanban_home: Path) -> None:
    task_id = "t_invalid_utf8"
    run_id = 77
    log_path = kb.worker_logs_dir() / f"{task_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(
        kb._worker_run_marker(run_id)
        + b"\x80" * kb.WORKER_DIAGNOSTIC_TAIL_BYTES
    )

    tail = kb._read_worker_run_log_tail(task_id, run_id)

    assert tail is not None
    assert len(tail.encode("utf-8")) <= kb.WORKER_DIAGNOSTIC_TAIL_BYTES


def test_default_spawn_marks_log_with_current_run(
    kanban_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (kanban_home / "profiles" / "worker").mkdir(parents=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class FakeProc:
        pid = 43212

    monkeypatch.setattr("subprocess.Popen", lambda *_args, **_kwargs: FakeProc())
    monkeypatch.setattr(kb, "_retag_legacy_worker_sessions", lambda _root: None)
    task = kb.Task(
        id="t_marked",
        title="marked",
        body=None,
        assignee="worker",
        status="running",
        priority=0,
        created_by=None,
        created_at=0,
        started_at=0,
        completed_at=None,
        workspace_kind="dir",
        workspace_path=str(workspace),
        claim_lock="host:worker",
        claim_expires=None,
        tenant=None,
        current_run_id=77,
    )

    assert kb._default_spawn(task, str(workspace)) == FakeProc.pid
    log = (kb.worker_logs_dir() / "t_marked.log").read_text(encoding="utf-8")
    assert "HERMES KANBAN WORKER RUN 77" in log

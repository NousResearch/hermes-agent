from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from tools import kanban_tools as kt


@pytest.fixture
def conn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db(board="default")
    with kb.connect_closing(board="default") as connection:
        yield connection


def _claim(conn) -> tuple[str, kb.Task]:
    task_id = kb.create_task(conn, title="worker", assignee="coder")
    claimed = kb.claim_task(conn, task_id, claimer="host:claim")
    assert claimed is not None
    assert claimed.current_run_id is not None
    assert claimed.claim_lock == "host:claim"
    return task_id, claimed


def _run_id(claimed: kb.Task) -> int:
    assert claimed.current_run_id is not None
    return claimed.current_run_id


def _claim_lock(claimed: kb.Task) -> str:
    assert claimed.claim_lock is not None
    return claimed.claim_lock


def _pid_state(conn, task_id: str, run_id: int):
    task_row = conn.execute(
        "SELECT worker_pid FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    run_row = conn.execute(
        "SELECT worker_pid FROM task_runs WHERE id = ?", (run_id,)
    ).fetchone()
    spawned = [event for event in kb.list_events(conn, task_id) if event.kind == "spawned"]
    return task_row["worker_pid"], run_row["worker_pid"], spawned


def _set_worker_identity(
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_id: str,
    run_id: int,
    claim_lock: str,
) -> None:
    monkeypatch.setenv("HERMES_KANBAN_DB", str(kb.kanban_db_path(board="default")))
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", claim_lock)


def test_register_worker_pid_updates_task_and_current_run_once(conn):
    task_id, claimed = _claim(conn)

    result = kb.register_worker_pid(
        conn,
        task_id,
        43210,
        expected_run_id=_run_id(claimed),
        expected_claim_lock=claimed.claim_lock,
        source="dispatcher",
    )
    again = kb.register_worker_pid(
        conn,
        task_id,
        43210,
        expected_run_id=_run_id(claimed),
        expected_claim_lock=claimed.claim_lock,
        source="worker_start",
    )

    task_pid, run_pid, spawned = _pid_state(
        conn, task_id, _run_id(claimed)
    )
    assert result == "registered"
    assert again == "already_registered"
    assert task_pid == run_pid == 43210
    assert len(spawned) == 1
    assert spawned[0].payload == {"pid": 43210, "source": "dispatcher"}


@pytest.mark.parametrize(
    ("pin_change", "expected_pid"),
    [
        ("run", None),
        ("claim", None),
    ],
)
def test_register_worker_pid_rejects_stale_identity(conn, pin_change, expected_pid):
    task_id, claimed = _claim(conn)
    run_id = _run_id(claimed)
    expected_run_id = run_id + 1 if pin_change == "run" else run_id
    expected_claim_lock = (
        "host:stale" if pin_change == "claim" else claimed.claim_lock
    )

    result = kb.register_worker_pid(
        conn,
        task_id,
        43210,
        expected_run_id=expected_run_id,
        expected_claim_lock=expected_claim_lock,
        source="dispatcher",
    )

    task_pid, run_pid, spawned = _pid_state(conn, task_id, run_id)
    assert result == "rejected"
    assert task_pid == run_pid == expected_pid
    assert spawned == []


def test_register_worker_pid_rejects_wrong_claim_lock(conn):
    task_id, claimed = _claim(conn)

    result = kb.register_worker_pid(
        conn,
        task_id,
        43210,
        expected_run_id=_run_id(claimed),
        expected_claim_lock="host:stale",
        source="dispatcher",
    )

    task_pid, run_pid, spawned = _pid_state(
        conn, task_id, _run_id(claimed)
    )
    assert result == "rejected"
    assert task_pid is None
    assert run_pid is None
    assert spawned == []


def test_register_worker_pid_rejects_different_existing_pid(conn):
    task_id, claimed = _claim(conn)
    conn.execute(
        "UPDATE tasks SET worker_pid = ? WHERE id = ?", (11111, task_id)
    )
    conn.execute(
        "UPDATE task_runs SET worker_pid = ? WHERE id = ?",
        (11111, _run_id(claimed)),
    )
    conn.commit()

    result = kb.register_worker_pid(
        conn,
        task_id,
        43210,
        expected_run_id=_run_id(claimed),
        expected_claim_lock=claimed.claim_lock,
        source="worker_start",
    )

    task_pid, run_pid, spawned = _pid_state(
        conn, task_id, _run_id(claimed)
    )
    assert result == "rejected"
    assert task_pid == run_pid == 11111
    assert spawned == []


def test_register_worker_pid_rejects_ended_run(conn):
    task_id, claimed = _claim(conn)
    conn.execute(
        "UPDATE task_runs SET status = 'failed', ended_at = 123 "
        "WHERE id = ?",
        (_run_id(claimed),),
    )
    conn.commit()

    result = kb.register_worker_pid(
        conn,
        task_id,
        43210,
        expected_run_id=_run_id(claimed),
        expected_claim_lock=claimed.claim_lock,
        source="dispatcher",
    )

    task_pid, run_pid, spawned = _pid_state(
        conn, task_id, _run_id(claimed)
    )
    assert result == "rejected"
    assert task_pid is None
    assert run_pid is None
    assert spawned == []


# ---------------------------------------------------------------------------
# Worker self-registration from environment
# ---------------------------------------------------------------------------


def test_register_current_worker_requires_complete_identity(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_demo")
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    assert kt.register_current_worker_from_env() is None

    monkeypatch.setenv("HERMES_KANBAN_DB", "/tmp/kanban.db")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    assert kt.register_current_worker_from_env() is None

    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "host:claim")
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    assert kt.register_current_worker_from_env() is None

    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    assert kt.register_current_worker_from_env() is None


def test_register_current_worker_from_env_registers_real_pid(conn, monkeypatch):
    task_id, claimed = _claim(conn)
    run_id = _run_id(claimed)
    _set_worker_identity(
        monkeypatch,
        task_id=task_id,
        run_id=run_id,
        claim_lock=_claim_lock(claimed),
    )

    result = kt.register_current_worker_from_env(source="worker_start")

    task_pid, run_pid, spawned = _pid_state(conn, task_id, run_id)
    assert result == "registered"
    assert task_pid == run_pid == os.getpid()
    assert len(spawned) == 1
    assert spawned[0].payload["source"] == "worker_start"
    assert spawned[0].payload["pid"] == os.getpid()


def test_register_current_worker_from_env_rejects_stale_run(conn, monkeypatch):
    task_id, claimed = _claim(conn)
    run_id = _run_id(claimed)
    conn.execute(
        "UPDATE task_runs SET status = 'failed', ended_at = 123 "
        "WHERE id = ?",
        (run_id,),
    )
    conn.commit()
    _set_worker_identity(
        monkeypatch,
        task_id=task_id,
        run_id=run_id,
        claim_lock=_claim_lock(claimed),
    )

    result = kt.register_current_worker_from_env(source="worker_start")

    task_pid, run_pid, spawned = _pid_state(conn, task_id, run_id)
    assert result == "rejected"
    assert task_pid is None
    assert run_pid is None
    assert spawned == []


def test_register_current_worker_from_env_rejects_wrong_claim_lock(conn, monkeypatch):
    task_id, claimed = _claim(conn)
    run_id = _run_id(claimed)
    _set_worker_identity(
        monkeypatch,
        task_id=task_id,
        run_id=run_id,
        claim_lock="host:stale",
    )

    result = kt.register_current_worker_from_env(source="worker_start")

    task_pid, run_pid, spawned = _pid_state(conn, task_id, run_id)
    assert result == "rejected"
    assert task_pid is None
    assert run_pid is None
    assert spawned == []


def test_heartbeat_current_worker_from_env_repairs_registration(
    conn, monkeypatch
):
    task_id, claimed = _claim(conn)
    run_id = _run_id(claimed)
    _set_worker_identity(
        monkeypatch,
        task_id=task_id,
        run_id=run_id,
        claim_lock=_claim_lock(claimed),
    )

    repair_calls: list[str] = []
    original = kt.register_current_worker_from_env

    def _fake_register(*, source: str = "worker_start") -> str | None:
        repair_calls.append(source)
        return original(source=source)

    monkeypatch.setattr(kt, "register_current_worker_from_env", _fake_register)
    monkeypatch.setattr(kt, "_auto_heartbeat_last_attempt", 0.0)
    monkeypatch.setattr(
        "time.monotonic", lambda: kt._AUTO_HEARTBEAT_MIN_INTERVAL_SECONDS + 1.0
    )

    kt.heartbeat_current_worker_from_env()

    assert repair_calls == ["heartbeat_repair"]
    task_pid, run_pid, _ = _pid_state(conn, task_id, run_id)
    assert task_pid == run_pid == os.getpid()


def test_register_current_worker_from_env_in_subprocess(conn, monkeypatch):
    task_id, claimed = _claim(conn)
    run_id = _run_id(claimed)
    db_path = kb.kanban_db_path(board="default")

    claim_lock = claimed.claim_lock
    assert claim_lock is not None
    child_env = os.environ.copy()
    child_env.update(
        HERMES_KANBAN_DB=str(db_path),
        HERMES_KANBAN_TASK=task_id,
        HERMES_KANBAN_RUN_ID=str(run_id),
        HERMES_KANBAN_CLAIM_LOCK=claim_lock,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tools.kanban_tools import register_current_worker_from_env; "
            "import os; "
            "result = register_current_worker_from_env(); "
            "assert result == 'registered'; "
            "print(result, os.getpid())",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=child_env,
        timeout=15,
    )
    result, child_pid = proc.stdout.strip().split()
    assert result == "registered"

    with kb.connect_closing(board="default") as check_conn:
        task_pid, run_pid, spawned = _pid_state(check_conn, task_id, run_id)
        assert task_pid == run_pid == int(child_pid)
        assert len(spawned) == 1
        assert spawned[0].payload["source"] == "worker_start"

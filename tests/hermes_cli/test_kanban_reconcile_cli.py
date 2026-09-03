"""Fail-closed runtime reconcile for running kanban workers.

Uses real SQLite files and real subprocesses whose argv contains the
consecutive tokens ``work kanban task <id>``. Process matching must be
exact-token, never a substring of the task id.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db(board="default")
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect_closing(board="default") as connection:
        yield connection


def _claim_running(conn, title: str = "runtime") -> kb.Task:
    tid = kb.create_task(conn, title=title, assignee="coder")
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    assert claimed.status == "running"
    assert claimed.current_run_id is not None
    return claimed


def _age_past_pid_grace(conn, task_id: str) -> None:
    conn.execute(
        "UPDATE tasks SET started_at = ? WHERE id = ?",
        (int(time.time()) - 60, task_id),
    )
    conn.commit()


def _task_row(conn, task_id: str):
    return conn.execute(
        "SELECT status, claim_lock, worker_pid, current_run_id, last_heartbeat_at "
        "FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()


@contextmanager
def _matching_worker(task_id: str, extra_token: str | None = None):
    argv = [
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        "work",
        "kanban",
        "task",
        task_id if extra_token is None else extra_token,
    ]
    proc = subprocess.Popen(argv)
    try:
        time.sleep(0.05)
        assert proc.poll() is None
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _parse_cli(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    kc.build_parser(sub)
    return parser.parse_args(["kanban", *argv])


def _run_cli(*argv: str) -> tuple[int, str, str]:
    import contextlib
    import io

    args = _parse_cli(*argv)
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        rc = kc.kanban_command(args)
    return int(rc or 0), buf_out.getvalue(), buf_err.getvalue()


def test_consecutive_tokens_do_not_match_task_id_prefix(conn):
    from hermes_cli.kanban_reconcile import inspect_task_runtime

    task = _claim_running(conn)
    _age_past_pid_grace(conn, task.id)
    task = kb.get_task(conn, task.id)
    assert task is not None

    with _matching_worker(task.id, extra_token=f"{task.id}x"):
        finding = inspect_task_runtime(conn, task, board="default")

    assert finding.classification == "missing_pid_no_process"
    assert finding.matching_pids == ()
    assert finding.fix_allowed is True


def test_missing_pid_no_process_fix_requeues(conn):
    from hermes_cli.kanban_reconcile import reconcile_board

    task = _claim_running(conn)
    _age_past_pid_grace(conn, task.id)

    findings = reconcile_board(board="default", task_id=task.id, fix=True)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.classification == "missing_pid_no_process"
    assert finding.fix_allowed is True
    assert finding.fixed is True
    assert finding.matching_pids == ()
    row = _task_row(conn, task.id)
    assert row["status"] == "ready"
    assert row["claim_lock"] is None
    assert row["worker_pid"] is None
    assert row["current_run_id"] is None
    run = kb.latest_run(conn, task.id)
    assert run is not None
    assert run.status == "reclaimed"
    events = [event for event in kb.list_events(conn, task.id) if event.kind == "reconciled"]
    assert len(events) == 1
    assert events[0].payload["reason"] == "missing_pid_no_process"


def test_live_unregistered_process_refuses_fix(conn):
    from hermes_cli.kanban_reconcile import reconcile_board

    task = _claim_running(conn)
    _age_past_pid_grace(conn, task.id)

    with _matching_worker(task.id) as proc:
        findings = reconcile_board(board="default", task_id=task.id, fix=True)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.classification == "live_process_unregistered"
        assert finding.fix_allowed is False
        assert finding.fixed is False
        assert finding.matching_pids == (proc.pid,)
        assert "sleep" not in finding.detail
        assert "import time" not in finding.detail
        row = _task_row(conn, task.id)
        assert row["status"] == "running"
        assert row["claim_lock"] is not None
        assert row["current_run_id"] == task.current_run_id


def test_duplicate_live_workers_refuse_fix(conn):
    from hermes_cli.kanban_reconcile import reconcile_board

    task = _claim_running(conn)
    _age_past_pid_grace(conn, task.id)

    with _matching_worker(task.id) as first, _matching_worker(task.id) as second:
        findings = reconcile_board(board="default", task_id=task.id, fix=True)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.classification == "duplicate_live_workers"
        assert finding.fix_allowed is False
        assert finding.fixed is False
        assert set(finding.matching_pids) == {first.pid, second.pid}
        assert _task_row(conn, task.id)["status"] == "running"


def test_registered_pid_mismatch_refuses_fix(conn):
    from hermes_cli.kanban_reconcile import reconcile_board

    task = _claim_running(conn)
    impostor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        time.sleep(0.05)
        assert impostor.poll() is None
        result = kb.register_worker_pid(
            conn,
            task.id,
            impostor.pid,
            expected_run_id=task.current_run_id,
            expected_claim_lock=task.claim_lock,
            source="dispatcher",
        )
        assert result == "registered"
        findings = reconcile_board(board="default", task_id=task.id, fix=True)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.classification == "registered_pid_mismatch"
        assert finding.registered_pid == impostor.pid
        assert finding.fix_allowed is False
        assert finding.fixed is False
        assert _task_row(conn, task.id)["status"] == "running"
        assert _task_row(conn, task.id)["worker_pid"] == impostor.pid
    finally:
        impostor.terminate()
        impostor.wait(timeout=5)


def test_cas_returns_false_when_claim_lock_changes(conn):
    from hermes_cli.kanban_db import reconcile_running_task_if_unchanged

    task = _claim_running(conn)
    original_lock = task.claim_lock
    conn.execute(
        "UPDATE tasks SET claim_lock = ? WHERE id = ?",
        ("other-host:1", task.id),
    )
    conn.commit()

    changed = reconcile_running_task_if_unchanged(
        conn,
        task.id,
        expected_run_id=task.current_run_id,
        expected_claim_lock=original_lock,
        expected_worker_pid=task.worker_pid,
        reason="missing_pid_no_process",
    )

    assert changed is False
    row = _task_row(conn, task.id)
    assert row["status"] == "running"
    assert row["claim_lock"] == "other-host:1"
    assert row["current_run_id"] == task.current_run_id


def test_healthy_registered_matching_worker_is_left_alone(conn):
    from hermes_cli.kanban_reconcile import reconcile_board

    task = _claim_running(conn)
    with _matching_worker(task.id) as proc:
        result = kb.register_worker_pid(
            conn,
            task.id,
            proc.pid,
            expected_run_id=task.current_run_id,
            expected_claim_lock=task.claim_lock,
            source="dispatcher",
        )
        assert result == "registered"
        findings = reconcile_board(board="default", task_id=task.id, fix=True)
        assert len(findings) == 1
        finding = findings[0]
        assert finding.classification == "healthy"
        assert finding.fix_allowed is False
        assert finding.fixed is False
        assert finding.matching_pids == (proc.pid,)
        row = _task_row(conn, task.id)
        assert row["status"] == "running"
        assert row["worker_pid"] == proc.pid


def test_cli_unknown_task_returns_1(kanban_home):
    rc, _out, err = _run_cli("reconcile", "t_missing1", "--json")
    assert rc == 1
    assert "t_missing1" in err


def test_cli_all_boards_with_board_flag_returns_2(kanban_home):
    rc, _out, err = _run_cli("--board", "default", "reconcile", "--all-boards")
    assert rc == 2
    assert "--all-boards" in err
    assert "--board" in err


def test_cli_fix_on_live_unregistered_is_nonzero_and_unchanged(conn):
    task = _claim_running(conn)
    _age_past_pid_grace(conn, task.id)
    with _matching_worker(task.id):
        rc, out, err = _run_cli("reconcile", task.id, "--fix", "--json")
        assert rc != 0
        payload = json.loads(out)
        assert payload[0]["classification"] == "live_process_unregistered"
        assert payload[0]["fixed"] is False
        combined = out + err
        assert "unchanged" in combined.lower() or "refuse" in combined.lower()
        assert "import time" not in combined
        assert _task_row(conn, task.id)["status"] == "running"


def test_cli_readonly_does_not_mutate_missing_pid(conn):
    task = _claim_running(conn)
    _age_past_pid_grace(conn, task.id)
    rc, out, _err = _run_cli("reconcile", task.id, "--json")
    assert rc == 0
    payload = json.loads(out)
    assert payload[0]["classification"] == "missing_pid_no_process"
    assert payload[0]["fixed"] is False
    assert _task_row(conn, task.id)["status"] == "running"


def test_reconcile_is_denied_in_delegated_child(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    rc, _out, err = _run_cli("reconcile", "--fix")
    assert rc == 1
    assert "delegate_task child contexts cannot mutate Kanban tasks via the CLI" in err
    assert "reconcile" in kc._DELEGATED_CHILD_DENIED_ACTIONS


def test_live_unregistered_fix_allowed_injection_is_refused(conn, monkeypatch):
    """If live_process_unregistered were marked fixable, --fix would steal the card.

    This is the mutation-sensitive contract: flipping that classification to
    fix_allowed must fail this test.
    """
    from hermes_cli import kanban_reconcile as kr

    task = _claim_running(conn)
    _age_past_pid_grace(conn, task.id)

    original = kr.inspect_task_runtime

    def _injected(conn, task, *, board, process_snapshot=None):
        finding = original(
            conn, task, board=board, process_snapshot=process_snapshot
        )
        if finding.classification == "live_process_unregistered":
            return kr.ReconcileFinding(
                board=finding.board,
                db_path=finding.db_path,
                task_id=finding.task_id,
                classification=finding.classification,
                task_status=finding.task_status,
                current_run_id=finding.current_run_id,
                registered_pid=finding.registered_pid,
                matching_pids=finding.matching_pids,
                fix_allowed=True,
                fixed=False,
                detail=finding.detail,
            )
        return finding

    monkeypatch.setattr(kr, "inspect_task_runtime", _injected)

    with _matching_worker(task.id):
        findings = kr.reconcile_board(board="default", task_id=task.id, fix=True)

    assert findings[0].classification == "live_process_unregistered"
    assert findings[0].fix_allowed is False
    assert findings[0].fixed is False
    assert _task_row(conn, task.id)["status"] == "running"


def test_real_cli_subprocess_smoke(kanban_home):
    with kb.connect_closing(board="default") as conn:
        task = _claim_running(conn)
        _age_past_pid_grace(conn, task.id)
        task_id = task.id

    env = os.environ.copy()
    env["HERMES_HOME"] = str(kanban_home)
    env["HERMES_KANBAN_HOME"] = str(kanban_home)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("HERMES_KANBAN_BOARD", None)
    env.pop("HERMES_KANBAN_DB", None)
    env.pop("HERMES_DELEGATED_CHILD_CONTEXT", None)

    with _matching_worker(task_id):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "hermes_cli.main",
                "kanban",
                "reconcile",
                task_id,
                "--fix",
                "--json",
            ],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    assert proc.returncode != 0
    payload = json.loads(proc.stdout)
    assert payload[0]["task_id"] == task_id
    assert payload[0]["classification"] == "live_process_unregistered"
    assert payload[0]["fixed"] is False
    assert "import time" not in proc.stdout
    assert "import time" not in proc.stderr
    with kb.connect_closing(board="default") as conn:
        assert _task_row(conn, task_id)["status"] == "running"

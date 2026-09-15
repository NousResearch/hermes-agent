"""Crash-cause surfacing from worker logs in ``detect_crashed_workers``.

A worker that dies before doing real work (e.g. ``--skill`` references a
skill the worker profile cannot resolve) used to surface only
``pid N exited with code 1`` / ``pid N not alive`` in ``hermes kanban show``
and diagnostics; the actual error line was buried in the per-board worker
log. The reclaim path now reads the worker log tail and appends the cause,
and the dispatch ``board`` is threaded through so the lookup reads the right
board's log even when the explicit dispatch board differs from the current
board (worker logs are per-board precisely because task IDs can collide
across boards).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # The reaped-exit registry is process-global; clear it so a pid recorded
    # by an earlier test in the same run cannot reclassify this test's worker.
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


def _exited_status(code: int) -> int:
    """Raw wait-status for a WIFEXITED child with the given exit code."""
    return code << 8


def _plant_running_task(conn, *, pid: int) -> str:
    """Create + claim a task and point it at a host-local dead PID."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="crash-log", assignee="a")
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=? WHERE id=?",
        (pid, tid),
    )
    conn.commit()
    return tid


def _plant_worker_log(task_id: str, text: str, *, board=None) -> None:
    path = kb.worker_log_path(task_id, board=board)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _last_run_error(conn, task_id: str):
    row = conn.execute(
        "SELECT error FROM task_runs WHERE task_id=? AND error IS NOT NULL "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return row["error"] if row else None


def test_nonzero_exit_appends_cause_from_log(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kbc.connect() as conn:
        tid = _plant_running_task(conn, pid=91001)
        kbd._record_worker_exit(91001, _exited_status(1))
        _plant_worker_log(
            tid,
            "Query: work kanban task t1\nInitializing agent...\n"
            "Error: Unknown skill(s): org-design-pipeline\n",
        )

        crashed = kbd.detect_crashed_workers(conn)
        assert tid in crashed
        error = _last_run_error(conn, tid)
        assert error is not None
        assert error.startswith("pid 91001 exited with code 1")
        assert "→ Error: Unknown skill(s): org-design-pipeline" in error


def test_not_alive_worker_also_surfaces_cause(kanban_home, monkeypatch):
    """The shayani shape: the worker exits so fast the dispatcher never
    reaps an exit status, so only ``pid N not alive`` was recorded."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kbc.connect() as conn:
        tid = _plant_running_task(conn, pid=91002)
        # No _record_worker_exit call: classify falls to ("unknown", None).
        _plant_worker_log(
            tid,
            "Warning: Unknown toolsets: messaging\nInitializing agent...\n"
            "Error: Unknown skill(s): eventaservo-modernizacao\n",
        )

        crashed = kbd.detect_crashed_workers(conn)
        assert tid in crashed
        error = _last_run_error(conn, tid)
        assert error is not None
        assert error.startswith("pid 91002 not alive")
        assert "→ Error: Unknown skill(s): eventaservo-modernizacao" in error


def test_missing_log_keeps_plain_error(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kbc.connect() as conn:
        tid = _plant_running_task(conn, pid=91003)
        kbd._record_worker_exit(91003, _exited_status(1))

        crashed = kbd.detect_crashed_workers(conn)
        assert tid in crashed
        assert _last_run_error(conn, tid) == "pid 91003 exited with code 1"


def test_rate_limited_and_protocol_violation_keep_own_message(
    kanban_home, monkeypatch,
):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kbc.connect() as conn:
        tid = _plant_running_task(conn, pid=91004)
        kbd._record_worker_exit(91004, _exited_status(0))
        _plant_worker_log(tid, "Error: log noise that must not attach\n")

        kbd.detect_crashed_workers(conn)
        error = _last_run_error(conn, tid)
        assert error is not None
        assert "→" not in error
        assert "log noise" not in error


def test_explicit_board_reads_dispatch_board_log(kanban_home, monkeypatch):
    """The gateway dispatches each board with ``board=slug``; the crash-cause
    lookup must read that board's log, not the active-board chain's."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kbc.connect() as conn:
        tid = _plant_running_task(conn, pid=91005)
        kbd._record_worker_exit(91005, _exited_status(1))
        _plant_worker_log(
            tid, "Error: Unknown skill(s): alpha-only-skill\n", board="alpha",
        )

        crashed = kbd.detect_crashed_workers(conn, board="alpha")
        assert tid in crashed
        error = _last_run_error(conn, tid)
        assert error is not None
        assert "→ Error: Unknown skill(s): alpha-only-skill" in error


def test_board_mismatch_does_not_attach_colliding_log(kanban_home, monkeypatch):
    """Worker logs are per-board because task IDs can collide across boards;
    an explicit dispatch board must not pick up another board's log for the
    same task ID (the non-default-board regression from the sweeper review)."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kbc.connect() as conn:
        tid = _plant_running_task(conn, pid=91006)
        kbd._record_worker_exit(91006, _exited_status(1))
        # Same task ID logged under BOTH boards with different causes.
        _plant_worker_log(tid, "Error: WRONG-BOARD default-log\n")
        _plant_worker_log(
            tid, "Error: Unknown skill(s): right-board\n", board="alpha",
        )

        crashed = kbd.detect_crashed_workers(conn, board="alpha")
        assert tid in crashed
        error = _last_run_error(conn, tid)
        assert error is not None
        assert "→ Error: Unknown skill(s): right-board" in error
        assert "WRONG-BOARD" not in error


def test_dispatch_once_threads_board_to_reclaim(kanban_home, monkeypatch):
    """End-to-end for the threading: ``dispatch_once(board=...)`` reaches the
    reclaim-phase lookup (``_dispatch_once_locked`` → ``_run_reclaim_phase``
    → ``detect_crashed_workers``)."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kbc.connect() as conn:
        tid = _plant_running_task(conn, pid=91007)
        kbd._record_worker_exit(91007, _exited_status(1))
        _plant_worker_log(
            tid, "Error: Unknown skill(s): via-dispatch-once\n", board="beta",
        )

        result = kbd.dispatch_once(conn, board="beta")
        assert tid in result.crashed
        error = _last_run_error(conn, tid)
        assert error is not None
        assert "→ Error: Unknown skill(s): via-dispatch-once" in error

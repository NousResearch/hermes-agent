"""End-to-end: crash a real kanban worker through the exit-envelope lifecycle.

Drives the production path with only the agent brain stubbed (a script that
exits 7 after printing a distinctive token to the worker log):

    dispatch_once()  ->  envelope spawn (kb._default_spawn wrapper)
        -> real child process: python -m hermes_cli.kanban_worker_outcomes -- <stub> ...
        -> child prints token, exits 7; wrapper writes the terminal receipt
        -> later dispatch tick: the reaper matches the receipt and books
           provider-neutral TASK_LOGIC against the task budget
        -> respawn guard trips at the failure limit

Everything below the brain is real: dispatch lane, spawn wrapper, worker
subprocess, receipt file, reaper, breaker accounting.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from functools import partial
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_dispatch import dispatch_once
from hermes_cli.kanban_worker_outcomes import worker_exit_envelope_path

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="spawn flags (CREATE_NO_WINDOW / kill_on_parent_exit) are Windows-specific",
)

CRASH_EXIT_CODE = 7
FAIL_LIMIT = 2
CRASH_TOKEN = "e2e-envelope-crash-7Q2"


def _base_python() -> str:
    """Real interpreter executable (pid-parity with Popen).

    uv-managed venvs launch via a trampoline shim whose Popen pid is the shim's,
    not the interpreter that actually runs — spawning the shim would make the
    dispatcher stamp the shim's pid while the wrapper receipt carries the real
    interpreter's pid, so reaper matching could never succeed.
    """
    import os

    return str(Path(os.__file__).resolve().parent.parent / "python.exe")


def _crash_stub() -> str:
    """Replaces the hermes CLI: prints the token to the worker log, exits 7."""
    return (
        "import sys, time\n"
        f"print({CRASH_TOKEN!r})\n"
        "sys.stdout.flush()\n"
        "time.sleep(0.2)\n"
        f"sys.exit({CRASH_EXIT_CODE})\n"
    )


def _success_stub() -> str:
    return "import sys\nsys.exit(0)\n"


@pytest.fixture()
def sandbox(monkeypatch, tmp_path):
    """Isolated kanban home + profile env; the stub replaces the hermes CLI.

    ``_resolve_hermes_argv`` normally returns the hermes executable; here it
    returns ``python -c <stub>`` (a valid replacement — the trailing ``-p``,
    ``--cli`` … flags land in the stub's ``sys.argv`` and are ignored).
    """
    home = tmp_path / "hermes-home"
    (home / "profiles" / "default" / ".hermes").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home / ".hermes"))
    monkeypatch.setenv("HERMES_PROFILE", "default")
    # The wrapper child runs with the scratch workspace as CWD; make the
    # source tree importable so ``-m hermes_cli.kanban_worker_outcomes``
    # resolves (an installed hermes would not need this).
    repo_root = str(Path(__file__).resolve().parents[2])
    monkeypatch.setenv("PYTHONPATH", repo_root)
    stub_argv = {"cmd": None}
    # The wrapper interpreter must have pid parity with its Popen handle:
    # uv-managed venvs launch via a trampoline shim, so spawning
    # ``sys.executable`` stamps the shim's pid while the wrapper receipt
    # carries the real interpreter's pid — reaper matching can never succeed.
    # (Production follow-up: resolve a parity interpreter in _default_spawn.)
    monkeypatch.setattr(sys, "executable", _base_python())
    monkeypatch.setattr(
        kb, "_resolve_hermes_argv", lambda: [_base_python(), *stub_argv["cmd"]]
    )
    monkeypatch.setattr(kb, "_resolve_worker_cli_toolsets", lambda _home: None)
    return {
        "home": home,
        "kanban_home": home / ".hermes",
        "stub_argv": stub_argv,
    }


def _make_task(conn: sqlite3.Connection, sandbox, *, stub: str) -> str:
    sandbox["stub_argv"]["cmd"] = ["-c", stub]
    return kb.create_task(conn, title="e2e envelope crash", assignee="default")


def _spawn_tick(conn: sqlite3.Connection):
    """Envelope-enabled spawn: ``_call_spawn_fn`` passes only
    ``(task, workspace, board=board)``, so the wrapper flags must be bound
    here — exactly the hook a systemd-hosted dispatcher uses."""
    return dispatch_once(
        conn,
        spawn_fn=partial(
            kb._default_spawn, _write_exit_envelope=True, _detach_worker=True,
        ),
        max_spawn=1,
    )


def _reap_tick(conn: sqlite3.Connection):
    """The production reclaim phase (exactly what dispatch's reclaim calls)
    with the launch-window grace disabled, so the crashed worker is reaped
    immediately. Reclaim-only: a dispatch tick would re-spawn the ready task
    in the same pass, which the spawn ticks below test explicitly."""
    import os

    old = os.environ.get("HERMES_KANBAN_CRASH_GRACE_SECONDS")
    os.environ["HERMES_KANBAN_CRASH_GRACE_SECONDS"] = "0"
    try:
        return kb.detect_crashed_workers(conn)
    finally:
        if old is None:
            os.environ.pop("HERMES_KANBAN_CRASH_GRACE_SECONDS", None)
        else:
            os.environ["HERMES_KANBAN_CRASH_GRACE_SECONDS"] = old


def _wait_for_receipt(logs_dir: Path, task_id: str, run_id: int) -> Path:
    envelope = worker_exit_envelope_path(logs_dir, task_id, run_id)
    deadline = time.time() + 30
    while time.time() < deadline and not envelope.exists():
        time.sleep(0.2)
    assert envelope.exists(), "wrapper never wrote the terminal exit receipt"
    return envelope


def _claimed_row(conn: sqlite3.Connection, task_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT status, worker_pid, current_run_id FROM tasks WHERE id=?",
        (task_id,),
    ).fetchone()
    assert row["status"] == "running"
    assert row["worker_pid"]
    return row


def test_real_worker_crash_writes_matched_receipt_and_trips_breaker(sandbox):
    conn = kb.connect()
    try:
        task_id = _make_task(conn, sandbox, stub=_crash_stub())

        first = _spawn_tick(conn)
        assert first.crashed == []
        row = _claimed_row(conn, task_id)
        run_id = int(row["current_run_id"])

        # Wait for the real child to run the stub, exit 7, and for the wrapper
        # to write the terminal receipt atomically.
        envelope = _wait_for_receipt(kb.worker_logs_dir(), task_id, run_id)
        log_text = (
            kb.worker_logs_dir() / f"{task_id}.log"
        ).read_text(encoding="utf-8", errors="replace")
        assert CRASH_TOKEN in log_text

        second = _reap_tick(conn)
        assert second == [task_id]
        row = conn.execute(
            "SELECT status, consecutive_failures FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["consecutive_failures"] == 1

        # Run-outcome label follows main's reaper contract ("crashed"); the
        # typed receipt lives in the run metadata / event payload.
        run = conn.execute(
            "SELECT outcome, error, metadata FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()
        assert run["outcome"] == "crashed"
        assert CRASH_TOKEN in (run["error"] or "")

        event = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        assert event["kind"] == "crashed"
        assert "task_logic" in (event["payload"] or "")

        # Second crash: breaker trips at the failure limit.
        third = _spawn_tick(conn)
        assert third.crashed == []
        row = _claimed_row(conn, task_id)
        run_id2 = int(row["current_run_id"])
        _wait_for_receipt(kb.worker_logs_dir(), task_id, run_id2)

        fourth = _reap_tick(conn)
        assert fourth == [task_id]
        row = conn.execute(
            "SELECT status, consecutive_failures FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "blocked"
        assert row["consecutive_failures"] == FAIL_LIMIT
    finally:
        conn.close()


def test_success_exit_is_booked_as_protocol_violation(sandbox):
    conn = kb.connect()
    try:
        task_id = _make_task(conn, sandbox, stub=_success_stub())

        first = _spawn_tick(conn)
        assert first.crashed == []
        row = _claimed_row(conn, task_id)
        run_id = int(row["current_run_id"])
        _wait_for_receipt(kb.worker_logs_dir(), task_id, run_id)

        # Exit 0 while 'running' is a worker protocol violation with a bounded
        # violation-only budget (never the unified task budget).
        second = _reap_tick(conn)
        assert second == [task_id]
        row = conn.execute(
            "SELECT status, consecutive_failures, last_failure_error FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["consecutive_failures"] == 0
        assert "protocol violation" in (row["last_failure_error"] or "").casefold()
        # A clean exit without a terminal kanban call is a typed
        # ``worker_protocol`` violation run; zero task-budget charge.
        run = conn.execute(
            "SELECT outcome, error FROM task_runs WHERE id=?", (run_id,)
        ).fetchone()
        assert run["outcome"] == "worker_protocol"
        assert "protocol violation" in (run["error"] or "").casefold()
    finally:
        conn.close()

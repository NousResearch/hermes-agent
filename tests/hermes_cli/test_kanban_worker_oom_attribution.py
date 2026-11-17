"""A worker whose scope the kernel OOM-killed is booked as that, not as a protocol violation.

A systemd-hosted dispatcher runs each worker in a transient
``hermes-worker-kanban-<task>-run-<n>.scope`` with the default ``OOMPolicy=stop``: when the
kernel kills a process inside it, systemd stops the whole scope and the worker's signal path
exits 0 with no terminal kanban call made, or the scope is gone before the sweep polls the pid.
The sweep used to read the rc=0 as a worker that skipped its paperwork, book a protocol
violation and feed the violation streak, so a card could ``give_up`` on a death the kernel
caused, and nothing on the board pointed at memory.

``--collect`` removes the scope on exit, so only the user manager's journal still knows. The
sweep asks it about the run's own unit by exact name (a glob over every run's scope is slow on
a busy journal) and BEFORE it opens its write transaction, so a slow journal never holds the
board's write lock.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

_OOM_LINES = (
    "hermes-worker-kanban-x.scope: The kernel OOM killer killed some processes in this unit.\n"
    "hermes-worker-kanban-x.scope: Failed with result 'oom-kill'.\n"
)
_CLEAN_LINES = "hermes-worker-kanban-x.scope: Deactivated successfully.\n"


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


@pytest.fixture
def journal(monkeypatch: pytest.MonkeyPatch):
    """A fake ``journalctl`` on PATH; every other subprocess runs for real.

    ``journal.stdout`` is what the fake prints; ``journal.calls`` records each argv together
    with whether the sweep's connection was inside a transaction at that moment.
    """

    class _Journal:
        stdout = ""
        conn = None
        on_call = None

        def __init__(self) -> None:
            self.calls: list[tuple[list[str], bool]] = []

    fake = _Journal()
    real_run, real_which = subprocess.run, shutil.which

    def run(argv, *args, **kwargs):
        if not (isinstance(argv, (list, tuple)) and argv and argv[0] == "journalctl"):
            return real_run(argv, *args, **kwargs)
        fake.calls.append((list(argv), bool(fake.conn is not None and fake.conn.in_transaction)))
        if fake.on_call is not None:
            fake.on_call()
        return subprocess.CompletedProcess(argv, 0, stdout=fake.stdout, stderr="")

    def which(name, *args, **kwargs):
        return "/usr/bin/journalctl" if name == "journalctl" else real_which(name, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(shutil, "which", which)
    return fake


def _dead_worker(conn, tid: str, pid: int, rc: int | None) -> int:
    """Claim *tid* for a worker that already died; *rc* is its logged exit, None for no trailer."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    if rc is not None:
        log = kb.worker_log_path(tid)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(f"working\n{KANBAN_WORKER_EXIT_TRAILER}{rc}\n", encoding="utf-8")
    return kb.get_task(conn, tid).current_run_id


def _last_booking(conn, tid: str):
    event = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)
    ).fetchone()
    run = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    return event, run, kb._json_dict(run["metadata"])


def test_oom_killed_clean_exit_is_a_crash_not_a_protocol_violation(kanban_home, journal):
    journal.stdout = _OOM_LINES
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker(conn, tid, 70001, rc=0)

        assert kbd.detect_crashed_workers(conn) == [tid]

        event, run, meta = _last_booking(conn, tid)
        assert event["kind"] == "crashed"
        assert meta.get("oom_killed") is True
        assert "protocol_violation" not in meta
        assert "OOM killer stopped its worker scope" in run["error"]
        assert "no terminal kanban call was possible" in run["error"]
        assert kbd._protocol_violation_streak(conn, tid) == 0


def test_oom_killed_bare_death_names_the_cause(kanban_home, journal):
    """No trailer and no reaped status: the scope was collected before the sweep."""
    journal.stdout = _OOM_LINES
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker(conn, tid, 70002, rc=None)

        kbd.detect_crashed_workers(conn)

        event, run, meta = _last_booking(conn, tid)
        assert event["kind"] == "crashed"
        assert meta.get("oom_killed") is True
        assert run["error"].startswith("pid 70002 not alive - the kernel OOM killer stopped")


def test_clean_exit_without_an_oom_stays_a_protocol_violation(kanban_home, journal):
    journal.stdout = _CLEAN_LINES
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker(conn, tid, 70003, rc=0)

        kbd.detect_crashed_workers(conn)

        event, _run, meta = _last_booking(conn, tid)
        assert event["kind"] == "protocol_violation"
        assert meta.get("protocol_violation") is True
        assert "oom_killed" not in meta


def test_rate_limited_exit_keeps_its_own_booking(kanban_home, journal):
    journal.stdout = _OOM_LINES
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker(conn, tid, 70004, rc=kb.KANBAN_RATE_LIMIT_EXIT_CODE)

        kbd.detect_crashed_workers(conn)

        event, _run, meta = _last_booking(conn, tid)
        assert event["kind"] == "rate_limited"
        assert "oom_killed" not in meta


def test_journal_is_asked_for_this_runs_own_unit_outside_the_write_txn(kanban_home, journal):
    journal.stdout = _OOM_LINES
    with kbc.connect() as conn:
        journal.conn = conn
        tid = kb.create_task(conn, title="t", assignee="a")
        run_id = _dead_worker(conn, tid, 70005, rc=0)

        kbd.detect_crashed_workers(conn)

    assert len(journal.calls) == 1, journal.calls
    argv, in_txn = journal.calls[0]
    assert in_txn is False, "the journal was read while the sweep held the board's write lock"
    unit = argv[argv.index("-u") + 1]
    # The exact unit of THIS run: an indexed match, and no earlier run of the card can answer.
    assert unit == f"hermes-worker-kanban-{tid}-run-{run_id}.scope"
    assert "*" not in " ".join(argv)
    assert "--user" in argv


def test_worker_respawned_while_the_journal_was_read_is_left_alone(kanban_home, journal):
    """The sweep books only the worker it judged dead; a respawn in between keeps its claim."""
    journal.stdout = _OOM_LINES
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker(conn, tid, 70006, rc=0)

        def respawn():
            conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (70007, tid))
            conn.commit()

        journal.on_call = respawn
        assert kbd.detect_crashed_workers(conn) == []

        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.worker_pid == 70007


def test_probe_survives_a_missing_journalctl(monkeypatch):
    from hermes_cli import kanban_db_worker_scopes as scopes

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("journalctl")

    monkeypatch.setattr(subprocess, "run", missing)
    assert scopes.worker_scope_oom_killed("t_00000001", 3) is False

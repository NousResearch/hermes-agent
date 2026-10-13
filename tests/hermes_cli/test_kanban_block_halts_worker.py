"""Invariant: a block filed by anyone other than the worker halts that worker now.

``block_task`` closes the run in one txn and used to leave the OS process alone;
only ``reap_terminal_workers`` ended it, and only once the run had been closed
for ``TERMINAL_WORKER_REAP_GRACE_SECONDS`` — a grace that exists for a worker
finishing its OWN terminal call. A human pressing Stop (Linear's agent contract:
"must not perform any further actions") therefore left the worker calling tools
for two to three minutes. An external block now runs the reaper's per-run
routine at once, with the same host check and spawn-fingerprint proof; a
worker's own block keeps the grace, and an unproven identity is never signalled.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
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
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


@pytest.fixture
def worker():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    yield proc
    if proc.poll() is None:
        proc.kill()
        proc.wait()


@pytest.fixture
def signals(monkeypatch):
    """Every non-probe signal sent through ``os.kill``; probes (sig 0) and real kills pass through."""
    sent: list[tuple[int, int]] = []
    real_kill = os.kill

    def recording_kill(pid, sig):
        if sig:
            sent.append((pid, sig))
        real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", recording_kill)
    return sent


def _running_card(conn, pid: int) -> tuple[str, int]:
    tid = kb.create_task(conn, title="in flight", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    return tid, run_id


def _run(conn, run_id: int):
    return conn.execute(
        "SELECT worker_pid, worker_started_at, ended_at FROM task_runs WHERE id = ?", (run_id,),
    ).fetchone()


def _event_kinds(conn, tid: str) -> list[str]:
    return [r["kind"] for r in conn.execute("SELECT kind FROM task_events WHERE task_id = ?", (tid,))]


def test_external_block_terminates_the_verified_worker_before_returning(conn, worker):
    tid, run_id = _running_card(conn, worker.pid)

    # The Linear plugin's Stop: a supervisor block that proves no run ownership.
    assert kb.block_task(conn, tid, reason="linear: stopped by Pat", kind="needs_input") is True

    # Gone by the time block_task returns: no 120 s grace, no dispatcher tick.
    assert worker.wait(timeout=10) == -signal.SIGTERM
    assert kb.get_task(conn, tid).status == "blocked"
    run = _run(conn, run_id)
    assert run["ended_at"] is not None
    assert run["worker_pid"] is None and run["worker_started_at"] is None
    assert "terminal_worker_reaped" in _event_kinds(conn, tid)


def test_worker_self_block_keeps_the_finalization_grace(conn, worker, signals):
    tid, run_id = _running_card(conn, worker.pid)

    assert kb.block_task(conn, tid, reason="need creds", kind="needs_input", expected_run_id=run_id) is True

    assert signals == [] and worker.poll() is None
    assert _run(conn, run_id)["worker_pid"] == worker.pid  # evidence kept for the reaper
    assert kbd.reap_terminal_workers(conn) == []  # still inside the grace window
    assert "terminal_worker_reaped" not in _event_kinds(conn, tid)


def _unverified(conn, run_id):
    conn.execute("UPDATE task_runs SET worker_started_at = ? WHERE id = ?",
                 (kbd.UNVERIFIED_WORKER_FINGERPRINT, run_id))


def _legacy(conn, run_id):
    conn.execute("UPDATE task_runs SET worker_started_at = NULL WHERE id = ?", (run_id,))


def _other_host(conn, run_id):
    conn.execute("UPDATE task_runs SET claim_lock = 'some-other-host:1:abc' WHERE id = ?", (run_id,))


def _recycled(conn, run_id):
    conn.execute("UPDATE task_runs SET worker_started_at = 'stale-epoch|1' WHERE id = ?", (run_id,))


@pytest.mark.parametrize("unproven", [_unverified, _legacy, _other_host, _recycled],
                         ids=["unverified-fingerprint", "legacy-no-fingerprint", "other-host", "recycled-pid"])
def test_external_block_never_signals_an_unproven_identity(conn, worker, signals, unproven):
    tid, run_id = _running_card(conn, worker.pid)
    unproven(conn, run_id)

    assert kb.block_task(conn, tid, reason="linear: stopped by Pat", kind="needs_input") is True

    assert kb.get_task(conn, tid).status == "blocked"
    assert [s for pid, s in signals if pid == worker.pid] == []
    assert worker.poll() is None


def test_external_block_never_signals_the_calling_process(conn, signals):
    """A worker whose own block reaches block_task without run proof must not kill itself."""
    tid, _run_id = _running_card(conn, os.getpid())

    assert kb.block_task(conn, tid, reason="goal judge: unachievable") is True

    assert [s for pid, s in signals if pid == os.getpid()] == []

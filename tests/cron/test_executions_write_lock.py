"""Ledger writes retry SQLite contention and serialize behind one advisory lock.

A burst of same-instant cron workers used to make one writer lose the SQLite busy race and
raise ``database is locked`` straight out of ``adopt_claimed_execution()``, which killed the
worker. These tests pin the two halves of the fix: the cross-process mutex (a loser waits
instead of raising) and the bounded retry (a transient lock that outlasts the busy timeout is
retried, while a real error still surfaces on the first raise).
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from cron import executions


def test_is_busy_matches_only_contention() -> None:
    assert executions._is_busy(sqlite3.OperationalError("database is locked"))
    assert executions._is_busy(sqlite3.OperationalError("database table is locked: executions"))
    assert not executions._is_busy(sqlite3.OperationalError("no such table: executions"))
    assert not executions._is_busy(sqlite3.IntegrityError("UNIQUE constraint failed"))


def test_write_retries_transient_contention(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    monkeypatch.setattr(executions.time, "sleep", lambda _s: None)
    calls = []

    @executions._ledger_write
    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise sqlite3.OperationalError("database is locked")
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 3, "two contentions should be retried, not surfaced"


def test_write_does_not_retry_real_errors(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    monkeypatch.setattr(executions.time, "sleep", lambda _s: None)
    calls = []

    @executions._ledger_write
    def broken() -> None:
        calls.append(1)
        raise sqlite3.OperationalError("no such table: executions")

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        broken()
    assert len(calls) == 1, "a genuine error must stay loud instead of being masked as contention"


def test_write_gives_up_after_budget(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    monkeypatch.setattr(executions.time, "sleep", lambda _s: None)
    calls = []

    @executions._ledger_write
    def always_busy() -> None:
        calls.append(1)
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        always_busy()
    assert len(calls) == executions.LEDGER_BUSY_RETRIES + 1


def test_write_lock_serializes_processes(monkeypatch, tmp_path) -> None:
    """The flock is what stops two *processes* from writing at once (threads are covered by _lock)."""
    db = tmp_path / "executions.db"
    log = tmp_path / "order.log"
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", db)

    repo_root = str(Path(__file__).resolve().parents[2])
    code = (
        "import os, sys, time\n"
        "sys.path.insert(0, os.environ['REPO'])\n"
        "from pathlib import Path\n"
        "from cron import executions\n"
        "executions.EXECUTIONS_FILE = Path(os.environ['DB'])\n"
        "log = Path(os.environ['LOG'])\n"
        "with executions._ledger_write_lock():\n"
        "    log.open('a').write('in %d\\n' % os.getpid())\n"
        "    time.sleep(0.2)\n"
        "    log.open('a').write('out %d\\n' % os.getpid())\n"
    )
    env = {**os.environ, "REPO": repo_root, "DB": str(db), "LOG": str(log)}
    procs = [subprocess.Popen([sys.executable, "-c", code], env=env) for _ in range(3)]
    for p in procs:
        assert p.wait(timeout=60) == 0

    events = [line.split()[0] for line in log.read_text().splitlines() if line.strip()]
    assert len(events) == 6, events
    for first, second in zip(events, events[1:]):
        assert first != second, "a process entered the lock while another still held it: %s" % events


def test_write_lock_is_reentrant(monkeypatch, tmp_path) -> None:
    """A nested write on the same thread must not deadlock (recover_interrupted_executions does this)."""
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")

    with executions._ledger_write_lock():
        with executions._ledger_write_lock():
            pass

"""``enforce_max_runtime`` must not count host suspend against a worker.

``task_runs.started_at`` is wall clock, and wall clock advances across an
S3/s2idle suspend. A run that spanned a sleeping host therefore looked overrun
on resume and was SIGTERM'd for time its worker never got to use -- and that
kill is booked as a failure against the respawn breaker.

Each run now records the host's accumulated suspend at start
(``CLOCK_BOOTTIME`` minus ``CLOCK_MONOTONIC``). That value is
process-independent, so the baseline stays comparable after a gateway restart
within the same boot, which a monotonic timestamp would not.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, *, limit: int, ran_for: int, suspend_base: Optional[int]) -> str:
    """A claimed, running task whose active run started ``ran_for`` seconds ago."""
    tid = kb.create_task(
        conn, title="spanned a suspend", assignee="worker", max_runtime_seconds=limit,
    )
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    started = int(time.time()) - ran_for
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (started, tid))
        conn.execute(
            "UPDATE task_runs SET started_at = ?, suspend_base_seconds = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (started, suspend_base, tid),
        )
    return tid


def _enforce(conn, monkeypatch, *, suspended_now: Optional[int]) -> tuple[list, list]:
    """Run the sweep with a stubbed suspend clock; returns (timed_out, signals)."""
    signals: list = []
    monkeypatch.setattr(kb, "_suspended_seconds", lambda: suspended_now)
    # The grace poll would otherwise wait on a pid that never dies here.
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    timed_out = kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: signals.append((pid, sig)))
    return timed_out, signals


def test_suspend_is_discounted_from_max_runtime(kanban_home, monkeypatch):
    """295s of the 300s wall elapsed was host sleep: 5s of real runtime is under
    the 10s cap, so the worker must survive."""
    conn = kbc.connect()
    try:
        tid = _running_task(conn, limit=10, ran_for=300, suspend_base=1_000)
        timed_out, signals = _enforce(conn, monkeypatch, suspended_now=1_000 + 295)

        assert timed_out == []
        assert signals == [], "a worker that only slept must not be signalled"
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


def test_runtime_beyond_the_suspend_still_times_out(kanban_home, monkeypatch):
    """The discount is not a blanket reprieve: 300s wall minus 100s sleep is
    200s of real runtime, still far past the 10s cap."""
    conn = kbc.connect()
    try:
        tid = _running_task(conn, limit=10, ran_for=300, suspend_base=1_000)
        timed_out, signals = _enforce(conn, monkeypatch, suspended_now=1_000 + 100)

        assert timed_out == [tid]
        assert signals and signals[0][0] == os.getpid()
        assert kb.get_task(conn, tid).status == "ready"
    finally:
        conn.close()


def test_without_a_baseline_wall_clock_behaviour_is_unchanged(kanban_home, monkeypatch):
    """Same 300s run as the first test, but no usable suspend clock (legacy row,
    or a platform without CLOCK_BOOTTIME): the old raw-wall-elapsed kill stands.
    This is the A/B that isolates the discount as the only difference."""
    conn = kbc.connect()
    try:
        tid = _running_task(conn, limit=10, ran_for=300, suspend_base=1_000)
        timed_out, _ = _enforce(conn, monkeypatch, suspended_now=None)

        assert timed_out == [tid]
    finally:
        conn.close()


def test_reboot_does_not_extend_the_deadline(kanban_home, monkeypatch):
    """Both clocks restart at boot, so the live counter can sit below a stale
    baseline. That must clamp to no discount, never widen the cap."""
    conn = kbc.connect()
    try:
        tid = _running_task(conn, limit=10, ran_for=300, suspend_base=5_000)
        timed_out, _ = _enforce(conn, monkeypatch, suspended_now=10)

        assert timed_out == [tid]
    finally:
        conn.close()

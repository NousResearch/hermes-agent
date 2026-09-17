"""Tests: the crash sweep reclaims host-local ``running`` rows that never
recorded a worker PID (regression for #113610).

A spawn that reports no PID (``_set_worker_pid`` never called) leaves
``tasks.status='running'`` with ``worker_pid IS NULL``. The reaper used to
sweep only ``worker_pid IS NOT NULL`` rows, so those rows stayed ``running``
forever — ``detect_stale_running`` is off by default and nothing else closed
them. The sweep now treats a NULL-pid running row as dead once the
launch-window grace has passed.
"""

from __future__ import annotations

from pathlib import Path

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def _kanban_fixtures(tmp_path, monkeypatch, grace):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    if grace is not None:
        monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", grace)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


import pytest


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    _kanban_fixtures(tmp_path, monkeypatch, "0")
    with kbc.connect() as c:
        yield c


@pytest.fixture
def kanban_home_grace(tmp_path, monkeypatch):
    _kanban_fixtures(tmp_path, monkeypatch, None)
    with kbc.connect() as c:
        yield c


def test_null_pid_running_row_is_reclaimed(kanban_home):
    """A running row with no worker PID must be closed by the crash sweep —
    it is dead by definition (nothing to probe for liveness)."""
    conn = kanban_home
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="no-pid", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    # Spawn reported no PID: worker_pid stays NULL.
    row = conn.execute("SELECT worker_pid FROM tasks WHERE id=?", (tid,)).fetchone()
    assert row["worker_pid"] is None

    crashed = kbd.detect_crashed_workers(conn)

    assert tid in crashed
    final = conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
    assert final["status"] != "running"
    run = conn.execute(
        "SELECT status FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()
    assert run["status"] == "crashed"


def test_fresh_null_pid_claim_survives_grace(kanban_home_grace):
    """The launch-window grace still protects a just-claimed NULL-pid row —
    another dispatcher may be between ``claim_task`` and ``_set_worker_pid``."""
    conn = kanban_home_grace
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="fresh-no-pid", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")

    crashed = kbd.detect_crashed_workers(conn)

    assert tid not in crashed
    final = conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
    assert final["status"] == "running"

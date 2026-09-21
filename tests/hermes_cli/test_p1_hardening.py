"""P1 hardening tests (2026-06-12 kanban/orchestration review).

Covers:
  D-1  dispatcher runaway regression; live worker not double-spawned
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# --------------------------------------------------------------------------
# D-1: dispatcher runaway regression test
# --------------------------------------------------------------------------

def test_live_worker_not_reclaimed_prevents_duplicate_spawn(kanban_home, monkeypatch):
    """A running task whose worker PID is alive and survived termination
    must NOT be reclaimed to 'ready'.  Reclaiming would release the claim
    and let the dispatcher spawn a second worker beside the first; the
    duplication loop that D-1 prevents."""
    now = int(time.time())
    claim_lock = "host:test:12345"
    worker_pid = 99999

    with kb.connect() as conn:
        # Create a running task with an expired claim
        conn.execute(
            "INSERT INTO tasks (id, title, status, claim_lock, claim_expires, worker_pid, created_at) "
            "VALUES (?, ?, 'running', ?, ?, ?, ?)",
            ("d1-test-001", "D-1 runaway test", claim_lock, now - 60, worker_pid, now),
        )
        conn.commit()

    # Mock: PID is alive
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: True)
    # Mock: termination was attempted but worker survived
    monkeypatch.setattr(
        kb, "_terminate_reclaimed_worker",
        lambda pid, lock, signal_fn=None: {
            "termination_attempted": True,
            "host_local": True,
            "terminated": False,
        },
    )

    with kb.connect() as conn:
        reclaimed = kb.release_stale_claims(conn)
        # Must be 0; the live worker blocked reclaim
        assert reclaimed == 0

        # Task must still be 'running', not 'ready'
        row = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", ("d1-test-001",)
        ).fetchone()
        assert row is not None
        assert row["status"] == "running"
        assert row["claim_lock"] == claim_lock

        # A reclaim_deferred event must exist
        event = conn.execute(
            "SELECT 1 FROM task_events "
            "WHERE task_id = ? AND kind = 'reclaim_deferred'",
            ("d1-test-001",),
        ).fetchone()
        assert event is not None, "reclaim_deferred event missing; defer path not exercised"


def test_dead_worker_still_reclaimed(kanban_home, monkeypatch):
    """A worker whose PID is dead (terminated) must still be reclaimed
    normally.  The D-1 guard only blocks reclaim for live workers."""
    now = int(time.time())
    claim_lock = "host:test:67890"
    worker_pid = 88888

    with kb.connect() as conn:
        conn.execute(
            "INSERT INTO tasks (id, title, status, claim_lock, claim_expires, worker_pid, created_at) "
            "VALUES (?, ?, 'running', ?, ?, ?, ?)",
            ("d1-test-002", "D-1 dead worker", claim_lock, now - 60, worker_pid, now),
        )
        conn.commit()

    # Mock: PID is dead
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    # Mock: termination succeeded
    monkeypatch.setattr(
        kb, "_terminate_reclaimed_worker",
        lambda pid, lock, signal_fn=None: {
            "termination_attempted": True,
            "host_local": True,
            "terminated": True,
        },
    )

    with kb.connect() as conn:
        reclaimed = kb.release_stale_claims(conn)
        assert reclaimed == 1

        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", ("d1-test-002",)
        ).fetchone()
        assert row is not None
        assert row["status"] == "ready"

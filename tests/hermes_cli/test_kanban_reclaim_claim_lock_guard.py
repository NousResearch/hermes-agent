"""Tests: reclaim paths are claim-lock-aware so they can't desync a re-claimed
task (issue #36910).

A stale crash/stale-claim/max-runtime reclaim, computed from a snapshot of an
OLD worker, used to reset ``tasks.status`` back to ``ready`` with only a
``WHERE status='running'`` guard. If the task had since been reclaimed AND
re-claimed by a NEW worker (new run, new claim_lock, live pid), that stale
UPDATE clobbered the live task: ``tasks.status='ready'`` while the new
``task_runs.status='running'`` and the worker kept executing — the board showed
the task in the Ready lane and the dispatcher could treat live work as
available. The reset is now gated on the snapshot's ``claim_lock`` (and pid),
so it only fires when the task is still owned by the worker the reclaim was
computed for.
"""

from __future__ import annotations

import json
import signal
import subprocess
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


def test_stale_crash_reset_rejected_for_reclaimed_task(conn):
    """A reset carrying an OLD worker's claim_lock must NOT clobber a task
    that has since been re-claimed by a new worker."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="desync", assignee="w")

    # Worker A claims, then dies.
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kb._set_worker_pid(conn, tid, dead.pid)
    old = conn.execute(
        "SELECT claim_lock, worker_pid FROM tasks WHERE id=?", (tid,)
    ).fetchone()

    # Reclaim + re-claim by worker B (alive).
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, current_run_id=NULL WHERE id=?",
        (tid,),
    )
    conn.commit()
    kb.claim_task(conn, tid, claimer=f"{host}:B")
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        kb._set_worker_pid(conn, tid, sleeper.pid)

        # The stale reset for worker A — same shape as the guarded UPDATE in
        # detect_crashed_workers — must reject (rowcount 0) because B owns it.
        cur = conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, "
            "claim_expires=NULL, worker_pid=NULL "
            "WHERE id=? AND status='running' AND worker_pid=? AND claim_lock IS ?",
            (tid, old["worker_pid"], old["claim_lock"]),
        )
        conn.commit()
        assert cur.rowcount == 0, "stale reclaim wrongly clobbered the re-claimed task"

        final = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        assert final["status"] == "running"
        assert final["claim_lock"] == f"{host}:B"
    finally:
        sleeper.terminate()


def test_genuine_crash_still_reclaims(conn):
    """When the claim_lock still matches the dead worker, the crash reclaim
    fires normally — the guard must not break the legitimate path."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="legit", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kb._set_worker_pid(conn, tid, dead.pid)
    # Rewind started_at so the launch grace window doesn't skip the check.
    conn.execute("UPDATE tasks SET started_at = started_at - 9999 WHERE id=?", (tid,))
    conn.execute(
        "UPDATE task_runs SET started_at = started_at - 9999 WHERE task_id=?", (tid,)
    )
    conn.commit()
    kb._record_worker_exit(dead.pid, 1 << 8)  # nonzero exit → crash

    crashed = kb.detect_crashed_workers(conn)
    assert tid in crashed
    final = conn.execute("SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
    assert final["status"] in ("ready", "blocked", "todo")


def test_manual_reclaim_defers_when_worker_survives_termination(conn, monkeypatch):
    """Manual reclaim must not release a claim held by a live worker."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="live manual reclaim", assignee="w")
    claim_lock = f"{host}:manual-reclaim"
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
        "worker_pid=? WHERE id=?",
        (claim_lock, now - 1, 12345, tid),
    )
    conn.execute(
        "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
        "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
        (tid, claim_lock, now - 1, 12345, now),
    )
    run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, tid))
    conn.commit()

    signals = []
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(kb.time, "sleep", lambda _seconds: None)

    result = kb.reclaim_task(
        conn,
        tid,
        reason="operator retry",
        signal_fn=lambda pid, sig: signals.append((pid, sig)),
    )

    row = conn.execute(
        "SELECT status, claim_lock, worker_pid, claim_expires "
        "FROM tasks WHERE id=?",
        (tid,),
    ).fetchone()
    events = [
        (event["kind"], json.loads(event["payload"]))
        for event in conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id",
            (tid,),
        )
    ]

    assert result is False
    assert row["status"] == "running"
    assert row["claim_lock"] == claim_lock
    assert row["worker_pid"] == 12345
    assert row["claim_expires"] >= now + kb.RECLAIM_DEFER_GRACE_SECONDS
    assert [kind for kind, _payload in events].count("reclaim_deferred") == 1
    assert not any(kind == "reclaimed" for kind, _payload in events)
    deferred = next(payload for kind, payload in events if kind == "reclaim_deferred")
    assert deferred["reason"] == "manual_reclaim_worker_alive"
    assert deferred["termination_attempted"] is True
    assert deferred["terminated"] is False
    assert signals == [(12345, signal.SIGTERM), (12345, signal.SIGKILL)]


def test_manual_reclaim_defers_foreign_live_pid_without_identity_proof(
    conn, monkeypatch,
):
    """A hostname change must not make an arbitrary live PID killable."""
    tid = kb.create_task(conn, title="foreign live worker", assignee="w")
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
        "worker_pid=? WHERE id=?",
        ("old-host:claim", now - 1, 24680, tid),
    )
    conn.execute(
        "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
        "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
        (tid, "old-host:claim", now - 1, 24680, now),
    )
    run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, tid))
    conn.commit()

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    signals = []
    result = kb.reclaim_task(
        conn,
        tid,
        signal_fn=lambda pid, sig: signals.append((pid, sig)),
    )

    row = conn.execute(
        "SELECT status, claim_lock, worker_pid FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    deferred = [
        row for row in kb.list_events(conn, tid)
        if row.kind == "reclaim_deferred"
    ]
    assert result is False
    assert row["status"] == "running"
    assert row["claim_lock"] == "old-host:claim"
    assert row["worker_pid"] == 24680
    assert signals == []
    assert len(deferred) == 1
    assert deferred[0].payload["reason"] == (
        "manual_reclaim_worker_identity_unverified"
    )


def test_manual_reclaim_force_local_requires_exact_worker_identity(
    conn, monkeypatch,
):
    """The explicit hostname override only permits an exact worker match."""
    tid = kb.create_task(conn, title="exact worker", assignee="w")
    now = int(time.time())
    claim_lock = "old-host:exact"
    conn.execute(
        "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
        "worker_pid=? WHERE id=?",
        (claim_lock, now - 1, 13579, tid),
    )
    conn.execute(
        "INSERT INTO task_runs (task_id, status, claim_lock, claim_expires, "
        "worker_pid, started_at) VALUES (?, 'running', ?, ?, ?, ?)",
        (tid, claim_lock, now - 1, 13579, now),
    )
    run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, tid))
    conn.commit()

    alive = {"value": True}
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: alive["value"])
    monkeypatch.setattr(kb, "_pid_looks_like_hermes_worker", lambda _pid: True)
    monkeypatch.setattr(
        kb,
        "_read_process_environ",
        lambda _pid: {
            "HERMES_KANBAN_TASK": tid,
            "HERMES_KANBAN_RUN_ID": str(run_id),
            "HERMES_KANBAN_CLAIM_LOCK": claim_lock,
        },
    )
    monkeypatch.setattr(kb.time, "sleep", lambda _seconds: None)
    signals = []

    def signal(pid, sig):
        signals.append((pid, sig))
        if sig == signal_module.SIGKILL:
            alive["value"] = False

    import signal as signal_module

    assert kb.reclaim_task(
        conn, tid, force_local=True, signal_fn=signal,
    ) is True
    row = conn.execute(
        "SELECT status, claim_lock, worker_pid FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    assert row["status"] in {"ready", "todo", "review"}
    assert row["claim_lock"] is None
    assert row["worker_pid"] is None
    assert signals == [(13579, signal_module.SIGTERM), (13579, signal_module.SIGKILL)]

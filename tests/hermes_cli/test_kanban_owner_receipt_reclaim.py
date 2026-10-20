"""A dead attempt behind an unresolved owner receipt must not hold its claim forever.

Regression for #140858. Card t_5ea03658 sat ``running`` for 16 hours: claimed 07:48, worker
killed by a restart at 08:59, claim expired 08:03, no heartbeat, and no ``worker_bound``
event. ``owner_reclaim_paused`` was true — a killed owner leaves a ``terminal/interrupted``
admission, or a pid that is gone — and every reclaim path read that as "leave it alone", so
``release_stale_claims``, ``detect_crashed_workers`` and ``reconcile_orphaned_running`` all
declined. ``bound_interpreter_gone``, the one override, needs the ``worker_bound`` event that
run never wrote. The card held its profile's ``max_in_progress_per_profile`` slot while every
card queued behind it was deferred, and no diagnostics rule could see a ``running`` card at
all, so the alarms named the victims and never the cause.

The pause is about effects that may still be in flight. These tests pin both sides of that
line: a live worker, a fresh heartbeat, a live claim, a foreign host's lock or an identity
that cannot be read all keep the card; a provably dead attempt is reclaimed by the platform's
own path.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_diagnostics as kd
from hermes_cli import kanban_owner_recovery
from hermes_cli.process_identity import process_birth
from hermes_cli.kanban_owner_recovery import (
    bound_interpreter_gone,
    owner_attempt_provably_over,
    owner_reclaim_blocked,
    owner_reclaim_paused,
)

NOW = int(time.time())
HOST = kb._claimer_id().split(":", 1)[0]
FINGERPRINT = "6a232414-2dbf-40ab-a534-d9b71db05658:116|4162240"


def _absent_pid(base: int = 3_500_000) -> int:
    """A pid that provably does not exist on this host, so a fixture can never hit a process."""
    pid = base
    while psutil.pid_exists(pid):
        pid += 1
    return pid


class _Board:
    """A scratch board plus the scratch admission ledger an owner receipt points at."""

    def __init__(self, root: Path):
        self.path = root / "kanban.db"
        kbc.init_db(self.path)
        self.conn = kbc.connect(self.path)
        self.ledger = root / "state.db"
        with sqlite3.connect(self.ledger) as store:
            store.execute(
                "CREATE TABLE session_admissions "
                "(target_session_id TEXT, request_id TEXT, status TEXT, outcome TEXT)"
            )

    def close(self) -> None:
        self.conn.close()

    # --- seeding --------------------------------------------------------------------------
    def dead_attempt(
        self, tid: str = "t_ghost", *, worker_pid=None, worker_fingerprint=FINGERPRINT,
        worker_bound: bool = False, record_worker_pid: bool = True, owner_pid=None,
        owner_birth=None, admission=("terminal", "interrupted"), dead_hours: int = 15,
        expires_age_seconds=None, lock_host=None, heartbeat_age=None,
    ) -> int:
        """A ``running`` card whose worker died the way t_5ea03658's did."""
        worker_pid = _absent_pid() if worker_pid is None else worker_pid
        owner_pid = _absent_pid(base=3_600_000) if owner_pid is None else owner_pid
        lock = f"{lock_host or HOST}:{owner_pid}"
        expires = NOW - (expires_age_seconds if expires_age_seconds is not None
                         else dead_hours * 3600)
        started = NOW - dead_hours * 3600
        run_id = self.conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'default', 'running', ?, ?, ?, ?)",
            (tid, lock, expires, worker_pid if record_worker_pid else None, started),
        ).lastrowid
        assert run_id is not None
        run = int(run_id)
        self.conn.execute(
            "INSERT INTO tasks (id, title, assignee, status, created_at, claim_lock, claim_expires, "
            "worker_pid, worker_started_at, last_heartbeat_at, current_run_id, started_at, priority) "
            "VALUES (?, ?, 'default', 'running', ?, ?, ?, ?, ?, ?, ?, ?, 5)",
            (tid, tid, NOW - 86400, lock, expires,
             worker_pid if record_worker_pid else None,
             worker_fingerprint if record_worker_pid else None,
             (NOW - heartbeat_age) if heartbeat_age is not None else None, run, started),
        )
        self.event(tid, run, "claimed", {"source_status": "ready"}, started)
        if record_worker_pid:
            spawned = {"pid": worker_pid, "started_at": worker_fingerprint}
        else:
            spawned = {}
        self.event(tid, run, "spawned", spawned, started)
        payload = {"db": str(self.ledger), "session_id": "sess-1", "pid": owner_pid,
                   "request_id": f'kanban:["default","{tid}",1]'}
        if owner_birth is not None:
            payload["birth"] = owner_birth
        self.event(tid, run, "owner_admitted", payload, started + 30)
        if worker_bound:
            self.event(tid, run, "worker_bound",
                       {"pid": worker_pid, "claim_lock": lock, "started_at": worker_fingerprint},
                       started + 60)
        if admission:
            with sqlite3.connect(self.ledger) as store:
                store.execute("INSERT INTO session_admissions VALUES ('sess-1', ?, ?, ?)",
                              (payload["request_id"], admission[0], admission[1]))
        self.conn.commit()
        return run

    def event(self, tid: str, run: int, kind: str, payload: dict, ts: int) -> None:
        self.conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)", (tid, run, kind, json.dumps(payload), ts))

    def ready(self, tid: str) -> None:
        self.conn.execute(
            "INSERT INTO tasks (id, title, assignee, status, created_at, priority) "
            "VALUES (?, ?, 'default', 'ready', ?, 5)", (tid, tid, NOW - 86400))
        self.conn.commit()

    # --- reading --------------------------------------------------------------------------
    def state(self, tid: str) -> dict:
        return dict(self.conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())

    def events(self, tid: str) -> list:
        return self.conn.execute(
            "SELECT kind, payload, created_at FROM task_events WHERE task_id = ? ORDER BY id",
            (tid,)).fetchall()

    def diagnostics(self, tid: str, *, config: dict | None = None) -> list:
        return kd.compute_task_diagnostics(
            self.state(tid), self.events(tid), [], now=NOW, config=config or {})


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    scratch = _Board(tmp_path)
    yield scratch
    scratch.close()


# --- the incident shape is named by diagnostics and reclaimed by the platform's own paths --

def test_dead_attempt_behind_an_unresolved_receipt_is_named_and_reclaimed(board):
    """Every reclaim path declined this card and no diagnostic could see it; both must now act."""
    board.dead_attempt("t_ghost")
    board.dead_attempt("t_ghost2")
    board.dead_attempt("t_live", worker_pid=os.getpid(), worker_fingerprint=kbd._process_fingerprint(os.getpid()))
    assert owner_reclaim_paused(board.conn, "t_ghost") is True      # the pause still applies
    assert bound_interpreter_gone(board.conn, "t_ghost") is False   # nothing bound to read
    assert owner_reclaim_blocked(board.conn, "t_ghost") is False

    hits = [d for d in board.diagnostics("t_ghost") if d.kind == "stale_running"]
    assert [d.severity for d in hits] == ["critical"]                  # 15x the 1h grace
    assert [a.kind for a in hits[0].actions] == ["reclaim", "cli_hint"]
    assert not [d for d in board.diagnostics("t_live") if d.kind == "stale_running"]

    assert kb.release_stale_claims(board.conn) == 2
    row = board.state("t_ghost")
    assert (row["status"], row["claim_lock"], row["worker_pid"], row["current_run_id"]) == ("ready", None, None, None)
    run = board.conn.execute("SELECT status, outcome FROM task_runs WHERE task_id = 't_ghost'").fetchone()
    assert tuple(run) == ("reclaimed", "reclaimed")
    assert board.state("t_live")["status"] == "running"


@pytest.mark.parametrize("shape", [
    dict(worker_pid=os.getpid(), worker_fingerprint=kbd._process_fingerprint(os.getpid())),  # live worker
    dict(worker_bound=True, worker_pid=os.getpid(),
         worker_fingerprint=kbd._process_fingerprint(os.getpid())),                          # live bound interpreter
    dict(heartbeat_age=300),                                                                 # fresh heartbeat
    dict(expires_age_seconds=-3600),                                                         # live claim
    dict(lock_host="some-other-host"),                                                       # foreign host
    dict(owner_pid=os.getpid(), owner_birth=None),                                           # live, uncomparable owner
    dict(record_worker_pid=False),                                                           # no pid recorded
], ids=["live-worker", "live-bound", "fresh-heartbeat", "live-claim", "foreign-host", "live-owner", "no-pid"])
def test_the_pause_still_holds_every_attempt_that_may_be_alive(board, shape):
    """Missing evidence is never proof: reclaiming beside a possibly-live attempt repeats its effects."""
    board.dead_attempt("t_ghost", **shape)
    assert owner_attempt_provably_over(board.conn, "t_ghost") is False
    assert kb.release_stale_claims(board.conn) == 0
    assert kbd.detect_crashed_workers(board.conn) == []
    assert board.state("t_ghost")["status"] == "running"


# --- the gate is decided under the same write lock as the release ---------------------------

_RECLAIM_PATHS = {
    "ttl": lambda conn: kb.release_stale_claims(conn),
    "stale": lambda conn: kbd.detect_stale_running(conn, stale_timeout_seconds=1, signal_fn=lambda *_: None),
    "runtime": lambda conn: kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None),
}


@pytest.mark.parametrize("path", sorted(_RECLAIM_PATHS))
def test_an_interpreter_bound_after_the_gate_keeps_its_claim(board, monkeypatch, path):
    """Binding changes neither claim_lock nor worker_pid, so only a re-check inside the
    reclaim transaction can see an owner that recovered after the first look."""
    run = board.dead_attempt("t_ghost")
    board.conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = 't_ghost'")
    board.conn.commit()
    real_gate, calls = owner_reclaim_blocked, []

    def gate_then_bind(conn, task_id, **kwargs):
        verdict = real_gate(conn, task_id, **kwargs)
        if not calls:  # the recovered owner binds a live interpreter right after the first look
            board.event(task_id, run, "worker_bound",
                        {"pid": os.getpid(), "started_at": kbd._process_fingerprint(os.getpid())}, NOW)
            board.conn.commit()
        calls.append(verdict)
        return verdict

    monkeypatch.setattr(kanban_owner_recovery, "owner_reclaim_blocked", gate_then_bind)
    assert not _RECLAIM_PATHS[path](board.conn)
    assert calls[0] is False and calls[-1] is True
    row = board.state("t_ghost")
    assert (row["status"], row["current_run_id"]) == ("running", run)


@pytest.mark.platforms("linux", "macos")
def test_an_exited_unreaped_owner_no_longer_holds_the_slot(board):
    """A zombie keeps its birth until reaped, but it has exited: it cannot be executing anything."""
    child = subprocess.Popen([sys.executable, "-c", "input()"], stdin=subprocess.PIPE)
    try:
        birth = process_birth(psutil.Process(child.pid))
        child.stdin.close()
        deadline = time.time() + 10
        while psutil.Process(child.pid).status() != psutil.STATUS_ZOMBIE:
            assert time.time() < deadline, "child never exited"
            time.sleep(0.02)
        board.dead_attempt("t_ghost", owner_pid=child.pid, owner_birth=birth)
        assert kb.release_stale_claims(board.conn) == 1
        assert board.state("t_ghost")["status"] == "ready"
    finally:
        child.wait()


def test_an_earlier_runs_live_pid_does_not_hide_the_current_dead_run(board):
    """Only the current attempt's pids speak for it in the stale_running diagnostic."""
    run = board.dead_attempt("t_ghost")
    board.event("t_ghost", run - 1, "spawned", {"pid": os.getpid()}, NOW - 100_000)
    board.conn.commit()
    assert [d.kind for d in board.diagnostics("t_ghost") if d.kind == "stale_running"] == ["stale_running"]

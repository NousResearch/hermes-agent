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
import time
from pathlib import Path

import psutil
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_diagnostics as kd
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


# --- the incident shape is reclaimed by the platform's own paths ---------------------------

def test_dead_attempt_behind_an_unresolved_receipt_is_reclaimed(board):
    """Every reclaim path declined this card; the stale-claim path must now take it."""
    board.dead_attempt("t_ghost")
    assert owner_reclaim_paused(board.conn, "t_ghost") is True      # the pause still applies
    assert bound_interpreter_gone(board.conn, "t_ghost") is False   # nothing bound to read
    assert owner_attempt_provably_over(board.conn, "t_ghost") is True
    assert owner_reclaim_blocked(board.conn, "t_ghost") is False

    assert kb.release_stale_claims(board.conn) == 1

    row = board.state("t_ghost")
    assert row["status"] == "ready"
    assert (row["claim_lock"], row["claim_expires"], row["worker_pid"]) == (None, None, None)
    assert row["current_run_id"] is None
    run = board.conn.execute(
        "SELECT status, outcome FROM task_runs WHERE task_id = 't_ghost'").fetchone()
    assert tuple(run) == ("reclaimed", "reclaimed")


def test_dead_attempt_behind_an_unresolved_receipt_is_reclaimed_by_the_crash_sweep(board):
    """The dead-pid path reclaims the same shape instead of deferring to the receipt."""
    board.dead_attempt("t_ghost")
    assert kbd.detect_crashed_workers(board.conn) == ["t_ghost"]
    assert board.state("t_ghost")["status"] == "ready"


def test_reclaiming_the_dead_attempt_frees_its_profile_lane(board):
    """The point of the fix: the card stops holding its profile's worker slot."""
    board.dead_attempt("t_ghost")
    board.ready("t_ready0")
    counted = "SELECT COUNT(*) FROM tasks WHERE status = 'running' AND assignee = 'default'"
    assert board.conn.execute(counted).fetchone()[0] == 1   # the jam: a dead card holds the slot

    kbd.dispatch_once(board.conn, dry_run=True, max_in_progress=2,
                      max_in_progress_per_profile=1)
    assert board.state("t_ghost")["status"] == "ready"
    assert board.conn.execute(counted).fetchone()[0] == 0   # the slot is free again

    # And the next tick spends it instead of deferring the queue forever.
    lane = kbd.dispatch_once(board.conn, dry_run=True, max_in_progress=2,
                             max_in_progress_per_profile=1)
    assert len(lane.spawned) == 1


# --- the pause still holds every attempt that may be alive ---------------------------------

def test_a_live_worker_holds_the_card_even_with_the_pause(board):
    """Real pid, real fingerprint: reclaiming beside it would spawn a duplicate."""
    board.dead_attempt("t_ghost", worker_pid=os.getpid(),
                       worker_fingerprint=kbd._process_fingerprint(os.getpid()))
    assert owner_reclaim_paused(board.conn, "t_ghost") is True
    assert owner_attempt_provably_over(board.conn, "t_ghost") is False
    assert owner_reclaim_blocked(board.conn, "t_ghost") is True
    assert kb.release_stale_claims(board.conn) == 0
    assert board.state("t_ghost")["status"] == "running"
    assert board.state("t_ghost")["worker_pid"] == os.getpid()


def test_a_live_bound_interpreter_holds_the_card(board):
    """A ``worker_bound`` interpreter that is still running is the same refusal."""
    board.dead_attempt("t_ghost", worker_bound=True, worker_pid=os.getpid(),
                       worker_fingerprint=kbd._process_fingerprint(os.getpid()))
    assert bound_interpreter_gone(board.conn, "t_ghost") is False
    assert owner_reclaim_blocked(board.conn, "t_ghost") is True
    assert kb.release_stale_claims(board.conn) == 0
    assert board.state("t_ghost")["status"] == "running"


def test_a_fresh_heartbeat_holds_the_card(board):
    """A slow worker inside one tool-free call keeps its claim (#23025)."""
    board.dead_attempt("t_ghost", heartbeat_age=300)
    assert owner_attempt_provably_over(board.conn, "t_ghost") is False
    assert kb.release_stale_claims(board.conn) == 0
    assert board.state("t_ghost")["status"] == "running"


def test_a_live_claim_is_never_overridden(board):
    """An unexpired claim is the owner still working, whatever the receipt says."""
    board.dead_attempt("t_ghost", expires_age_seconds=-3600)
    assert owner_attempt_provably_over(board.conn, "t_ghost") is False
    assert kb.release_stale_claims(board.conn) == 0
    assert board.state("t_ghost")["status"] == "running"


def test_another_hosts_lock_is_never_overridden(board):
    """A foreign lock's pids say nothing here, and a numeric coincidence is not a death."""
    board.dead_attempt("t_ghost", lock_host="some-other-host")
    assert bound_interpreter_gone(board.conn, "t_ghost") is False
    assert owner_attempt_provably_over(board.conn, "t_ghost") is False
    assert kb.release_stale_claims(board.conn) == 0
    assert board.state("t_ghost")["status"] == "running"


def test_a_live_owner_without_a_birth_is_never_read_as_dead(board):
    """The receipt names a live pid but no identity to compare: unprovable, so it holds."""
    board.dead_attempt("t_ghost", owner_pid=os.getpid(), owner_birth=None)
    assert owner_reclaim_paused(board.conn, "t_ghost") is True
    assert owner_attempt_provably_over(board.conn, "t_ghost") is False
    assert kb.release_stale_claims(board.conn) == 0
    assert board.state("t_ghost")["status"] == "running"


def test_a_recycled_owner_pid_is_dead_evidence(board):
    """A live number whose birth moved on is a stranger, exactly as the pause reads it."""
    board.dead_attempt("t_ghost", owner_pid=os.getpid(), owner_birth=1.0)
    assert owner_reclaim_paused(board.conn, "t_ghost") is True
    assert owner_attempt_provably_over(board.conn, "t_ghost") is True
    assert kb.release_stale_claims(board.conn) == 1


def test_owner_identity_reads_birth_through_the_shared_helper(board, monkeypatch):
    """A live owner's birth must be read the way the pause reads it, or a clock step looks like death.

    ``create_time()`` shifts on macOS after a boot-time change while the kernel's own start does
    not (#140580), so comparing the fence's reading against the receipt's would call a live owner
    recycled and release a claim that is still being worked. The fence defers to
    ``process_identity.process_birth``; this pins that it does, whatever the platform.
    """
    from hermes_cli import process_identity
    seen: list[int] = []

    def fake_birth(proc):
        seen.append(proc.pid)
        return 1234.0

    monkeypatch.setattr(process_identity, "process_birth", fake_birth)
    board.dead_attempt("t_ghost", owner_pid=os.getpid(), owner_birth=1234.0)
    assert owner_reclaim_paused(board.conn, "t_ghost") is True
    assert owner_attempt_provably_over(board.conn, "t_ghost") is False
    assert seen == [os.getpid()]
    assert kb.release_stale_claims(board.conn) == 0
    assert board.state("t_ghost")["status"] == "running"


def test_no_recorded_worker_pid_is_never_proof(board):
    """A run that recorded no worker pid cannot be shown to be dead — it holds."""
    board.dead_attempt("t_ghost", record_worker_pid=False)
    assert owner_attempt_provably_over(board.conn, "t_ghost") is False
    assert kb.release_stale_claims(board.conn) == 0
    assert board.state("t_ghost")["status"] == "running"


# --- the jam itself becomes visible to `kanban diagnostics` --------------------------------

def test_stale_running_diagnostic_names_the_jam(board):
    board.dead_attempt("t_ghost")
    hits = [d for d in board.diagnostics("t_ghost") if d.kind == "stale_running"]
    assert len(hits) == 1
    hit = hits[0]
    assert hit.severity == "critical"       # 15x the 1h grace
    assert hit.data["expired_seconds"] >= 3600
    assert hit.data["recorded_pids"] and str(hit.data["recorded_pids"][0]) in hit.detail
    assert [a.kind for a in hit.actions] == ["reclaim", "cli_hint"]
    assert hit.actions[0].suggested is True


@pytest.mark.parametrize("expired_for,severity", [
    (30 * 60, ""),              # inside the grace: the dispatcher has not had its tick yet
    (3 * 3600, "error"),        # 3x grace
    (7 * 3600, "critical"),     # 7x grace
])
def test_stale_running_diagnostic_escalates_with_age(board, expired_for, severity):
    board.dead_attempt("t_ghost", expires_age_seconds=expired_for)
    hits = [d for d in board.diagnostics("t_ghost") if d.kind == "stale_running"]
    assert [d.severity for d in hits] == ([severity] if severity else [])


def test_stale_running_diagnostic_is_silent_for_a_live_worker(board):
    board.dead_attempt("t_ghost", worker_pid=os.getpid(),
                       worker_fingerprint=kbd._process_fingerprint(os.getpid()))
    assert not [d for d in board.diagnostics("t_ghost") if d.kind == "stale_running"]


def test_stale_running_diagnostic_is_silent_for_a_fresh_heartbeat(board):
    board.dead_attempt("t_ghost", heartbeat_age=300)
    assert not [d for d in board.diagnostics("t_ghost") if d.kind == "stale_running"]


def test_stale_running_diagnostic_is_silent_for_a_live_claim(board):
    board.dead_attempt("t_ghost", expires_age_seconds=-60)
    assert not [d for d in board.diagnostics("t_ghost") if d.kind == "stale_running"]


def test_stale_running_diagnostic_is_silent_for_another_hosts_lock(board):
    board.dead_attempt("t_ghost", lock_host="some-other-host")
    assert not [d for d in board.diagnostics("t_ghost") if d.kind == "stale_running"]


def test_stale_running_diagnostic_is_silent_for_a_ready_card(board):
    board.ready("t_ready0")
    assert not [d for d in board.diagnostics("t_ready0") if d.kind == "stale_running"]

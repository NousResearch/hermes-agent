"""A lost managed worker must be DISPOSED OF on the board, within one dispatcher tick.

Regression for the 2026-10-22 stall (card ``t_f5fad63f``, run 730). The fan gateway's managed
worker died 24 s into the attempt, inside the ``worker.adopt`` handshake
(``TimeoutError: timed out while waiting for handshake response``, ``open_timeout=5``). The
gateway logged ``Managed worker lost: managed_worker_failed``, settled the admission
``terminal/interrupted`` — and the card went on saying ``running`` for 49 minutes:

  * ``tasks.claim_expires`` was 2067 s in the past and ``last_heartbeat_at`` was NULL;
  * every recorded worker pid was dead;
  * ``consecutive_failures`` stayed 0, so the ``failure_limit`` breaker never applied;
  * ``release_stale_claims`` and ``detect_crashed_workers`` both declined: the owner receipt
    was "ambiguous", and ``owner_attempt_provably_over`` required the OWNER PROCESS to be
    gone — but the owner is the live gateway. Only a human ``reclaim`` ended it.

These tests pin the disposition the dispatcher owes every lost attempt:

  A1  a killed worker under a live claim is booked ``crashed`` within one tick;
  A2  an elapsed ``claim_expires`` never leaves the card ``running`` after one tick, even
      with no heartbeat and no pid (and even behind a settled owner receipt);
  A3  every such booking increments ``consecutive_failures``, so ``kanban.failure_limit``
      covers this outcome class and the card blocks at the limit.
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
from hermes_cli import kanban_owner_recovery

NOW = int(time.time())
HOST = kb._claimer_id().split(":", 1)[0]
FINGERPRINT = "6a232414-2dbf-40ab-a534-d9b71db05658:116|4162240"


def _absent_pid(base: int = 3_800_000) -> int:
    pid = base
    while psutil.pid_exists(pid):
        pid += 1
    return pid


@pytest.fixture
def board(tmp_path, monkeypatch):
    """An isolated HERMES_HOME board plus the owner ledger a receipt points at."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kbc.connect()
    ledger = tmp_path / "state.db"
    with sqlite3.connect(ledger) as store:
        store.execute(
            "CREATE TABLE session_admissions "
            "(target_session_id TEXT, request_id TEXT, status TEXT, outcome TEXT)"
        )
    scratch = _Board(conn, ledger)
    yield scratch
    conn.close()


class _Board:
    def __init__(self, conn, ledger: Path):
        self.conn = conn
        self.ledger = ledger

    # --- seeding ----------------------------------------------------------------------
    def running(
        self, tid: str = "t_lost", *, worker_pid=None, record_worker_pid: bool = True,
        claim_expires_in: int = 900, heartbeat_age=None, owner: bool = True,
        admission=("terminal", "interrupted"), worker_lost=None, dead_seconds: int = 120,
    ) -> int:
        """A ``running`` card whose worker died the way run 730's did."""
        worker_pid = _absent_pid() if worker_pid is None else int(worker_pid)
        lock = f"{HOST}:{_absent_pid(base=3_900_000)}"
        started = NOW - dead_seconds
        run_id = int(self.conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'fan', 'running', ?, ?, ?, ?)",
            (tid, lock, NOW + claim_expires_in, worker_pid if record_worker_pid else None, started),
        ).lastrowid)
        self.conn.execute(
            "INSERT INTO tasks (id, title, assignee, status, created_at, claim_lock, claim_expires, "
            "worker_pid, worker_started_at, last_heartbeat_at, current_run_id, started_at, priority) "
            "VALUES (?, ?, 'fan', 'running', ?, ?, ?, ?, ?, ?, ?, ?, 5)",
            (tid, tid, NOW - 86400, lock, NOW + claim_expires_in,
             worker_pid if record_worker_pid else None,
             FINGERPRINT if record_worker_pid else None,
             (NOW - heartbeat_age) if heartbeat_age is not None else None, run_id, started),
        )
        self.event(tid, run_id, "claimed", {"lock": lock, "expires": NOW - claim_expires_in * 0}, started)
        self.event(tid, run_id, "spawned",
                   {"pid": worker_pid, "started_at": FINGERPRINT} if record_worker_pid else {}, started)
        if owner:
            request_id = f'kanban:["default","{tid}",{run_id}]'
            self.event(tid, run_id, "owner_admitted",
                       {"db": str(self.ledger), "session_id": "sess-1", "pid": _absent_pid(base=3_950_000),
                        "request_id": request_id}, started + 3)
            if admission:
                with sqlite3.connect(self.ledger) as store:
                    store.execute("INSERT INTO session_admissions VALUES ('sess-1', ?, ?, ?)",
                                  (request_id, admission[0], admission[1]))
        if worker_lost:
            self.event(tid, run_id, "worker_lost",
                       {"reason": worker_lost, "claim_lock": lock, "run_id": run_id,
                        "pid": worker_pid, "bound": False}, started + 24)
        self.conn.commit()
        return run_id

    def event(self, tid: str, run: int, kind: str, payload: dict, ts: int) -> None:
        self.conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?)", (tid, run, kind, json.dumps(payload), ts))

    # --- reading ---------------------------------------------------------------------
    def state(self, tid: str) -> dict:
        return dict(self.conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())

    def run(self, run_id: int) -> dict:
        return dict(self.conn.execute("SELECT * FROM task_runs WHERE id = ?", (run_id,)).fetchone())

    def tick(self, *, failure_limit: int | None = None) -> kbd.DispatchResult:
        """One real dispatcher tick: reclaim phase + promotion, no spawn slot."""
        return kbd.dispatch_once(self.conn, max_spawn=0, failure_limit=failure_limit,
                                 stale_timeout_seconds=0)


# --- A1: a killed worker under a LIVE claim is booked crashed within one tick --------------

def test_a_killed_managed_worker_is_disposed_of_within_one_tick(board):
    run_id = board.running("t_killed", claim_expires_in=900, heartbeat_age=30,
                           worker_lost="managed_worker_failed")
    board.tick(failure_limit=2)
    row, run = board.state("t_killed"), board.run(run_id)
    assert row["status"] == "ready"                      # left ``running``
    assert (row["claim_lock"], row["worker_pid"], row["current_run_id"]) == (None, None, None)
    assert run["outcome"] == "crashed"
    assert "managed worker lost: managed_worker_failed" in run["error"]
    assert json.loads(run["metadata"])["attempt_note"] == "managed worker lost: managed_worker_failed"
    assert row["consecutive_failures"] == 1               # A3: the breaker sees it
    assert row["last_failure_error"]


def test_a_killed_managed_worker_blocks_the_card_at_the_failure_limit(board):
    """The breaker covers this class: one loss short of the limit, the next loss blocks.

    ``_account_crashes`` books a plain crash against ``DEFAULT_FAILURE_LIMIT`` (the sweep
    resolves the threshold there rather than against the dispatcher's ``failure_limit``),
    so the card is seeded one failure short of that limit instead of pinning a threshold
    this path never reads.
    """
    board.running("t_killed", claim_expires_in=900, heartbeat_age=30,
                  worker_lost="managed_worker_failed")
    board.conn.execute("UPDATE tasks SET consecutive_failures = ? WHERE id = 't_killed'",
                       (kbd.DEFAULT_FAILURE_LIMIT - 1,))
    board.conn.commit()
    board.tick()
    row = board.state("t_killed")
    assert row["status"] == "blocked"
    assert row["consecutive_failures"] == kbd.DEFAULT_FAILURE_LIMIT
    kinds = [r["kind"] for r in board.conn.execute(
        "SELECT kind FROM task_events WHERE task_id = 't_killed' ORDER BY id")]
    assert "gave_up" in kinds


def test_a_lost_worker_that_exited_cleanly_still_charges_the_breaker(board):
    """rc=0 after a lost managed execution is not a protocol violation: the turn never ran."""
    run_id = board.running("t_clean", claim_expires_in=900, heartbeat_age=30,
                           worker_lost="managed_worker_failed")
    # The worker's own exit trailer says rc=0 (the CLI's epilogue after a lost turn).
    log = kb.worker_log_path("t_clean")
    log.parent.mkdir(parents=True, exist_ok=True)
    from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER
    log.write_text(f"{KANBAN_WORKER_EXIT_TRAILER}0\n", encoding="utf-8")
    board.tick(failure_limit=2)
    run = board.run(run_id)
    assert run["outcome"] == "crashed"
    assert "managed worker lost" in run["error"]
    assert board.state("t_clean")["consecutive_failures"] == 1


# --- A2: an elapsed claim lock is never "still running" after one tick ---------------------

@pytest.mark.parametrize("shape", [
    dict(record_worker_pid=False, owner=False),                       # no pid, no owner at all
    dict(record_worker_pid=False, owner=True),                        # no pid, settled receipt
    dict(owner=False),                                                # dead pid, no owner
    dict(owner=True, worker_lost="managed_worker_failed"),            # the incident shape
], ids=["no-pid", "no-pid-receipt", "dead-pid", "dead-pid-lost"])
def test_an_elapsed_claim_never_stays_running_after_one_tick(board, shape):
    run_id = board.running("t_lapsed", claim_expires_in=-2067, heartbeat_age=None, **shape)
    board.tick(failure_limit=2)
    row, run = board.state("t_lapsed"), board.run(run_id)
    assert row["status"] != "running"
    assert row["status"] == "ready"
    assert run["outcome"] in {"crashed", "reclaimed"}
    assert run["error"]
    assert row["consecutive_failures"] == 1


def test_the_incident_shape_names_the_loss_on_the_reclaim(board):
    """Run 730's exact shape, behind a settled receipt: the reclaim says what happened."""
    run_id = board.running("t_730", claim_expires_in=-2067, owner=True,
                           worker_lost="managed_worker_failed")
    board.tick(failure_limit=2)
    run = board.run(run_id)
    assert run["outcome"] == "reclaimed"
    assert "managed worker lost: managed_worker_failed" in run["error"]
    event = board.conn.execute(
        "SELECT payload FROM task_events WHERE task_id = 't_730' AND kind = 'reclaimed' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    assert json.loads(event["payload"])["attempt_note"] == "managed worker lost: managed_worker_failed"


def test_a_settled_receipt_with_a_live_gateway_owner_no_longer_holds_the_claim(board):
    """The fence itself: a live owner PROCESS is not a live ATTEMPT."""
    board.running("t_gateway", claim_expires_in=-2067, owner=True, admission=("terminal", "interrupted"))
    assert kanban_owner_recovery.owner_reclaim_paused(board.conn, "t_gateway") is True
    assert kanban_owner_recovery.owner_attempt_provably_over(board.conn, "t_gateway") is True
    board.tick(failure_limit=2)
    assert board.state("t_gateway")["status"] == "ready"


# --- the reclaim's kill step: an absent pid is gone, not "survived" -----------------------

def test_terminating_an_absent_pid_reports_terminated(board):
    """A pid that is already gone must not be reported as unproven termination.

    Windows raises a bare ``OSError`` (WinError 87) for an absent PID instead of
    ``ProcessLookupError``; reading that as "termination not proven" made the reclaim hold
    the claim and defer every tick — the elapsed lock stayed ``running`` forever.
    """
    info = kbd._terminate_reclaimed_worker(_absent_pid(), f"{HOST}:{_absent_pid(base=3_960_000)}")
    assert info["host_local"] is True
    assert info["termination_attempted"] is True
    assert info["terminated"] is True
    if os.name == "nt":
        # Only Windows reports an absent pid as a bare OSError, so only there does the kill
        # step take the re-ask branch that marks this verdict; POSIX reaches the same
        # ``terminated`` through ProcessLookupError (pinned on every platform by
        # test_a_signalled_absent_pid_is_terminated_not_unproven).
        assert info["already_gone"] is True


def test_a_signalled_absent_pid_is_terminated_not_unproven(board):
    """The same reading through a signal seam: an OSError on a dead pid is a success."""
    gone = _absent_pid(base=3_970_000)

    def windows_kill(pid, sig):
        raise OSError(87, "The parameter is incorrect")

    info = kbd._terminate_reclaimed_worker(gone, f"{HOST}:{_absent_pid(base=3_980_000)}",
                                           signal_fn=windows_kill)
    assert info["terminated"] is True
    assert info["already_gone"] is True


# --- D1: the OWNER writes the loss where the dispatcher can read it -----------------------
# ``worker.adopt`` precedes ``bind_worker_context``, so a worker that dies in the handshake
# writes neither ``worker_bound`` nor ``worker_result``: without this event the run learns
# nothing about its own loss and the dispatcher can only guess from a dead pid.

def _owner_policy(board, tid, run_id):
    from types import SimpleNamespace
    row = board.state(tid)
    return SimpleNamespace(kanban_json=json.dumps({
        "task_id": tid, "db": str(kb.kanban_db_path()), "run_id": run_id,
        "claim_lock": row["claim_lock"], "board": None,
    }))


def test_the_owner_records_the_loss_on_the_board_and_one_tick_disposes_of_it(board):
    from gateway.session_kanban import record_worker_loss
    run_id = board.running("t_owner", claim_expires_in=900, heartbeat_age=30)
    policy = _owner_policy(board, "t_owner", run_id)
    record_worker_loss(policy, "managed_worker_failed", pid=board.state("t_owner")["worker_pid"],
                       bound=False)
    event = board.conn.execute(
        "SELECT payload FROM task_events WHERE task_id='t_owner' AND kind='worker_lost' "
        "ORDER BY id DESC LIMIT 1").fetchone()
    assert json.loads(event["payload"])["reason"] == "managed_worker_failed"

    board.tick(failure_limit=2)
    run = board.run(run_id)
    assert run["outcome"] == "crashed"
    assert "managed worker lost: managed_worker_failed" in run["error"]
    assert board.state("t_owner")["consecutive_failures"] == 1


def test_a_loss_recorded_against_a_superseded_run_is_dropped(board):
    """The owner's note is run-scoped: it may never be evidence about a LATER attempt."""
    from gateway.session_kanban import record_worker_loss
    run_id = board.running("t_stale", claim_expires_in=900, heartbeat_age=30)
    policy = _owner_policy(board, "t_stale", run_id)
    board.conn.execute("UPDATE tasks SET status='ready', current_run_id=NULL WHERE id='t_stale'")
    board.conn.commit()
    record_worker_loss(policy, "managed_worker_failed", pid=1234, bound=False)
    n = board.conn.execute(
        "SELECT count(*) FROM task_events WHERE task_id='t_stale' AND kind='worker_lost'"
    ).fetchone()[0]
    assert n == 0

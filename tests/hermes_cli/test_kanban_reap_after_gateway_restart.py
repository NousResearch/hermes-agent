"""A card whose worker died with the gateway is reaped by the dispatcher, not by an operator.

Regression for the 4 Nov incident. The default-profile gateway restarted at 21:17 while two cards
were in flight; both managed workers died with ``EOFError`` (``worker_lost`` was recorded), and both
cards stayed ``running`` with a dead pid for 38 minutes until a human reclaimed them. The spawn path
was fine — the RECLAIM paths were blind:

* ``detect_crashed_workers`` is the only path that can book a run whose owner receipt is unresolved
  and whose bound interpreter is dead (the TTL and stale sweeps are held by ``owner_reclaim_blocked``),
  and it starts with "is this claim host-local?".
* That test compared the raw ``host:`` prefix. The claim had been written by the process that saw
  ``TheMateMachine:15104``; every process after the 21:56 restart saw ``The-Mate-Machine:3612``. Same
  machine, different spelling, so the sweep skipped both rows on every tick. With
  ``kanban.max_in_progress=2`` the two orphans then held every spawn slot.

These tests pin the recovery the board owes a killed worker: same machine = reapable however the
machine's name was spelled, a genuinely foreign host = still left alone, and the full dispatcher
tick re-spawns the card with no operator involved.
"""

from __future__ import annotations

import json
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

NOW = int(time.time())
HOST = kb._claimer_id().split(":", 1)[0]
# A spawn fingerprint that matches no live process on any host (unknown boot witness, old start).
FINGERPRINT = "6a232414-2dbf-40ab-a534-d9b71db05658:116|4162240"


def _absent_pid(base: int = 3_800_000) -> int:
    """A pid that provably does not exist on this host."""
    pid = base
    while psutil.pid_exists(pid):
        pid += 1
    return pid


def _respelled_host() -> str:
    """This machine's name as a previous process spelled it: other case, no separators."""
    variant = HOST.replace("-", "").replace("_", "").replace(".", "").upper()
    # Vacuity guard: the test only means something while the OLD raw-prefix gate would have skipped
    # the claim. On a host whose name survives the respelling unchanged, fail loudly instead of
    # passing for the wrong reason.
    assert not f"{variant}:1".startswith(kb._host_prefix()), (
        f"host {HOST!r} does not change under respelling; the test cannot exercise the gate")
    return variant


class _Board:
    """A scratch board plus the owner ledger an ``owner_admitted`` receipt points at."""

    def __init__(self, root: Path) -> None:
        self.path = root / "kanban.db"
        kbc.init_db(self.path)
        self.conn = kbc.connect(self.path)
        self.ledger = root / "state.db"
        with sqlite3.connect(self.ledger) as store:
            store.execute(
                "CREATE TABLE session_admissions "
                "(target_session_id TEXT, request_id TEXT, status TEXT, outcome TEXT)"
            )
            # What a gateway killed mid-turn leaves behind: a settled, interrupted admission.
            store.execute("INSERT INTO session_admissions VALUES ('sess-1', 'req-1', 'terminal', 'interrupted')")

    def close(self) -> None:
        self.conn.close()

    def event(self, tid: str, run: int | None, kind: str, payload: dict, ts: int) -> None:
        self.conn.execute(
            "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) VALUES (?, ?, ?, ?, ?)",
            (tid, run, kind, json.dumps(payload), ts),
        )

    def running(self, tid: str, *, lock_host: str, source_status: str = "review",
                worker_lost: str | None = "EOFError") -> int:
        """A ``running`` card exactly as the restart left it: dead bound worker, unresolved owner."""
        worker_pid = _absent_pid()
        owner_pid = _absent_pid(base=3_900_000)
        lock = f"{lock_host}:{owner_pid}"
        started = NOW - 900
        raw_run_id = self.conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, claim_lock, claim_expires, "
            "worker_pid, started_at) VALUES (?, 'default', 'running', ?, ?, ?, ?)",
            (tid, lock, NOW - 300, worker_pid, started),
        ).lastrowid
        assert raw_run_id is not None
        run_id = int(raw_run_id)
        self.conn.execute(
            "INSERT INTO tasks (id, title, assignee, status, created_at, claim_lock, claim_expires, "
            "worker_pid, worker_started_at, last_heartbeat_at, current_run_id, started_at, priority) "
            "VALUES (?, ?, 'default', 'running', ?, ?, ?, ?, ?, ?, ?, ?, 5)",
            (tid, tid, NOW - 86_400, lock, NOW - 300, worker_pid, FINGERPRINT, NOW - 1200, run_id, started),
        )
        self.event(tid, run_id, "claimed", {"lock": lock, "expires": NOW - 300,
                                            "run_id": run_id, "source_status": source_status}, started)
        self.event(tid, run_id, "spawned", {"pid": worker_pid, "started_at": FINGERPRINT}, started + 1)
        self.event(tid, run_id, "owner_admitted",
                   {"db": str(self.ledger), "session_id": "sess-1", "pid": owner_pid,
                    "request_id": "req-1"}, started + 2)
        self.event(tid, run_id, "worker_bound",
                   {"pid": worker_pid, "claim_lock": lock, "started_at": FINGERPRINT}, started + 3)
        if worker_lost:
            self.event(tid, run_id, "worker_lost",
                       {"reason": worker_lost, "claim_lock": lock, "run_id": run_id,
                        "pid": worker_pid, "bound": True}, started + 4)
        self.conn.commit()
        return run_id

    def ready(self, tid: str) -> None:
        self.conn.execute(
            "INSERT INTO tasks (id, title, assignee, status, created_at, priority) "
            "VALUES (?, ?, 'default', 'ready', ?, 5)", (tid, tid, NOW - 60),
        )
        self.conn.commit()

    # --- reading ------------------------------------------------------------------------
    def state(self, tid: str) -> dict:
        return dict(self.conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone())

    def run(self, tid: str) -> dict:
        return dict(self.conn.execute(
            "SELECT status, outcome, error FROM task_runs WHERE task_id = ? ORDER BY id DESC", (tid,)
        ).fetchone())

    def events(self, tid: str) -> list:
        return self.conn.execute(
            "SELECT kind, payload, created_at, run_id FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
        ).fetchall()

    def spawned_events(self, tid: str) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'spawned'", (tid,)
        ).fetchone()[0])


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    scratch = _Board(tmp_path)
    yield scratch
    scratch.close()


# --- the failing shape: same machine, a different spelling of its name ---------------------

def test_a_claim_written_by_a_previous_process_on_this_machine_is_reaped(board):
    """The gateway restart renamed the box (NetBIOS/DNS spelling): the claim must stay reapable."""
    variant = _respelled_host()
    board.running("t_orphan", lock_host=variant)

    assert kbd.detect_crashed_workers(board.conn) == ["t_orphan"]
    row = board.state("t_orphan")
    # Back to the lane it came from, claim released, no dead pid left behind.
    assert (row["status"], row["claim_lock"], row["worker_pid"], row["worker_started_at"]) == (
        "review", None, None, None)
    run = board.run("t_orphan")
    assert (run["status"], run["outcome"]) == ("crashed", "crashed")
    assert "EOFError" in run["error"] or "not alive" in run["error"]
    kinds = [e["kind"] for e in board.events("t_orphan")]
    assert "crashed" in kinds and "worker_lost" in kinds


def test_a_genuinely_foreign_hosts_claim_is_still_left_alone(board):
    """The gate exists so another machine's pids are never judged (and signalled) here."""
    board.running("t_remote", lock_host="some-other-host")

    assert kbd.detect_crashed_workers(board.conn) == []
    assert board.state("t_remote")["status"] == "running"


def test_a_reap_after_a_restart_leaves_no_stuck_diagnostics(board):
    """No ``stale_running`` for a card the dispatcher just recovered."""
    variant = _respelled_host()
    board.running("t_orphan", lock_host=variant)
    assert kbd.detect_crashed_workers(board.conn) == ["t_orphan"]
    row = board.state("t_orphan")
    diags = kd.compute_task_diagnostics(row, board.events("t_orphan"), [], now=NOW + 4000, config={})
    assert [d.kind for d in diags if d.kind in {"stale_running", "stranded_in_ready"}] == []


# --- end to end through the real dispatcher: kill the worker, tick, re-spawned --------------

def test_a_killed_worker_is_reaped_and_respawned_without_an_operator(board, monkeypatch):
    """The shipped recovery: kill a running worker mid-task; the next tick books the crash and
    re-spawns the card. Nothing here calls a reclaim by hand."""
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: (lambda name: True))
    tid = "t_killed"
    board.ready(tid)

    children: list[subprocess.Popen] = []
    spawned: list[int] = []

    def spawn(task, workspace, board=None):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        children.append(child)
        spawned.append(child.pid)
        return child.pid

    try:
        first = kbd.dispatch_once(board.conn, spawn_fn=spawn)
        assert len(spawned) == 1 and first.spawned, first
        assert board.state(tid)["status"] == "running"

        # Kill the worker mid-task, exactly as a gateway restart does (the pid simply vanishes).
        proc = children[0]
        proc.kill()
        proc.wait(timeout=30)
        assert not psutil.pid_exists(proc.pid)

        after = kbd.dispatch_once(board.conn, spawn_fn=spawn)
        assert tid in after.crashed, after.crashed
        # The reclaim phase runs before the spawn phase, so the same tick re-spawns it.
        assert len(spawned) == 2, (spawned, after)
        row = board.state(tid)
        assert row["status"] == "running" and row["worker_pid"] == spawned[1]
        # The killed attempt is booked once; the re-spawn opened the next (still open) run.
        assert [r["outcome"] for r in board.conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id = ? ORDER BY id", (tid,))] == ["crashed", None]
        # Nothing is left stranded: the live card is claimed and running, not queued behind itself.
        diags = kd.compute_task_diagnostics(
            row, board.events(tid), [], now=NOW + 4000, config={})
        assert [d.kind for d in diags if d.kind in {"stale_running", "stranded_in_ready"}] == []
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=30)

"""Respawn containment across process boundaries.

Every dispatcher tick here runs in a FRESH Python process against the same
board DB, which is what a gateway or dispatcher restart looks like to the
board, and workers are real child processes that exit or are SIGKILLed. Backoff,
exhaustion parking and failure history must come from durable board state.
Nothing may live only in a dispatcher's memory.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc

pytestmark = pytest.mark.platforms("posix")

# One dispatcher tick in a new process. FAKE_NOW pins the clock so backoff
# windows are checked exactly; WORKER picks what the spawned worker does.
_TICK = r"""
import json, os, subprocess, sys, time
if os.environ.get("FAKE_NOW"):
    _now = float(os.environ["FAKE_NOW"])
    time.time = lambda: _now
from hermes_cli import profiles
profiles.profile_exists = lambda name: True
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

body = {
    # Dies by SIGKILL with no exit trailer, like an OOM kill. It kills itself, so no
    # test process signals a PID outside its subtree.
    "killed": "import os, signal, time; time.sleep(2); os.kill(os.getpid(), signal.SIGKILL)",
    "rate_limited": f"print({KANBAN_WORKER_EXIT_TRAILER!r} + '{kb.KANBAN_RATE_LIMIT_EXIT_CODE}'); "
                    f"raise SystemExit({kb.KANBAN_RATE_LIMIT_EXIT_CODE})",
}[os.environ["WORKER"]]
spawned = []

def spawn(task, workspace, board=None):
    log = kb.worker_log_path(task.id, board=board)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as out:
        proc = subprocess.Popen([sys.executable, "-c", body], stdout=out, stderr=out)
    spawned.append(proc.pid)
    return proc.pid

with kbc.connect() as conn:
    kbd.dispatch_once(conn, spawn_fn=spawn)
print(json.dumps(spawned))
"""


@pytest.fixture
def board(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    home = tmp_path / ".hermes"
    home.mkdir()
    env = {**os.environ, "HERMES_HOME": str(home), "HOME": str(tmp_path),
           "HERMES_KANBAN_CRASH_GRACE_SECONDS": "0",
           "HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS": "300",
           "HERMES_KANBAN_RATE_LIMIT_MAX_COOLDOWN_SECONDS": "3600",
           "HERMES_KANBAN_RATE_LIMIT_MAX_CONSECUTIVE": "3"}
    env.pop("HERMES_KANBAN_TASK", None)
    for key in ("HERMES_HOME", "HERMES_KANBAN_CRASH_GRACE_SECONDS", "HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS",
                "HERMES_KANBAN_RATE_LIMIT_MAX_COOLDOWN_SECONDS", "HERMES_KANBAN_RATE_LIMIT_MAX_CONSECUTIVE"):
        monkeypatch.setenv(key, env[key])
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="quota-bound", assignee="worker")
    return {"env": env, "tid": tid}


def _tick(board: dict, *, worker: str, now: int | None = None) -> list[int]:
    """Run one dispatcher tick in a new process; return the worker pids it spawned."""
    env = {**board["env"], "WORKER": worker, **({"FAKE_NOW": str(now)} if now is not None else {})}
    out = subprocess.run([sys.executable, "-c", _TICK], env=env, capture_output=True, text=True,
                         timeout=120, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _cli(board: dict, command: str, *, profile: str) -> str:
    """``hermes kanban <command>`` as an operator in their own process."""
    env = {**board["env"], "HERMES_PROFILE": profile}
    out = subprocess.run(
        [sys.executable, "-c", "import sys; from hermes_cli import kanban; print(kanban.run_slash(sys.argv[1]))",
         command], env=env, capture_output=True, text=True, timeout=120, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def _wait_dead(pid: int) -> None:
    deadline = time.time() + 20
    while time.time() < deadline:
        if not kb._pid_alive(pid):
            return
        time.sleep(0.05)
    raise AssertionError(f"worker {pid} still alive")


def _snapshot(tid: str) -> tuple[kb.Task, list, list]:
    with kbc.connect() as conn:
        runs = conn.execute(
            "SELECT id, outcome, ended_at FROM task_runs WHERE task_id = ? ORDER BY id", (tid,)).fetchall()
        return kb.get_task(conn, tid), runs, kb.list_events(conn, tid)


def _rate_limited_cycle(board: dict, now: int) -> int:
    """Spawn a worker that hits the quota wall, then book its exit in a later
    process. Returns the booked run's ``ended_at``."""
    (pid,) = _tick(board, worker="rate_limited", now=now)
    _wait_dead(pid)
    assert _tick(board, worker="rate_limited", now=now + 10) == []
    rate_limited = [r for r in _snapshot(board["tid"])[1] if r["outcome"] == "rate_limited"]
    assert rate_limited[-1]["ended_at"] == now + 10
    return now + 10


def test_backoff_and_exhaustion_survive_dispatcher_restarts_then_unblock_retries_once(board):
    """(1) backoff holds across fresh dispatcher processes, (2) exhaustion
    parking survives restarts, (4) an authorized unblock gets one controlled
    retry that keeps the earlier failure history."""
    tid, t0 = board["tid"], 1_000_000

    ended = _rate_limited_cycle(board, t0)
    assert _snapshot(tid)[0].status == "ready"
    assert _tick(board, worker="rate_limited", now=ended + 299) == []  # 300s window, new process

    ended = _rate_limited_cycle(board, ended + 301)
    assert _snapshot(tid)[0].status == "ready"
    assert _tick(board, worker="rate_limited", now=ended + 599) == []  # doubled to 600s, new process
    assert _tick(board, worker="rate_limited", now=ended + 301) == []

    ended = _rate_limited_cycle(board, ended + 601)  # third consecutive: exhausted → parked
    task, runs, events = _snapshot(tid)
    assert (task.status, task.block_kind) == ("blocked", "needs_input")
    assert [r["outcome"] for r in runs].count("rate_limited") == 3
    guarded = [e.payload["reason"] for e in events if e.kind == "respawn_guarded"]
    assert guarded.count("rate_limit_exhausted") == 1

    for offset in (60, 86_400, 30 * 86_400):  # gateway restarts long after any backoff
        assert _tick(board, worker="rate_limited", now=ended + offset) == []
    task, runs_after, events_after = _snapshot(tid)
    assert task.status == "blocked"
    assert len(runs_after) == len(runs)
    assert [e for e in events_after if e.kind == "blocked"] == [e for e in events if e.kind == "blocked"]

    assert "Unblocked" in _cli(board, f"unblock {tid}", profile="ops")
    (pid,) = _tick(board, worker="killed")
    assert _tick(board, worker="killed") == []  # one retry, not a burst
    task, runs_final, events_final = _snapshot(tid)
    assert task.status == "running" and task.worker_pid == pid
    assert [r["outcome"] for r in runs_final].count("rate_limited") == 3, "history kept"
    unblocked = [e.payload for e in events_final if e.kind == "unblocked"]
    assert unblocked == [{"actor": "ops", "source": "cli"}]
    claimed = [e.payload for e in events_final if e.kind == "claimed"][-1]
    assert (claimed["actor"], claimed["source"]) == ("dispatcher", "dispatcher")
    assert claimed["retry_of"]["outcome"] == "blocked"


def test_killed_workers_are_reclaimed_and_the_breaker_bounds_respawns(board):
    """(3) A SIGKILLed worker never leaves the card claimed: a fresh dispatcher
    process reclaims it, and repeated kills stop at the failure breaker instead
    of respawning forever."""
    tid, spawned = board["tid"], []
    for _ in range(6):
        pids = _tick(board, worker="killed")
        spawned += pids
        task, runs, _ = _snapshot(tid)
        if task.status != "running":
            break
        assert task.claim_lock and task.worker_pid == pids[-1]
        _wait_dead(task.worker_pid)

    task, runs, _ = _snapshot(tid)
    assert len(spawned) == 2, "DEFAULT_FAILURE_LIMIT spawns, then the breaker holds"
    assert task.status == "blocked"
    assert (task.claim_lock, task.worker_pid, task.current_run_id) == (None, None, None)
    assert all(r["ended_at"] is not None for r in runs), "no run left open"
    assert task.consecutive_failures == 2
    assert _tick(board, worker="killed") == []


def test_manual_claim_records_identity_or_explicit_unknown(board):
    """Manual claims say who did it. A CLI call names its profile. A library
    call with no identity records ``unknown`` and does not guess a name."""
    tid = board["tid"]
    assert "Claimed" in _cli(board, f"claim {tid}", profile="ops")
    with kbc.connect() as conn:
        assert kb.reclaim_task(conn, tid, reason="test", signal_fn=lambda *_: None)
        assert kb.claim_task(conn, tid) is not None
        claims = [e.payload for e in kb.list_events(conn, tid) if e.kind == "claimed"]
    assert [(c["actor"], c["source"]) for c in claims] == [("ops", "cli"), ("unknown", "api")]
    assert claims[1]["retry_of"]["outcome"] == "reclaimed"

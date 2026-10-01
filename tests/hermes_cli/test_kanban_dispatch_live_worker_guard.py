"""Invariant: two worker processes for the same task id are never alive
simultaneously (parent t_810b425e, evidence t_643c3a0e runs 425/427 — run 427
spawned a second worker while run 425's process was still alive).

The dispatcher's claim path is the last line of defence: before claiming a
ready/review row, any LIVE worker process still recorded for the task
(retained ``task_runs`` pid + spawn fingerprint, or a fingerprinted
``tasks.worker_pid``) defers the dispatch — no claim, no run row, no spawn,
no circuit-breaker charge. The next tick retries once the predecessor is
gone. Liveness is fingerprint-aware (:func:`_worker_alive`), so a recycled
pid, a legacy row without a fingerprint and a non-local ``claim_lock`` never
block a successor; a run still inside ``TERMINAL_WORKER_REAP_GRACE_SECONDS``
belongs to the terminal-worker reaper, not to this guard.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


@pytest.fixture
def spawnable(monkeypatch):
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


def _sleeper():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.2)
    return proc


def _redispatchable_card(conn, proc, *, ended_ago: "int | None" = 600) -> tuple[str, int]:
    """A ready card whose previous run retains a live worker pid + fingerprint.

    ``ended_ago=None`` keeps the run row open (``ended_at IS NULL``) — the
    literal t_643c3a0e shape where a successor claim was taken while the
    predecessor's run was never closed; otherwise the run is closed that long
    ago, the shape a reclaim leaves behind (evidence kept for the reaper)."""
    tid = kb.create_task(conn, title="re-dispatch after reclaim", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    kbd._set_worker_pid(conn, tid, proc.pid)
    # A reclaim released the claim and returned the card to ready; the run row
    # keeps pid + fingerprint until the reaper clears them.
    conn.execute(
        "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
        "worker_pid=NULL, worker_started_at=NULL, current_run_id=NULL, "
        "last_heartbeat_at=NULL WHERE id=?",
        (tid,),
    )
    if ended_ago is not None:
        conn.execute(
            "UPDATE task_runs SET ended_at = ?, status='reclaimed', "
            "outcome='reclaimed' WHERE id=?",
            (int(time.time()) - ended_ago, run_id),
        )
    conn.commit()
    return tid, run_id


@pytest.fixture
def reaper_holds_fire(monkeypatch):
    """The terminal-worker reaper terminates a live predecessor before the
    lanes run; stub it so the CLAIM guard is exercised in isolation (the shape
    of a kill that has not landed yet — termination is retried every tick)."""
    monkeypatch.setattr(kbd, "reap_terminal_workers", lambda conn, **k: [])


def test_live_predecessor_defers_the_claim(conn, spawnable, reaper_holds_fire):
    proc = _sleeper()
    try:
        tid, old_run_id = _redispatchable_card(conn, proc)
        spawned: list[str] = []

        def spawn_fn(task, workspace, board=None):
            spawned.append(task.id)
            return 4242

        result = kbd.dispatch_once(conn, spawn_fn=spawn_fn)

        assert result.live_worker_deferred == [tid]
        assert spawned == []  # no second worker beside the live predecessor
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid, current_run_id FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        assert (row["status"], row["claim_lock"], row["worker_pid"], row["current_run_id"]) == (
            "ready", None, None, None,
        )
        # No new run row: the claim itself was refused.
        runs = conn.execute("SELECT id FROM task_runs WHERE task_id=?", (tid,)).fetchall()
        assert [r["id"] for r in runs] == [old_run_id]
        # Visible on the board, naming the live pid and the run it belonged to.
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='dispatch_deferred_live_worker'",
            (tid,),
        ).fetchall()
        assert len(events) == 1
        payload = json.loads(events[0]["payload"])
        assert payload["pids"][0]["pid"] == proc.pid
        assert payload["pids"][0]["run_id"] == old_run_id
    finally:
        proc.kill()
        proc.wait()


def test_deferral_never_charges_the_breaker(conn, spawnable, reaper_holds_fire):
    """Deferral is not a spawn failure: repeated ticks while the predecessor
    hangs around never move ``consecutive_failures`` (the auto-block breaker
    must not see a busy predecessor)."""
    proc = _sleeper()
    try:
        tid, _ = _redispatchable_card(conn, proc)
        for _ in range(5):
            result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0, failure_limit=2)
            assert result.live_worker_deferred == [tid]
            assert result.auto_blocked == []
        row = conn.execute(
            "SELECT status, consecutive_failures, block_kind FROM tasks WHERE id=?", (tid,)
        ).fetchone()
        assert (row["status"], row["consecutive_failures"], row["block_kind"]) == ("ready", 0, None)
        # Rate-limited event: identical evidence is written once, not per tick.
        events = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='dispatch_deferred_live_worker'",
            (tid,),
        ).fetchall()
        assert len(events) == 1
    finally:
        proc.kill()
        proc.wait()


def test_dead_predecessor_dispatch_proceeds(conn, spawnable):
    proc = _sleeper()
    proc.kill()  # dead before the tick: the successor must not be blocked
    proc.wait()
    tid, _ = _redispatchable_card(conn, proc)

    def spawn_fn(task, workspace, board=None):
        return 5151

    result = kbd.dispatch_once(conn, spawn_fn=spawn_fn)
    assert result.live_worker_deferred == []
    assert result.spawned and result.spawned[0][0] == tid
    row = conn.execute("SELECT status, worker_pid FROM tasks WHERE id=?", (tid,)).fetchone()
    assert (row["status"], row["worker_pid"]) == ("running", 5151)


def test_recycled_or_legacy_or_remote_evidence_never_blocks(conn, spawnable):
    """A recycled pid (fingerprint mismatch), a legacy row without a
    fingerprint and a non-local claim_lock are never treated as a live
    predecessor — bare existence is not worker identity."""
    stranger = _sleeper()  # unrelated live process
    try:
        tid, run_id = _redispatchable_card(conn, stranger)
        # 1. Recycled: the recorded fingerprint no longer matches the live pid.
        conn.execute(
            "UPDATE task_runs SET worker_started_at = worker_started_at - 1000000 WHERE id=?",
            (run_id,),
        )
        # 2. Legacy row without a fingerprint.
        legacy = _sleeper()
        try:
            conn.execute(
                "UPDATE task_runs SET worker_pid=?, worker_started_at=NULL WHERE id=?",
                (legacy.pid, run_id),
            )
            # 3. Non-local claim lock: a remote pid means nothing on this host.
            conn.execute(
                "UPDATE task_runs SET worker_pid=?, worker_started_at=(SELECT worker_started_at FROM task_runs WHERE id=?), claim_lock='otherhost:999' WHERE id=?",
                (stranger.pid, run_id, run_id),
            )
            conn.commit()
            result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0)
            assert result.live_worker_deferred == []
            assert result.spawned and result.spawned[0][0] == tid
        finally:
            legacy.kill()
            legacy.wait()
    finally:
        stranger.kill()
        stranger.wait()


def test_run_within_terminal_reap_grace_does_not_defer(conn, spawnable):
    """A just-closed run's worker is finalising its own transition; the
    terminal-worker reaper owns it after the grace window, and the lane keeps
    dispatching meanwhile."""
    proc = _sleeper()
    try:
        tid, _ = _redispatchable_card(conn, proc, ended_ago=0)
        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0)
        assert result.live_worker_deferred == []
        assert result.spawned and result.spawned[0][0] == tid
    finally:
        proc.kill()
        proc.wait()


def test_review_lane_defers_too(conn, spawnable, monkeypatch, reaper_holds_fire):
    proc = _sleeper()
    try:
        tid, _ = _redispatchable_card(conn, proc)
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (tid,))
        conn.commit()
        monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: True)
        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0)
        assert result.live_worker_deferred == [tid]
        assert result.spawned == []
    finally:
        proc.kill()
        proc.wait()


def test_open_run_with_live_pid_defers_even_when_the_reaper_runs(conn, spawnable):
    """The literal incident shape (t_643c3a0e): the predecessor's run row was
    never closed. The reaper only looks at closed runs, so the claim guard is
    what stands between the live worker and a second spawn."""
    proc = _sleeper()
    try:
        tid, _ = _redispatchable_card(conn, proc, ended_ago=None)
        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0)
        assert result.live_worker_deferred == [tid]
        assert result.spawned == []
    finally:
        proc.kill()
        proc.wait()


def test_reaper_landing_clears_the_way_same_tick(conn, spawnable):
    """Compose: a fingerprinted live predecessor past the reap grace is
    terminated by the reaper phase of the SAME tick, and the lanes then
    dispatch normally — guard and reaper never deadlock."""
    proc = _sleeper()
    try:
        tid, _ = _redispatchable_card(conn, proc)  # ended 600s ago: reapable
        result = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 0)
        assert result.reaped_terminal_workers == [tid]
        assert result.live_worker_deferred == []
        assert result.spawned and result.spawned[0][0] == tid
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

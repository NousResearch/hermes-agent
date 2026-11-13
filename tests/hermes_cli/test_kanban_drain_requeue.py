"""A worker that dies inside its owner's update wave never charges the card (t_90b63d52).

Tonight's proof (t_304ef164, t_e1a0e19c): workers cut by `hermes update` restart waves exited
plainly (no drain sentinel yet), the crash sweep booked them ``crashed``, and two of those
tripped the breaker (``failure_limit=2``), parking a card that never ran. Now the sweep books
such deaths through ``_record_task_failure(infrastructure=True)``: no ``consecutive_failures``
increment, the breaker can't trip, and the respawn guard spaces the retries — the interim
``kanban.dispatch_interval_seconds`` patch becomes unnecessary instead of load-bearing.
"""

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kbd._owner_fences_seen.clear()
    kb.init_db()
    return home


def _dead_worker_with_log(conn, tid: str, pid: int, rc: int) -> None:
    """Claim ``tid`` for a worker that already exited ``rc`` and wrote its log — never reaped here."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"the model said something\n\nResume this session with:\n  hermes --resume x\n\n{KANBAN_WORKER_EXIT_TRAILER}{rc}\n")


def _last_run_and_event(conn, tid):
    ev = conn.execute(
        "SELECT kind FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
    run = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (tid,)).fetchone()
    return ev, run


def test_drain_sentinel_exit_requeues_without_counting_a_failure(kanban_home):
    """The worker's own fence give-up (74) is booked as its owner's infrastructure: the event
    and the run say ``drained``, ``consecutive_failures`` never moves, and the respawn guard
    spaces the retry — never a breaker trip, wave or not."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker_with_log(conn, tid, 70001, kb.KANBAN_OWNER_DRAIN_EXIT_CODE)

        crashed = kbd.detect_crashed_workers(conn)
        assert tid not in crashed  # a drain requeue is not a crash
        assert kbd.detect_crashed_workers._last_drained == [tid]
        ev, run = _last_run_and_event(conn, tid)
        assert ev["kind"] == "drained"
        assert run["outcome"] == "drained"
        assert kb._json_dict(run["metadata"]).get("infrastructure") is True
        assert kb._json_dict(run["metadata"]).get("exit_code") == 74
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert "not alive" not in (run["error"] or "")
        # The retry is SPACED by the infrastructure cooldown (same rule as a refused spawn):
        # the wave may still be running, so the card is not re-spawned on the next tick...
        assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"
        # ...and no number of drained waves can ever reach the breaker: two more waves of
        # the same death (the old run 3666/3668 shape) still leave the card ready.
        for n in range(2):
            kb.recompute_ready(conn, failure_limit=2)
            _dead_worker_with_log(conn, tid, 70002 + n, kb.KANBAN_OWNER_DRAIN_EXIT_CODE)
            assert kbd.detect_crashed_workers(conn) == []
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 0


def test_fenced_owner_death_rebooks_a_plain_crash_as_drained(kanban_home, monkeypatch):
    """A worker killed without a sentinel (tonight's exit 130) is rebooked from the fence the
    sweep can see live — and the same exact death stays a crash once the fence is gone."""
    monkeypatch.setattr(kbd, "_owner_update_fenced", lambda: True)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _dead_worker_with_log(conn, tid, 70011, 1)  # a wave SIGINT often reads as exit 1/130
        assert kbd.detect_crashed_workers(conn) == []
        ev, run = _last_run_and_event(conn, tid)
        assert ev["kind"] == "drained"
        assert run["outcome"] == "drained"
        assert kb.get_task(conn, tid).consecutive_failures == 0

    # Control: the same exit 1 with no fence evidence anywhere is still a card crash the
    # breaker hears about.
    with kbc.connect() as conn:
        tid2 = kb.create_task(conn, title="t2", assignee="a")
        _dead_worker_with_log(conn, tid2, 70012, 1)
        kbd._owner_fences_seen.clear()
        monkeypatch.setattr(kbd, "_owner_update_fenced", lambda: False)
        assert kbd.detect_crashed_workers(conn) == [tid2]
        ev, run = _last_run_and_event(conn, tid2)
        assert ev["kind"] == "crashed"
        assert run["outcome"] == "crashed"
        assert kb.get_task(conn, tid2).consecutive_failures == 1


def test_recent_fence_rebooks_only_kill_shaped_deaths(kanban_home, monkeypatch):
    """A sweep that runs AFTER the wave may no longer see the fence live. Only deaths (or
    exit codes) the updater could have caused — signal-shaped, unknown, 130/143 — are excused
    on the remembered fence; a plain card crash stays a card crash."""
    monkeypatch.setattr(kbd, "_owner_update_fenced", lambda: False)

    # Stale memory (older than the TTL) is pruned, never excused.
    kbd._owner_fences_seen[:] = [time.time() - kbd._RECENT_OWNER_FENCE_TTL_SECONDS - 10]
    with kbc.connect() as conn:
        tid_old = kb.create_task(conn, title="old", assignee="a")
        _dead_worker_with_log(conn, tid_old, 70021, 130)
        assert kbd.detect_crashed_workers(conn) == [tid_old]
        assert kb.get_task(conn, tid_old).consecutive_failures == 1
        assert _last_run_and_event(conn, tid_old)[0]["kind"] == "crashed"

    # Fresh memory with a kill-shaped exit: excused as infrastructure.
    kbd._owner_fences_seen.clear()
    kbd._note_owner_fence()
    with kbc.connect() as conn:
        tid_kill = kb.create_task(conn, title="kill", assignee="a")
        _dead_worker_with_log(conn, tid_kill, 70022, 130)
        assert kbd.detect_crashed_workers(conn) == []
        ev, run = _last_run_and_event(conn, tid_kill)
        assert ev["kind"] == "drained" and run["outcome"] == "drained"
        assert kb.get_task(conn, tid_kill).consecutive_failures == 0

    # Fresh memory with a plain exit-1 crash: NOT excused (a card fault stayed a card fault).
    with kbc.connect() as conn:
        tid_plain = kb.create_task(conn, title="plain", assignee="a")
        _dead_worker_with_log(conn, tid_plain, 70023, 1)
        assert kbd.detect_crashed_workers(conn) == [tid_plain]
        assert _last_run_and_event(conn, tid_plain)[0]["kind"] == "crashed"


# --- enforce_max_runtime: a per-limit kill of a still-fencing attempt (G2) ---------------

import signal
import subprocess  # module-level local: only these tests need a live worker PID


def _claimed_sleeper(conn, tid: str, sleeper: subprocess.Popen, started_at: int) -> None:
    """Claim ``tid`` to the LIVE sleeper process with an attempt started ``started_at``."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{sleeper.pid}")
    if conn.execute("SELECT id FROM task_runs WHERE task_id=?", (tid,)).fetchone():
        conn.execute("UPDATE task_runs SET started_at=? WHERE task_id=?", (started_at, tid))
    conn.execute(
        "UPDATE tasks SET status='running', max_runtime_seconds=600, worker_pid=?, "
        "worker_started_at=?, started_at=? WHERE id=?",
        (sleeper.pid, kbd._process_fingerprint(sleeper.pid), started_at, tid),
    )
    conn.commit()


def _bound(conn, tid: str, sleeper: subprocess.Popen) -> None:
    """Record the run's ``worker_bound`` event: the attempt's working turn actually started."""
    lock_row = conn.execute(
        "SELECT claim_lock, current_run_id FROM tasks WHERE id=?", (tid,)).fetchone()
    kb._append_event(
        conn, tid, "worker_bound",
        {"pid": sleeper.pid, "claim_lock": lock_row[0]},
        run_id=lock_row[1],
    )


def test_enforcement_kill_of_a_fenced_attempt_is_infrastructure(kanban_home, monkeypatch):
    """A card WITH max_runtime_seconds whose owner is still fenced when the enforcement clock
    runs out: the kill books as the wave's ``drained`` infrastructure — no failure increment
    (a wave longer than a wave-sized budget must not spend the card's failure budget)."""
    monkeypatch.setattr(kbd, "_owner_update_fenced", lambda: True)
    started = int(time.time()) - 700
    sleeper = subprocess.Popen(["sleep", "120"])
    try:
        sigterms = []
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="fenced", assignee="a")
            _claimed_sleeper(conn, tid, sleeper, started)
            assert kbd.enforce_max_runtime(conn, signal_fn=lambda p, s: sigterms.append((p, s))) == [tid]
            assert sigterms and sigterms[0][1] == signal.SIGTERM  # graceful first
            ev = conn.execute(
                "SELECT kind FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (tid,)).fetchone()
            run = conn.execute(
                "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (tid,)).fetchone()
            task = kb.get_task(conn, tid)
        assert ev["kind"] == "drained" and run["outcome"] == "drained"
        metadata = kb._json_dict(run["metadata"])
        assert metadata.get("infrastructure") is True and metadata.get("owner_drain") is True
        assert metadata.get("limit_seconds") == 600
        assert task.consecutive_failures == 0
        assert task.status == "ready"
        assert kbd.check_respawn_guard(conn, tid) == "infrastructure_cooldown"
    finally:
        sleeper.kill()
        sleeper.wait()


def test_two_fenced_enforcement_waves_never_reach_the_breaker(kanban_home, monkeypatch):
    """Two wave-length attempts under the same card (failure_limit=2) leave it ready: the
    enforcement path can't be the breaker's third accounting lane for a card that never ran."""
    monkeypatch.setattr(kbd, "_owner_update_fenced", lambda: True)
    started = int(time.time()) - 700
    sleeper = subprocess.Popen(["sleep", "180"])
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="waves", assignee="a")
        for _ in range(3):
            with kbc.connect() as conn:
                _claimed_sleeper(conn, tid, sleeper, started)
                assert kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None) == [tid]
            with kbc.connect() as conn:
                kb.recompute_ready(conn, failure_limit=2)
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
    finally:
        sleeper.kill()
        sleeper.wait()


def test_enforcement_kill_of_a_counted_bound_attempt_without_fenced_evidence(kanban_home, monkeypatch):
    """Controls: an overslow BOUND attempt (its working turn really started) with NO fence live
    and no remembered fence inside the attempt's life is booked ``timed_out`` — a card failure
    the breaker hears about (a slow card can never be excused by a wave it predates)."""
    monkeypatch.setattr(kbd, "_owner_update_fenced", lambda: False)
    started = int(time.time()) - 700
    sleeper = subprocess.Popen(["sleep", "120"])
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="slow", assignee="a")
            _claimed_sleeper(conn, tid, sleeper, started)
            _bound(conn, tid, sleeper)
            assert kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None) == [tid]
            run = conn.execute(
                "SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (tid,)).fetchone()
            assert run["outcome"] == "timed_out"
            assert kb.get_task(conn, tid).consecutive_failures == 1

        # A remembered fence OLDER than the attempt began never excuses the attempt either.
        with kbc.connect() as conn:
            tid2 = kb.create_task(conn, title="slow2", assignee="a")
            kbd._owner_fences_seen[:] = [started - 50]  # observed before this attempt existed
            _claimed_sleeper(conn, tid2, sleeper, started)
            _bound(conn, tid2, sleeper)
            assert kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None) == [tid2]
            run2 = conn.execute(
                "SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (tid2,)).fetchone()
            assert run2["outcome"] == "timed_out"
            assert kb.get_task(conn, tid2).consecutive_failures == 1
    finally:
        sleeper.kill()
        sleeper.wait()


def test_enforcement_never_bound_attempt_is_infrastructure_without_fences(kanban_home, monkeypatch):
    """A run that never recorded a ``worker_bound`` was still waiting on its owner when the
    enforcement clock ran out: the card never spent any of its budget, so the kill books as
    the wave's infrastructure with NO fence evidence at all."""""
    monkeypatch.setattr(kbd, "_owner_update_fenced", lambda: False)
    kbd._owner_fences_seen.clear()
    started = int(time.time()) - 700
    sleeper = subprocess.Popen(["sleep", "120"])
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="unbound", assignee="a")
            _claimed_sleeper(conn, tid, sleeper, started)
            assert kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None) == [tid]
            run = conn.execute(
                "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (tid,)).fetchone()
        assert run["outcome"] == "drained"
        assert kb._json_dict(run["metadata"]).get("infrastructure") is True
        assert kb.get_task(conn, tid).consecutive_failures == 0
    finally:
        sleeper.kill()
        sleeper.wait()


def test_enforcement_excuses_on_fence_memory_inside_the_attempt(kanban_home, monkeypatch):
    """The sweep is not running (its live probe goes quiet between ticks); a fence this same
    dispatcher already observed inside the attempt's life is enough to shape the kill."""
    monkeypatch.setattr(kbd, "_owner_update_fenced", lambda: False)
    started = int(time.time()) - 700
    sleeper = subprocess.Popen(["sleep", "120"])
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="memorized", assignee="a")
            _owner_fences_seen = [started + 10]  # the wave started 10s into this attempt
            kbd._owner_fences_seen[:] = _owner_fences_seen
            _claimed_sleeper(conn, tid, sleeper, started)
            assert kbd.enforce_max_runtime(conn, signal_fn=lambda *_: None) == [tid]
            run = conn.execute(
                "SELECT outcome, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
                (tid,)).fetchone()
        assert run["outcome"] == "drained"
        assert kb._json_dict(run["metadata"]).get("infrastructure") is True
        assert kb.get_task(conn, tid).consecutive_failures == 0
    finally:
        sleeper.kill()
        sleeper.wait()

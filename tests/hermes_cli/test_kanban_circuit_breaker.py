"""Executable tests for the global dispatch circuit breaker.

Covers the four acceptance behaviours of the provider-outage circuit breaker
(t_256ccf38):

1. Signature detection — a worker that exits cleanly (rc=0) within
   CIRCUIT_FAST_EXIT_SECONDS of spawn AND has a provider 4xx/5xx in its log
   trips the breaker (and releases the card WITHOUT counting a failure).
2. Pause — while the breaker is open the dispatcher spawns nothing.
3. Single alert — the "tripped" alert is queued exactly once.
4. Resume — a successful probe clears the flag and queues a "resumed" alert;
   a failed probe leaves the breaker paused.

Tests assert on worker *runtime* (reaped_at minus started_at), not wall-clock
at inspection time — this is the measure that makes detection reachable under
the dispatcher's 30s crash grace, so a regression back to "how long ago did it
start" fails here instead of in production.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import hermes_cli.kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def circuit_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (same as kanban_home)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claim_dead_worker(conn, task_id, pid, *, started_at_ago=1):
    """Claim ``task_id`` and point its run at a dead pid that started
    ``started_at_ago`` seconds ago. Returns the started_at used."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, task_id, claimer=f"{host}:w{pid}")
    started_at = int(time.time()) - started_at_ago
    conn.execute(
        "UPDATE tasks SET worker_pid=?, consecutive_failures=0, "
        "started_at=? WHERE id=?",
        (pid, started_at, task_id),
    )
    conn.commit()
    return started_at


def _write_outage_log(circuit_home, task_id, *, provider_error: str):
    """Write a worker log for ``task_id`` containing a provider HTTP error.

    The default-board log lives at ``<root>/kanban/logs/<task_id>.log``
    (mirrors ``worker_log_path``). The content must match BOTH the respawn
    auth-blocker pattern and the provider 4xx/5xx pattern for detection.
    """
    log_dir = circuit_home / "kanban" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{task_id}.log").write_text(provider_error, encoding="utf-8")


def _read_outbox(circuit_home):
    path = circuit_home / "kanban" / "circuit-outbox.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(errors="ignore") or "{}")
    msgs = data.get("messages", [])
    return msgs if isinstance(msgs, list) else []


def _exited_status(code: int) -> int:
    return code << 8


# ---------------------------------------------------------------------------
# 1. Signature detection -> trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fast_window", [True, False])
def test_fast_clean_exit_with_provider_error_trips_breaker(
    circuit_home, monkeypatch, fast_window,
):
    """A worker that dies fast+clean (rc=0) with a provider 4xx/5xx in its log
    trips the breaker and releases the card without counting a failure.

    ``fast_window=True`` uses a 1s runtime (reaped promptly after spawn):
    detection fires. ``fast_window=False`` runs the worker far longer than the
    fast-exit window (a genuinely busy worker that happened to die clean) and
    the breaker must NOT trip — the fingerprint is specifically the *spawn
    bounce*, not any clean exit.
    """
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="outage", assignee="a")
        pid = 50000
        started_at = _claim_dead_worker(conn, tid, pid)
        # Reap registry: (raw_status, reaped_at). Runtime = reaped_at - started_at.
        runtime = 1 if fast_window else 60
        _kb._record_worker_exit(
            pid, _exited_status(0),
        )
        # Point the freshly-recorded entry's reaped_at at started_at+runtime.
        _recent = _kb._recent_worker_exits.get(int(pid))
        raw = _recent[0] if _recent else _exited_status(0)
        _kb._recent_worker_exits[int(pid)] = (
            raw, float(started_at + runtime),
        )
        _write_outage_log(
            circuit_home, tid,
            provider_error=(
                "ERROR openrouter: HTTP 403 Forbidden — provider returned "
                "monthly key limit reached (billing quota)\n"
            ),
        )

        crashed = kbd.detect_crashed_workers(conn)
        status = _kb._circuit_status(conn)

        task = kb.get_task(conn, tid)
        if fast_window:
            assert tid not in crashed
            assert status["state"] == "open", \
                "fast+clean exit with provider 4xx/5xx must open the breaker"
            # Released to source WITHOUT counting a failure (like rate-limit).
            assert task.consecutive_failures == 0, \
                "an outage-backed spawn bounce must not count a failure"
            assert task.last_failure_error is not None
            assert "provider" in task.last_failure_error
        else:
            assert status["state"] == "closed", \
                "a long-running clean exit is NOT the outage fingerprint"
            assert tid in crashed, \
                "a genuinely-crashed (clean-exit) long worker routes normally"


def test_fast_clean_exit_without_provider_error_does_not_trip(
    circuit_home, monkeypatch,
):
    """Fast+clean exit alone (no provider 4xx/5xx in log) is a protocol
    violation, not an outage — the breaker stays closed and the card is
    handled by the normal clean-exit path."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="noerr", assignee="a")
        pid = 50001
        started_at = _claim_dead_worker(conn, tid, pid)
        _kb._record_worker_exit(pid, _exited_status(0))
        _recent = _kb._recent_worker_exits.get(int(pid))
        raw = _recent[0] if _recent else _exited_status(0)
        _kb._recent_worker_exits[int(pid)] = (raw, float(started_at + 1))
        # No log file at all — detection requires a grounded log signal.
        (circuit_home / "kanban" / "logs").mkdir(parents=True, exist_ok=True)

        kbd.detect_crashed_workers(conn)
        status = _kb._circuit_status(conn)
        assert status["state"] == "closed", \
            "clean exit WITHOUT a provider error is a protocol violation, " \
            "not an outage"


# ---------------------------------------------------------------------------
# 2. Pause — dispatch spawns nothing while open
# ---------------------------------------------------------------------------


def test_pause_spawns_nothing_while_circuit_open(
    circuit_home, monkeypatch,
):
    """While the breaker is open, a dispatch tick returns ``frozen_by_circuit``
    and spawns no workers — even with a ready card and a spawn fn standing by."""
    import hermes_cli.kanban_db as _kb

    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        # Open the breaker directly (simulate a prior detection).
        _kb._circuit_trip(conn, "test outage", board="default")
        tid = kb.create_task(conn, title="queued", assignee="alice")
        monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")

        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn, board="default")

        assert res.frozen_by_circuit, \
            "an open breaker must freeze the tick"
        assert not res.spawned, "circuit-open tick must not spawn"
        assert not spawns, "spawn fn must not be called while paused"
        # Card stays queued (ready) for the next tick after resume.
        task = kb.get_task(conn, tid)
        assert task.status == "ready", \
            "a queued card must wait, not be dropped"


# ---------------------------------------------------------------------------
# 3. Single alert
# ---------------------------------------------------------------------------


def test_trip_queues_exactly_one_alert(circuit_home, monkeypatch):
    """Tripping the breaker queues the ''tripped'' alert exactly once; a
    second outage detection of the same kind does not append a second alert."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    def _simulate_outage(tid, pid):
        started_at = _claim_dead_worker(conn, tid, pid)
        _kb._record_worker_exit(pid, _exited_status(0))
        _recent = _kb._recent_worker_exits.get(int(pid))
        raw = _recent[0] if _recent else _exited_status(0)
        _kb._recent_worker_exits[int(pid)] = (raw, float(started_at + 1))
        _write_outage_log(
            circuit_home, tid,
            provider_error=(
                "ERROR openrouter: HTTP 403 Forbidden — provider returned "
                "monthly key limit reached (billing quota)\n"
            ),
        )

    with kbc.connect() as conn:
        # Two separate outage detections, back to back. The FIRST opens the
        # breaker and queues the 'tripped' alert; the second (same kind, same
        # board) must NOT append a duplicate.
        for n in range(2):
            tid = kb.create_task(conn, title=f"outage{n}", assignee="a")
            _simulate_outage(tid, 50000 + n)
            crashed = kbd.detect_crashed_workers(conn, board="default")
            assert tid not in crashed, \
                "an outage-backed spawn bounce must not count as a crash"

        msgs = _read_outbox(circuit_home)
        tripped = [m for m in msgs if m.get("kind") == "tripped"]
        assert len(tripped) == 1, \
            f"'tripped' alert must be queued exactly once, got {len(tripped)}"


def test_paused_probe_failure_keeps_breaker_open_and_no_resume_alert(
    circuit_home, monkeypatch,
):
    """A failed probe while paused leaves the breaker open and queues no
    ''resumed'' alert — the outage is still in force."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_CIRCUIT_PROBE_INTERVAL_SECONDS", "1")

    with kbc.connect() as conn:
        _kb._circuit_trip(conn, "outage", board="default")
        # Force due-for-probe: clear last_probe_at.
        _kb._ensure_dispatch_circuit_row(conn)
        conn.execute(
            "UPDATE dispatch_circuit SET last_probe_at = NULL WHERE id = ?",
            (_kb.CIRCUIT_ROW_ID,),
        )
        conn.commit()

        res = _kb._circuit_dispatch_paused(
            conn, circuit_probe_fn=lambda: False, board="default",
        )
        status = _kb._circuit_status(conn)
        msgs = _read_outbox(circuit_home)

        assert res.frozen_by_circuit
        assert status["state"] == "open", \
            "a failed probe must leave the breaker open"
        assert status["probe_count"] >= 1
        resumed = [m for m in msgs if m.get("kind") == "resumed"]
        assert not resumed, \
            "no 'resumed' alert while the probe still fails"


def test_successful_probe_closes_breaker_and_queues_resume_alert(
    circuit_home, monkeypatch,
):
    """A successful probe closes the breaker and queues one ''resumed'' alert;
    the resumed alert is distinct from a ''tripped'' alert."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_CIRCUIT_PROBE_INTERVAL_SECONDS", "1")

    with kbc.connect() as conn:
        _kb._circuit_trip(conn, "outage", board="default")
        _kb._ensure_dispatch_circuit_row(conn)
        conn.execute(
            "UPDATE dispatch_circuit SET last_probe_at = NULL WHERE id = ?",
            (_kb.CIRCUIT_ROW_ID,),
        )
        conn.commit()

        res = _kb._circuit_dispatch_paused(
            conn, circuit_probe_fn=lambda: True, board="default",
        )
        status = _kb._circuit_status(conn)
        msgs = _read_outbox(circuit_home)

        assert res.frozen_by_circuit, \
            "the tick during which the probe succeeds still reports paused"
        assert status["state"] == "closed", \
            "a successful probe must close the breaker"
        resumed = [m for m in msgs if m.get("kind") == "resumed"]
        assert len(resumed) == 1, \
            f"'resumed' alert must be queued once, got {len(resumed)}"


def test_resume_then_dispatch_spawns_pending_card(circuit_home, monkeypatch):
    """End-to-end: after the breaker closes, the next tick spawns the pending
    card normally (no longer frozen_by_circuit)."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_CIRCUIT_PROBE_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    # Make "alice" a spawnable profile — otherwise the ready loop buckets the
    # pending card as nonspawnable and never calls spawn_fn (the circuit
    # breaker is fine; the card just wouldn't spawn for the wrong reason).
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda _name: True,
    )
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        _kb._circuit_trip(conn, "outage", board="default")
        tid = kb.create_task(conn, title="pending", assignee="alice")

        # Paused tick: probe succeeds, closes the breaker.
        _kb._ensure_dispatch_circuit_row(conn)
        conn.execute(
            "UPDATE dispatch_circuit SET last_probe_at = NULL WHERE id = ?",
            (_kb.CIRCUIT_ROW_ID,),
        )
        conn.commit()

        # First tick: breaker open. Probe suture returns True so the probe
        # closes the circuit — but THIS tick still reports frozen (the probe
        # outcome is only visible to the NEXT tick).
        paused = kbd.dispatch_once(
            conn, spawn_fn=fake_spawn, board="default",
            circuit_probe_fn=lambda: True,
        )
        status_after_paused = _kb._circuit_status(conn)

        # The breaker is now closed (probe succeeded during the paused tick),
        # so the NEXT tick spawns the pending card normally.
        resumed = kbd.dispatch_once(
            conn, spawn_fn=fake_spawn, board="default",
            circuit_probe_fn=lambda: True,
        )
        task = kb.get_task(conn, tid)

        assert paused.frozen_by_circuit, \
            "the tick during which the probe succeeds still reports paused"
        assert not paused.spawned, "no spawn while frozen"
        assert status_after_paused["state"] == "closed", \
            "the probe that ran during the paused tick closes the breaker"
        assert not resumed.frozen_by_circuit, \
            "once closed, dispatch is no longer frozen"
        assert tid in resumed.spawned or tid in spawns, \
            "the pending card spawns in the tick after the breaker closes"
        assert task is not None
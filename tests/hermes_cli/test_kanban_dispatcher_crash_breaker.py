"""Crash-loop circuit breaker tests.

Covers ``evaluate_crash_breaker``, ``is_assignee_quarantined``, and the
dispatcher integration that quarantines a runaway assignee, blocks
subsequent dispatch, and probes after cooldown.

The breaker uses synthesized timestamps via ``monkeypatch.setattr(time,
"time", ...)`` so tests don't have to wait wall-clock seconds. Every
test runs with the ``all_assignees_spawnable`` fixture from
``tests/hermes_cli/conftest.py`` so synthetic assignees ("alice") aren't
filtered out by the profile-exists guard.
"""

from __future__ import annotations

import json
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


def _emit_crash(conn, task_id, at_epoch=None, *, error="boom"):
    """Synthesize a crashed event the same shape detect_crashed_workers
    would write — assignee comes from the task row, so the join inside
    evaluate_crash_breaker picks it up."""
    with kb.write_txn(conn):
        kb._append_event(
            conn, task_id, "crashed",
            {"pid": 123, "claimer": "host:1", "error": error},
        )
        if at_epoch is not None:
            conn.execute(
                "UPDATE task_events SET created_at = ? "
                "WHERE id = (SELECT MAX(id) FROM task_events WHERE task_id = ?)",
                (int(at_epoch), task_id),
            )


# ---------------------------------------------------------------------------
# evaluate_crash_breaker
# ---------------------------------------------------------------------------

def test_breaker_trips_on_nth_crash(kanban_home, all_assignees_spawnable):
    conn = kb.connect()
    try:
        now = int(time.time())
        t1 = kb.create_task(conn, title="a", assignee="alice")
        t2 = kb.create_task(conn, title="b", assignee="alice")
        t3 = kb.create_task(conn, title="c", assignee="alice")
        _emit_crash(conn, t1, at_epoch=now - 100)
        _emit_crash(conn, t2, at_epoch=now - 50)
        _emit_crash(conn, t3, at_epoch=now - 10, error="ModuleNotFoundError: hermes_cli")
        tripped = kb.evaluate_crash_breaker(
            conn, max_crashes=3, window_seconds=120,
            cooldown_seconds=300, max_cooldown_seconds=3600, now=now,
        )
        assert tripped == ["alice"]
        row = conn.execute(
            "SELECT * FROM kanban_quarantine WHERE assignee='alice'"
        ).fetchone()
        assert row is not None
        assert int(row["cooldown_until"]) == now + 300
        assert int(row["trip_count"]) == 1
        # Most-recent crashed task gets the quarantined event
        ev = conn.execute(
            "SELECT * FROM task_events WHERE task_id=? AND kind='quarantined'",
            (t3,),
        ).fetchone()
        assert ev is not None
        payload = json.loads(ev["payload"])
        assert payload["assignee"] == "alice"
        assert payload["crashes"] == 3
        assert "ModuleNotFoundError" in payload["last_reason"]
    finally:
        conn.close()


def test_breaker_does_not_trip_below_threshold(kanban_home, all_assignees_spawnable):
    conn = kb.connect()
    try:
        now = int(time.time())
        t1 = kb.create_task(conn, title="a", assignee="alice")
        t2 = kb.create_task(conn, title="b", assignee="alice")
        _emit_crash(conn, t1, at_epoch=now - 50)
        _emit_crash(conn, t2, at_epoch=now - 10)
        tripped = kb.evaluate_crash_breaker(
            conn, max_crashes=3, window_seconds=120,
            cooldown_seconds=300, max_cooldown_seconds=3600, now=now,
        )
        assert tripped == []
        assert conn.execute("SELECT COUNT(*) FROM kanban_quarantine").fetchone()[0] == 0
    finally:
        conn.close()


def test_breaker_does_not_trip_outside_window(kanban_home, all_assignees_spawnable):
    conn = kb.connect()
    try:
        now = int(time.time())
        t1 = kb.create_task(conn, title="a", assignee="alice")
        t2 = kb.create_task(conn, title="b", assignee="alice")
        t3 = kb.create_task(conn, title="c", assignee="alice")
        # All older than the 120s window
        _emit_crash(conn, t1, at_epoch=now - 500)
        _emit_crash(conn, t2, at_epoch=now - 300)
        _emit_crash(conn, t3, at_epoch=now - 200)
        tripped = kb.evaluate_crash_breaker(
            conn, max_crashes=3, window_seconds=120,
            cooldown_seconds=300, max_cooldown_seconds=3600, now=now,
        )
        assert tripped == []
    finally:
        conn.close()


def test_evaluate_does_not_double_trip_while_cooldown_active(
    kanban_home, all_assignees_spawnable,
):
    """Re-running evaluate inside the cooldown window must be a no-op —
    we already notified on the original trip."""
    conn = kb.connect()
    try:
        now = int(time.time())
        for i in range(3):
            t = kb.create_task(conn, title=f"t{i}", assignee="alice")
            _emit_crash(conn, t, at_epoch=now - (10 + i))
        kb.evaluate_crash_breaker(
            conn, max_crashes=3, window_seconds=120,
            cooldown_seconds=300, max_cooldown_seconds=3600, now=now,
        )
        # Second pass inside cooldown — must NOT emit a new quarantined event.
        ev_count_before = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind='quarantined'"
        ).fetchone()[0]
        tripped2 = kb.evaluate_crash_breaker(
            conn, max_crashes=3, window_seconds=120,
            cooldown_seconds=300, max_cooldown_seconds=3600, now=now + 30,
        )
        ev_count_after = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE kind='quarantined'"
        ).fetchone()[0]
        assert tripped2 == []
        assert ev_count_after == ev_count_before
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Dispatcher integration
# ---------------------------------------------------------------------------

def test_quarantine_prevents_dispatch(kanban_home, all_assignees_spawnable):
    conn = kb.connect()
    try:
        now = int(time.time())
        # Pre-quarantine alice manually.
        conn.execute(
            "INSERT INTO kanban_quarantine "
            "(assignee, cooldown_until, trip_count, last_trip_at, "
            " last_reason, probe_in_flight) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            ("alice", now + 600, 1, now, "broken venv"),
        )
        kb.create_task(conn, title="future task", assignee="alice")
        spawn_calls: list = []

        def _spawn(task, ws, **_kw):
            spawn_calls.append(task.id)
            return 12345

        res = kb.dispatch_once(
            conn, spawn_fn=_spawn, crash_breaker_enabled=True,
        )
        assert spawn_calls == []
        assert len(res.quarantine_blocked) == 1
        assert res.quarantine_blocked[0][1] == "alice"
    finally:
        conn.close()


def test_quarantine_blocks_review_column_too(
    kanban_home, all_assignees_spawnable,
):
    conn = kb.connect()
    try:
        now = int(time.time())
        conn.execute(
            "INSERT INTO kanban_quarantine "
            "(assignee, cooldown_until, trip_count, last_trip_at, "
            " probe_in_flight) VALUES (?, ?, 1, ?, 0)",
            ("reviewer-bot", now + 600, now),
        )
        tid = kb.create_task(conn, title="needs review", assignee="reviewer-bot")
        # Force the task into review.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='review' WHERE id=?", (tid,),
            )
        spawn_calls: list = []

        def _spawn(task, ws, **_kw):
            spawn_calls.append(task.id)
            return 999

        res = kb.dispatch_once(
            conn, spawn_fn=_spawn, crash_breaker_enabled=True,
        )
        assert spawn_calls == []
        assert any(
            t == tid and a == "reviewer-bot"
            for t, a in res.quarantine_blocked
        )
    finally:
        conn.close()


def test_cooldown_lifts_and_only_one_probe_spawns_per_tick(
    kanban_home, all_assignees_spawnable,
):
    conn = kb.connect()
    try:
        now = int(time.time())
        # Cooldown elapsed (cooldown_until in the past).
        conn.execute(
            "INSERT INTO kanban_quarantine "
            "(assignee, cooldown_until, trip_count, last_trip_at, "
            " probe_in_flight) VALUES (?, ?, 1, ?, 0)",
            ("alice", now - 10, now - 310),
        )
        tids = [kb.create_task(conn, title=f"t{i}", assignee="alice") for i in range(3)]
        spawn_calls: list = []

        def _spawn(task, ws, **_kw):
            spawn_calls.append(task.id)
            return 1000 + len(spawn_calls)

        res = kb.dispatch_once(
            conn, spawn_fn=_spawn, crash_breaker_enabled=True,
        )
        # Exactly one probe spawn, remaining two blocked.
        assert len(spawn_calls) == 1
        assert len(res.quarantine_probes) == 1
        assert res.quarantine_probes[0][1] == "alice"
        assert res.quarantine_probes[0][0] == spawn_calls[0]
        assert len(res.quarantine_blocked) == 2
        # All blocked entries are for alice tasks that weren't the probe.
        blocked_tids = {t for t, _a in res.quarantine_blocked}
        assert spawn_calls[0] not in blocked_tids
        assert blocked_tids.issubset(set(tids))
        # probe_task_id was recorded.
        row = conn.execute(
            "SELECT probe_task_id, probe_in_flight FROM kanban_quarantine "
            "WHERE assignee='alice'"
        ).fetchone()
        assert row["probe_task_id"] == spawn_calls[0]
        assert int(row["probe_in_flight"]) == 1
    finally:
        conn.close()


def test_probe_success_clears_quarantine(kanban_home, all_assignees_spawnable):
    conn = kb.connect()
    try:
        now = int(time.time())
        tid = kb.create_task(conn, title="probe", assignee="alice")
        conn.execute(
            "INSERT INTO kanban_quarantine "
            "(assignee, cooldown_until, trip_count, last_trip_at, "
            " probe_in_flight, probe_task_id) VALUES (?, ?, 1, ?, 1, ?)",
            ("alice", now - 10, now - 310, tid),
        )
        # Move task to running, then complete.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='running' WHERE id=?", (tid,),
            )
        kb.complete_task(conn, tid, summary="probe ok")
        row = conn.execute(
            "SELECT * FROM kanban_quarantine WHERE assignee='alice'"
        ).fetchone()
        assert row is None
    finally:
        conn.close()


def test_probe_crash_extends_cooldown_with_backoff(
    kanban_home, all_assignees_spawnable,
):
    """When the probe task crashes, extend_assignee_quarantine doubles
    the cooldown and bumps trip_count."""
    conn = kb.connect()
    try:
        now = int(time.time())
        tid = kb.create_task(conn, title="probe", assignee="alice")
        conn.execute(
            "INSERT INTO kanban_quarantine "
            "(assignee, cooldown_until, trip_count, last_trip_at, "
            " probe_in_flight, probe_task_id) VALUES (?, ?, 1, ?, 1, ?)",
            ("alice", now - 10, now - 310, tid),
        )
        new_until = kb.extend_assignee_quarantine(
            conn, "alice",
            cooldown_seconds=300, max_cooldown_seconds=3600,
            reason="probe crashed: ImportError: foo",
            now=now,
        )
        assert new_until is not None
        # trip_count 1 -> 2 -> cooldown 300 * 2^(2-1) = 600
        assert new_until == now + 600
        row = conn.execute(
            "SELECT trip_count, probe_in_flight, probe_task_id, last_reason "
            "FROM kanban_quarantine WHERE assignee='alice'"
        ).fetchone()
        assert int(row["trip_count"]) == 2
        assert int(row["probe_in_flight"]) == 0
        assert row["probe_task_id"] is None
        assert "ImportError" in row["last_reason"]

        # Once more: cooldown 300 * 2^(3-1) = 1200
        new_until2 = kb.extend_assignee_quarantine(
            conn, "alice",
            cooldown_seconds=300, max_cooldown_seconds=3600,
            now=now,
        )
        assert new_until2 == now + 1200

        # Eventually capped at max_cooldown_seconds.
        # trip_count -> 4 -> 300 * 2^3 = 2400; -> 5 -> 4800 capped at 3600.
        kb.extend_assignee_quarantine(
            conn, "alice", cooldown_seconds=300, max_cooldown_seconds=3600, now=now,
        )
        capped = kb.extend_assignee_quarantine(
            conn, "alice", cooldown_seconds=300, max_cooldown_seconds=3600, now=now,
        )
        assert capped == now + 3600
    finally:
        conn.close()


def test_existing_dispatch_unchanged_when_breaker_disabled(
    kanban_home, all_assignees_spawnable,
):
    """When ``crash_breaker_enabled=False`` (the dispatch_once default),
    no quarantine state is consulted even if a row exists from a prior
    enabled run."""
    conn = kb.connect()
    try:
        now = int(time.time())
        conn.execute(
            "INSERT INTO kanban_quarantine "
            "(assignee, cooldown_until, trip_count, last_trip_at, "
            " probe_in_flight) VALUES (?, ?, ?, ?, 0)",
            ("alice", now + 600, 1, now),
        )
        tid = kb.create_task(conn, title="t", assignee="alice")
        spawn_calls: list = []

        def _spawn(task, ws, **_kw):
            spawn_calls.append(task.id)
            return 4242

        res = kb.dispatch_once(conn, spawn_fn=_spawn)  # breaker NOT enabled
        assert spawn_calls == [tid]
        assert res.quarantine_blocked == []
        assert res.quarantined == []
    finally:
        conn.close()

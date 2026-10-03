"""Issue #125239: at-cap ticks are capacity, not a stuck dispatcher.

The gateway/standalone "dispatcher stuck" telemetry counts every tick with
a non-empty ready queue and zero spawns as bad - even when the tick could
not spawn by design (global/per-profile concurrency cap satisfied, sibling
dispatcher holding the lock, critical memory pressure). With
``max_in_progress: 1`` that cries wolf on every tick while a worker runs.

These tests drive real ``dispatch_once`` ticks and assert the shared
``capacity_hold`` guard names the capacity hold, so both telemetry sites
can reset ``bad_ticks`` instead of warning.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def hermetic_memory(monkeypatch):
    """Detach the tick from the host's live memory pressure reading."""
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda *a, **k: "unknown")


def _fake_spawn_factory(spawns):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    return fake_spawn


def test_at_cap_tick_reports_capacity_not_stuck(
    kanban_home, all_assignees_spawnable, hermetic_memory,
):
    """Acceptance 1: cap=1 with one worker running must not count as stuck."""
    spawns = []
    with kbc.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        assert kb.claim_task(conn, running) is not None
        kb.create_task(conn, title="waiting", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )
        pending = kbd.has_spawnable_ready(conn)
    assert not res.spawned and not spawns
    assert pending, "watcher probe sees ready work while capped (the wolf cry)"
    hold = kbd.capacity_hold([res])
    assert hold and "at_cap" in hold, f"cap hold unnamed: {hold!r}"
    assert "at_cap" in kbd.describe_suppression([res])


def test_per_profile_cap_reports_capacity_not_stuck(
    kanban_home, all_assignees_spawnable, hermetic_memory,
):
    """Same, for max_in_progress_per_profile: profile busy, retry later."""
    spawns = []
    with kbc.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        assert kb.claim_task(conn, running) is not None
        kb.create_task(conn, title="waiting", assignee="alice")
        res = kbd.dispatch_once(
            conn,
            spawn_fn=_fake_spawn_factory(spawns),
            max_in_progress_per_profile=1,
        )
        pending = kbd.has_spawnable_ready(conn)
    assert not res.spawned and not spawns
    assert res.skipped_per_profile_capped
    assert pending
    hold = kbd.capacity_hold([res])
    assert hold and "per_profile_capped" in hold, f"hold unnamed: {hold!r}"
    assert "per_profile_capped" in kbd.describe_suppression([res])


def test_respawn_guarded_tick_is_named_but_not_capacity(
    kanban_home, all_assignees_spawnable, hermetic_memory,
):
    """Guard-held work (e.g. recent_success, the ~1h cadence) is named in the
    warning but must NOT reset the stuck counter: a co-existing genuinely
    broken task would otherwise be masked."""
    now = int(time.time())
    spawns = []
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="redo", assignee="alice")
        conn.execute(
            "INSERT INTO task_runs (task_id, status, started_at, ended_at, outcome)"
            " VALUES (?, 'done', ?, ?, 'completed')",
            (tid, now - 100, now - 100),
        )
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid) == "recent_success"
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns))
    assert not res.spawned and not spawns
    assert dict(res.respawn_guarded) == {tid: "recent_success"}
    assert kbd.capacity_hold([res]) == ""
    assert "recent_success" in kbd.describe_suppression([res])


def test_empty_result_is_genuinely_stuck(
    kanban_home, all_assignees_spawnable, hermetic_memory,
):
    """No holds at all: the stuck counter must still fire."""
    assert kbd.capacity_hold([]) == ""
    assert kbd.capacity_hold([kbd.DispatchResult()]) == ""
    assert kbd.describe_suppression([kbd.DispatchResult()]) == ""

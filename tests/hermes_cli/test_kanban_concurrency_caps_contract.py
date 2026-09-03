"""Contract/regression tests for the kanban concurrency-cap naming trap.

BUI-942 item 3. ``kanban.max_spawn`` and ``kanban.max_in_progress`` are two
different knobs that read like synonyms:

* ``max_spawn`` is a **live concurrency ceiling** — the maximum number of
  workers running at any instant, NOT a per-tick launch budget. It counts
  tasks already in ``status='running'`` against the limit.
* ``max_in_progress`` is also a live cap on running tasks.

When BOTH are set, the effective global worker cap is
``min(max_spawn, max_in_progress)`` (e.g. ``min(2, 3) == 2``). These tests
pin that contract so a future refactor can't silently turn either knob into
a per-tick budget again.
"""

from __future__ import annotations

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


def _make_ready(conn, n):
    return [
        kb.create_task(conn, title=f"t{i}", assignee=f"prof{i}")
        for i in range(n)
    ]


def test_effective_cap_is_min_when_max_spawn_binds(
    kanban_home, all_assignees_spawnable
):
    """min(max_spawn=2, max_in_progress=3) == 2 workers this tick."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        _make_ready(conn, 5)
        res = kb.dispatch_once(
            conn, spawn_fn=fake_spawn, max_spawn=2, max_in_progress=3
        )
    assert len(res.spawned) == 2
    assert len(spawns) == 2


def test_dry_run_reports_only_capped_would_be_spawns(
    kanban_home, all_assignees_spawnable
):
    """A dry_run pass must honour max_spawn in its would-be-spawn report:
    max_spawn=2 with five ready tasks reports exactly two, not all five. The
    dry-run counter must advance against the cap just like a real spawn."""
    with kb.connect() as conn:
        _make_ready(conn, 5)
        res = kb.dispatch_once(conn, dry_run=True, max_spawn=2)
    assert len(res.spawned) == 2
    # dry_run must not mutate the board — all five stay ready.
    with kb.connect() as conn:
        ready = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'ready'"
        ).fetchone()[0]
    assert ready == 5


def test_effective_cap_is_min_when_max_in_progress_binds(
    kanban_home, all_assignees_spawnable
):
    """min(max_spawn=3, max_in_progress=2) == 2 workers this tick."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        _make_ready(conn, 5)
        res = kb.dispatch_once(
            conn, spawn_fn=fake_spawn, max_spawn=3, max_in_progress=2
        )
    assert len(res.spawned) == 2
    assert len(spawns) == 2


def test_max_spawn_is_live_ceiling_not_per_tick_budget(
    kanban_home, all_assignees_spawnable
):
    """Already-running workers count against max_spawn, so a busy board at the
    ceiling spawns nothing more this tick (concurrency ceiling, not budget)."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        running_a = kb.create_task(conn, title="ra", assignee="a")
        running_b = kb.create_task(conn, title="rb", assignee="b")
        kb.create_task(conn, title="ready", assignee="c")
        kb.claim_task(conn, running_a)
        kb.claim_task(conn, running_b)
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=2)
    assert res.spawned == []
    assert spawns == []


def test_running_tasks_count_against_the_live_ceiling(
    kanban_home, all_assignees_spawnable
):
    """Both knobs are *live* ceilings on concurrent workers, so an already-
    running worker consumes exactly ONE slot under the combined ceiling. With
    one worker running and both caps at 3, the effective live cap is
    min(3, 3) = 3, so this tick tops up TWO more (reaching 3 running) — not a
    fresh per-tick batch of 3, and not just 1 (the old double-subtraction bug,
    BUI-942 item 3, which counted the running task against both knobs)."""
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        running = kb.create_task(conn, title="run", assignee="a")
        kb.claim_task(conn, running)
        _make_ready(conn, 5)
        res = kb.dispatch_once(
            conn, spawn_fn=fake_spawn, max_spawn=3, max_in_progress=3
        )
    # One already running + two topped up == 3 running == min(3, 3) ceiling.
    assert len(res.spawned) == 2

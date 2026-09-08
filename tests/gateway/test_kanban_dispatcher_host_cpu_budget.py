"""Host-cycle-shared CPU admission budget for ``_KanbanDispatcher.tick_once``
(t_4008d306 rework).

The rejected candidate (68c83f0) sampled/classified CPU pressure separately
inside each board's own ``dispatch_once`` call. The production gateway calls
``dispatch_once`` once per board every tick
(``_KanbanDispatcher.tick_once`` -> ``tick_once_for_board``), so a per-board
reading let every board independently admit its own "at most one" elevated
worker: three boards -> three spawns in one host cycle, not the intended
host-wide cap of one.

This module proves the fix directly against a REAL ``_KanbanDispatcher`` over
three real boards/DBs (no mocking of ``tick_once`` or ``dispatch_once``
internals beyond the CPU sample and the process-spawning seam):

1. Elevated pressure: the SUM of new spawns across all boards in ONE
   ``tick_once()`` call is <= 1.
2. Critical pressure: the SUM of new spawns is exactly 0, while every board's
   reclaim/promotion bookkeeping still runs and affected tasks stay ready
   (deferred, not dropped).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from gateway.kanban_watchers_dispatcher import _DispatcherSettings, _KanbanDispatcher

BOARDS = ("alpha-board", "beta-board", "gamma-board")


def _cpu_sample(level: str) -> dict:
    if level == "critical":
        return {"load1": 40.0, "cpu_count": 8, "psi_some_avg60": 76.64}
    if level == "elevated":
        return {"load1": 9.0, "cpu_count": 8, "psi_some_avg60": 25.0}
    return {"load1": 1.0, "cpu_count": 8, "psi_some_avg60": 2.0}


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with three real boards, each holding a ready
    task, a stale (unheartbeated, never-pid-set) running task, and a
    done-parent/todo-child pair to exercise reclaim + promotion."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    board_state = {}
    for slug in BOARDS:
        kb.create_board(slug)
        with kbc.connect(board=slug) as conn:
            ready_id = kb.create_task(conn, title="ready", assignee="alice")
            stale_id = kb.create_task(conn, title="stale-running", assignee="alice")
            kb.claim_task(conn, stale_id, ttl_seconds=1)
            # ``ttl_seconds`` is clamped to a minimum of 1s by
            # ``_resolve_claim_ttl_seconds``; force the claim into the past
            # directly so ``release_stale_claims`` treats it as expired
            # without relying on real wall-clock sleeps in this test.
            conn.execute(
                "UPDATE tasks SET claim_expires = ? WHERE id = ?",
                (int(time.time()) - 10_000, stale_id),
            )
            parent_id = kb.create_task(conn, title="parent", assignee="alice")
            child_id = kb.create_task(conn, title="child", assignee="alice", parents=[parent_id])
            conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent_id,))
        board_state[slug] = {"ready": ready_id, "stale": stale_id, "child": child_id}
    return board_state


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Pretend every assignee maps to a real Hermes profile (mirrors
    ``tests/hermes_cli/conftest.py``'s fixture, which isn't visible to
    ``tests/gateway/`` — pytest conftest fixtures don't cross sibling dirs)."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


@pytest.fixture
def dispatcher(all_assignees_spawnable):
    settings = _DispatcherSettings(
        interval=60.0,
        max_spawn=None,
        max_in_progress=None,
        failure_limit=5,
        stale_timeout_seconds=0,
        reconcile_orphans=True,
        default_assignee=None,
        max_in_progress_per_profile=None,
    )
    return _KanbanDispatcher(kb, settings)


def _total_spawned(results) -> int:
    return sum(len(result.spawned) for _slug, result in results if result is not None)


def test_tick_once_elevated_cpu_pressure_spawns_at_most_one_across_all_boards(
    kanban_home, dispatcher, monkeypatch,
):
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _cpu_sample("elevated"))
    spawned = []

    def fake_default_spawn(task, workspace, *, board=None):
        spawned.append((board, task.id))
        return 4242

    monkeypatch.setattr(kbd, "_default_spawn", fake_default_spawn)

    results = dispatcher.tick_once()

    assert _total_spawned(results) <= 1
    assert len(spawned) <= 1
    # Every board's own reading agreed (one host-cycle sample), so a slot
    # was spent by at most the FIRST board that had spawnable work.
    for _slug, result in results:
        if result is not None:
            assert result.cpu_pressure in ("elevated", None)


def test_tick_once_elevated_cpu_pressure_with_pid_persistence_fault_still_launches_at_most_one(
    kanban_home, dispatcher, monkeypatch,
):
    """Regression for the t_194e18d6 review finding: the shared elevated slot
    must be reserved BEFORE ``spawn_fn`` is invoked, not after PID persistence
    succeeds. Inject ``_set_worker_pid`` raising ``OSError`` after ``spawn_fn``
    returns a real PID (4242) -- a real ambiguous/post-launch failure, since
    the child process was already created. The actual spawn CALLBACK count
    (not ``DispatchResult.spawned``, which undercounts this exact failure) must
    stay <= 1 across the whole cycle, and later boards must still reclaim and
    promote their stale/blocked work."""
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _cpu_sample("elevated"))
    launch_calls = []

    def fake_default_spawn(task, workspace, *, board=None):
        launch_calls.append((board, task.id))
        return 4242

    monkeypatch.setattr(kbd, "_default_spawn", fake_default_spawn)

    def failing_set_worker_pid(conn, task_id, pid):
        raise OSError("simulated PID persistence fault")

    monkeypatch.setattr(kbd, "_set_worker_pid", failing_set_worker_pid)

    results = dispatcher.tick_once()
    results_by_slug = dict(results)

    # The bug: DispatchResult.spawned undercounts here, since the failing
    # board's attempt raises inside the try/except and is recorded as a
    # failure, not a spawn. The invariant under test is on the real callback
    # invocation count, which is what actually matters for host load.
    assert len(launch_calls) <= 1, launch_calls
    assert _total_spawned(results) <= len(launch_calls)

    # Later boards (after the one that consumed the shared slot, whether its
    # attempt "succeeded" from DispatchResult's point of view or not) must
    # still run reclaim/promotion and keep their affected tasks retained.
    for slug in BOARDS:
        result = results_by_slug[slug]
        assert result is not None
        assert result.reclaimed >= 1
        assert result.promoted >= 1
        with kbc.connect(board=slug) as conn:
            for task_id in (kanban_home[slug]["stale"], kanban_home[slug]["child"]):
                row = kb.get_task(conn, task_id)
                assert row is not None and row.status == "ready", (slug, task_id, row and row.status)


def test_tick_once_critical_cpu_pressure_spawns_nothing_but_still_reclaims_and_promotes(
    kanban_home, dispatcher, monkeypatch,
):
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _cpu_sample("critical"))

    def must_not_spawn(task, workspace, *, board=None):
        raise AssertionError(f"spawn_fn must not be called under critical CPU pressure (board={board})")

    monkeypatch.setattr(kbd, "_default_spawn", must_not_spawn)

    results = dispatcher.tick_once()
    results_by_slug = dict(results)

    assert _total_spawned(results) == 0
    for slug in BOARDS:
        result = results_by_slug[slug]
        assert result is not None
        assert result.cpu_pressure == "critical"
        assert result.spawned == []
        assert result.reclaimed >= 1
        assert result.promoted >= 1
        with kbc.connect(board=slug) as conn:
            for task_id in (kanban_home[slug]["ready"], kanban_home[slug]["stale"], kanban_home[slug]["child"]):
                row = kb.get_task(conn, task_id)
                assert row is not None and row.status == "ready", (slug, task_id, row and row.status)

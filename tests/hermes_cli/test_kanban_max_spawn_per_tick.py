"""``max_spawn`` is a per-tick budget on NEW workers, not a concurrency cap.

Regression for the drain dead zone: with one worker already ``running`` on the
board, ``hermes kanban dispatch --max 1`` reported ``spawned: []`` AND
``respawn_guarded: []``. ``_tick_spawn_budget`` bailed on
``running_count >= max_spawn`` before the lane loops built a candidate list, so
the pass neither spawned nor recorded why, and ready cards sat idle for hours.
"""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    kb.init_db()
    return home


def _spawn_factory(spawns):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


def _running(conn, title="already-running"):
    tid = kb.create_task(conn, title=title, assignee="alice")
    assert kb.claim_task(conn, tid) is not None
    return tid


def test_max_spawn_one_still_spawns_with_a_worker_already_running(
    kanban_home, all_assignees_spawnable,
):
    """The dead window: 1 running + --max 1 must still spawn one NEW worker."""
    spawns: list = []
    with kbc.connect() as conn:
        _running(conn)
        ready_id = kb.create_task(conn, title="waiting", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_spawn_factory(spawns), max_spawn=1,
        )

    assert [task_id for task_id, *_ in res.spawned] == [ready_id]
    assert spawns == [ready_id]


def test_max_spawn_is_a_per_pass_budget(kanban_home, all_assignees_spawnable):
    """``--max 2`` means two NEW workers this pass, running count aside."""
    spawns: list = []
    with kbc.connect() as conn:
        _running(conn)
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_spawn_factory(spawns), max_spawn=2,
        )

    assert len(res.spawned) == 2
    assert len(spawns) == 2


def test_max_in_progress_still_caps_concurrency(
    kanban_home, all_assignees_spawnable,
):
    """``max_in_progress`` remains the concurrency ceiling."""
    spawns: list = []
    with kbc.connect() as conn:
        _running(conn)
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_spawn_factory(spawns),
            max_spawn=5, max_in_progress=2,
        )

    # 1 already running against a host cap of 2 → exactly one new worker.
    assert len(res.spawned) == 1


def test_max_spawn_zero_still_spawns_nothing(kanban_home, all_assignees_spawnable):
    """``max_spawn=0`` stays the "do not spawn" spelling (evals rely on it)."""
    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="a", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_spawn_factory(spawns), max_spawn=0,
        )

    assert not res.spawned and not spawns


def test_max_spawn_ignores_other_boards_running_workers(
    kanban_home, all_assignees_spawnable,
):
    """A worker on another board never eats this board's per-pass budget."""
    kb.create_board("second")
    with kbc.connect(board="second") as conn:
        busy = kb.create_task(conn, title="busy", assignee="alice")
        assert kb.claim_task(conn, busy) is not None

    spawns: list = []
    with kbc.connect() as conn:
        kb.create_task(conn, title="a", assignee="alice")
        res = kbd.dispatch_once(
            conn, spawn_fn=_spawn_factory(spawns), max_spawn=1,
        )

    assert len(res.spawned) == 1
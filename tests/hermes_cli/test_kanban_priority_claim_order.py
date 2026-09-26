"""Regression test for #edd065da — dispatcher ignored the P0-highest priority
convention.

``_lane_rows`` used ``ORDER BY priority DESC``: a numerically HIGHER priority
value was claimed first. Every card title / operator convention on this board
uses P0 = most urgent (release blockers, incidents), so a default-priority-0
card was dispatched dead last, and a reclaimed priority-3 card could re-claim
a freed slot ahead of a ready priority-0 card in the very same tick that
reclaimed it for that priority-0 card's sake.

Fixed to ``ORDER BY priority ASC``: lower number claims first.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def isolated_kanban_home(tmp_path, monkeypatch):
    test_home = tmp_path / ".hermes"
    test_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(test_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db
    yield kanban_db, test_home


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


def test_priority_zero_claims_the_single_free_slot_over_priority_three(
    isolated_kanban_home, all_assignees_spawnable,
):
    """Two ready cards, one free slot: the priority-0 (higher urgency) card
    must be the one claimed and spawned, not the priority-3 card, regardless
    of creation order."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        # Priority-3 card created FIRST (as in the live incident: it was
        # reclaimed and became ready before the priority-0 card's turn).
        low_urgency = kb.create_task(conn, title="low urgency", assignee="alice", priority=3)
        high_urgency = kb.create_task(conn, title="release blocker", assignee="alice", priority=0)

    spawns: list = []
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_spawn=1,
        )

    assert spawns == [high_urgency]
    assert [task_id for task_id, *_ in res.spawned] == [high_urgency]
    assert low_urgency not in spawns


def test_lane_rows_orders_priority_ascending(isolated_kanban_home):
    """Direct unit check on the ordering primitive the dispatcher claims from."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        t_lo_urgency = kb.create_task(conn, title="p3", assignee="alice", priority=3)
        t_hi_urgency = kb.create_task(conn, title="p0", assignee="alice", priority=0)
        rows = kbd._lane_rows(conn, "ready")

    assert [r["id"] for r in rows] == [t_hi_urgency, t_lo_urgency]

"""Regression tests for upstream issue #90124.

Two dispatcher allocator defects that starve high-priority work:

1. **Priority ordering inverted.** The ready queue, review queue, and
   default ``list_tasks`` ordering all used ``priority DESC`` so P2
   cards were selected before P1 cards. The documented semantics are
   lower number = higher priority (P0 > P1 > P2 > P3), FIFO within a
   band — i.e. ``priority ASC, created_at ASC``.

2. **Non-spawnable cards consume the demand-floor slot.** A ready card
   whose assignee is a terminal-lane profile (not a real Hermes
   profile) sorts first in the queue and the dispatcher skips it
   without spawning. The budget accounting must not count that skipped
   card against the spawn budget, so a spawnable card further down the
   queue can fill the slot.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures & helpers (mirror the style in test_kanban_host_cap.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


def _set_ready(conn: sqlite3.Connection, task_id: str) -> None:
    """Force a task into the 'ready' state with no claim lock."""
    conn.execute(
        "UPDATE tasks SET status = 'ready', claim_lock = NULL WHERE id = ?",
        (task_id,),
    )


# ---------------------------------------------------------------------------
# 1. Priority ordering: lower number = picked sooner (P0 > P1 > P2)
# ---------------------------------------------------------------------------


def test_ready_queue_picks_lower_priority_first(kanban_home, all_assignees_spawnable):
    """A P1 card must be spawned before a P2 card when both are ready.

    With a spawn budget of 1, only one card spawns. Under the old
    ``priority DESC`` ordering the P2 card won; the fix makes ``priority
    ASC`` so the P1 (lower number) card wins.
    """
    spawns: list = []
    with kb.connect() as conn:
        p2_id = kb.create_task(conn, title="p2-card", assignee="alice", priority=2)
        _set_ready(conn, p2_id)
        # Small sleep so created_at is strictly later for the P1 card —
        # this proves ordering is by priority, not by created_at.
        time.sleep(0.01)
        p1_id = kb.create_task(conn, title="p1-card", assignee="alice", priority=1)
        _set_ready(conn, p1_id)

        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )

    assert len(res.spawned) == 1
    assert res.spawned[0][0] == p1_id, (
        f"expected P1 card {p1_id} to spawn first, got {res.spawned[0][0]}"
    )


def test_ready_queue_fifo_within_same_priority(kanban_home, all_assignees_spawnable):
    """Within the same priority band, earlier created_at wins (FIFO)."""
    spawns: list = []
    with kb.connect() as conn:
        first_id = kb.create_task(conn, title="first", assignee="alice", priority=1)
        _set_ready(conn, first_id)
        time.sleep(0.01)
        second_id = kb.create_task(conn, title="second", assignee="alice", priority=1)
        _set_ready(conn, second_id)

        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )

    assert len(res.spawned) == 1
    assert res.spawned[0][0] == first_id


def test_p0_beats_p1_and_p2(kanban_home, all_assignees_spawnable):
    """P0 (priority 0) is picked before P1 and P2 regardless of created_at."""
    spawns: list = []
    with kb.connect() as conn:
        p2_id = kb.create_task(conn, title="p2", assignee="alice", priority=2)
        _set_ready(conn, p2_id)
        time.sleep(0.01)
        p1_id = kb.create_task(conn, title="p1", assignee="alice", priority=1)
        _set_ready(conn, p1_id)
        time.sleep(0.01)
        p0_id = kb.create_task(conn, title="p0", assignee="alice", priority=0)
        _set_ready(conn, p0_id)

        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )

    assert len(res.spawned) == 1
    assert res.spawned[0][0] == p0_id


def test_list_tasks_default_order_is_priority_asc(kanban_home):
    """The default listing order must show P0 before P1 before P2."""
    with kb.connect() as conn:
        kb.create_task(conn, title="p2", assignee="alice", priority=2)
        kb.create_task(conn, title="p0", assignee="alice", priority=0)
        kb.create_task(conn, title="p1", assignee="alice", priority=1)

        tasks = kb.list_tasks(conn, limit=10)

    priorities = [t.priority for t in tasks]
    assert priorities == sorted(priorities), (
        f"default list order should be priority ASC, got {priorities}"
    )


def test_list_tasks_sort_priority_is_asc(kanban_home):
    """Explicit ``--sort priority`` maps to priority ASC."""
    with kb.connect() as conn:
        kb.create_task(conn, title="p2", assignee="alice", priority=2)
        kb.create_task(conn, title="p0", assignee="alice", priority=0)
        kb.create_task(conn, title="p1", assignee="alice", priority=1)

        tasks = kb.list_tasks(conn, order_by="priority", limit=10)

    priorities = [t.priority for t in tasks]
    assert priorities == [0, 1, 2]


def test_list_tasks_sort_priority_desc_is_desc(kanban_home):
    """Explicit ``--sort priority-desc`` maps to priority DESC."""
    with kb.connect() as conn:
        kb.create_task(conn, title="p0", assignee="alice", priority=0)
        kb.create_task(conn, title="p2", assignee="alice", priority=2)
        kb.create_task(conn, title="p1", assignee="alice", priority=1)

        tasks = kb.list_tasks(conn, order_by="priority-desc", limit=10)

    priorities = [t.priority for t in tasks]
    assert priorities == [2, 1, 0]


# ---------------------------------------------------------------------------
# 2. Non-spawnable cards do not consume the demand-floor slot
# ---------------------------------------------------------------------------


def test_nonspawnable_card_does_not_consume_slot(kanban_home, monkeypatch):
    """A non-spawnable ready card must not consume the spawn budget.

    Setup: a terminal-lane card (assignee not a real profile) sorts
    FIRST in the ready queue (priority 0), followed by a spawnable
    card (priority 1). With a budget of 1, the spawnable card must
    still spawn — the non-spawnable card is skipped without consuming
    the slot.
    """
    import hermes_cli.profiles as profmod

    # Only 'alice' is a real profile; 'code-editor' is a terminal lane.
    monkeypatch.setattr(
        profmod, "profile_exists", lambda name: name == "alice"
    )

    spawns: list = []
    with kb.connect() as conn:
        nonspawn_id = kb.create_task(
            conn, title="terminal-lane", assignee="code-editor", priority=0,
        )
        _set_ready(conn, nonspawn_id)
        time.sleep(0.01)
        spawnable_id = kb.create_task(
            conn, title="spawnable", assignee="alice", priority=1,
        )
        _set_ready(conn, spawnable_id)

        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )

    spawned_ids = [s[0] for s in res.spawned]
    assert spawnable_id in spawned_ids, (
        f"spawnable card {spawnable_id} should have been spawned; "
        f"got {spawned_ids}"
    )
    assert nonspawn_id in res.skipped_nonspawnable, (
        f"non-spawnable card {nonspawn_id} should be in skipped_nonspawnable"
    )
    assert len(spawned_ids) == 1


def test_nonspawnable_cards_starvation_with_full_budget(
    kanban_home, monkeypatch,
):
    """Multiple non-spawnable cards at the front must not block spawnable work.

    With a budget of 2 and three non-spawnable cards at priority 0
    followed by two spawnable cards at priority 1, both spawnable cards
    must spawn. The non-spawnable cards are skipped without consuming
    the budget.
    """
    import hermes_cli.profiles as profmod

    monkeypatch.setattr(
        profmod, "profile_exists", lambda name: name == "alice"
    )

    spawns: list = []
    with kb.connect() as conn:
        for i in range(3):
            tid = kb.create_task(
                conn, title=f"terminal-{i}", assignee="code-editor", priority=0,
            )
            _set_ready(conn, tid)
        time.sleep(0.01)
        spawnable_ids = []
        for i in range(2):
            tid = kb.create_task(
                conn, title=f"spawnable-{i}", assignee="alice", priority=1,
            )
            _set_ready(conn, tid)
            spawnable_ids.append(tid)

        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=2,
        )

    spawned_ids = {s[0] for s in res.spawned}
    assert spawned_ids == set(spawnable_ids), (
        f"expected both spawnable cards {spawnable_ids} to spawn; "
        f"got {spawned_ids}"
    )
    assert len(res.skipped_nonspawnable) == 3


# ---------------------------------------------------------------------------
# 3. Review lane ordering also uses priority ASC
# ---------------------------------------------------------------------------


def test_review_queue_picks_lower_priority_first(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Review-lane dispatch must also order by priority ASC."""
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(
        cfgmod, "load_config",
        lambda *a, **k: {"kanban": {"review_dispatch": True}},
    )

    spawns: list = []
    with kb.connect() as conn:
        # Park two cards in review with different priorities.
        p2_id = kb.create_task(conn, title="review-p2", assignee="reviewer", priority=2)
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (p2_id,))
        time.sleep(0.01)
        p1_id = kb.create_task(conn, title="review-p1", assignee="reviewer", priority=1)
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (p1_id,))

        # Budget 1 so only one review card spawns.
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=1,
        )

    spawned_ids = [s[0] for s in res.spawned]
    assert len(spawned_ids) == 1
    assert spawned_ids[0] == p1_id, (
        f"expected P1 review card {p1_id} to spawn first, got {spawned_ids[0]}"
    )
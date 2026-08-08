"""Regression tests for #21582 — per-profile concurrency cap in dispatcher.

When ``kanban.max_in_progress_per_profile`` is set, no single profile
gets more than N workers running at once even if the global
``max_in_progress`` cap would allow it. Prevents one profile's local
model / API quota / browser pool from being overwhelmed by a fan-out.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home_with_profiles(monkeypatch):
    """Spin up a fresh HERMES_HOME with kanban DB + alpha/beta profiles."""
    test_home = tempfile.mkdtemp(prefix="kanban_per_profile_cap_test_")
    for prof in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345




def test_cap_2_balances_two_profiles(isolated_kanban_home_with_profiles):
    """With cap=2: 2 alpha + 2 beta dispatched; remaining 3 alpha + 1 beta
    deferred to skipped_per_profile_capped."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(5):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
        for i in range(3):
            kb.create_task(conn, title=f"b{i}", assignee="beta")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile=2,
        )
    spawn_assignees = [s[1] for s in res.spawned]
    capped_assignees = [c[1] for c in res.skipped_per_profile_capped]
    assert spawn_assignees.count("alpha") == 2
    assert spawn_assignees.count("beta") == 2
    assert capped_assignees.count("alpha") == 3
    assert capped_assignees.count("beta") == 1




def test_capped_tasks_dispatched_on_subsequent_tick(isolated_kanban_home_with_profiles):
    """A task deferred this tick because its profile was at cap should be
    eligible for dispatch on the next tick (after running tasks complete).
    This verifies the cap is per-tick state, not a permanent block."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        ids = [kb.create_task(conn, title=f"a{i}", assignee="alpha") for i in range(3)]

    # First tick: cap=1, only 1 alpha dispatched
    with kb.connect_closing() as conn:
        res1 = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            max_in_progress_per_profile=1,
        )
    assert len(res1.spawned) == 1
    assert len(res1.skipped_per_profile_capped) == 2

    # Simulate the running task completing — set it back to done so the
    # 'running' count drops
    spawned_id = res1.spawned[0][0]
    with kb.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL WHERE id = ?",
                (spawned_id,),
            )

    # Second tick: 1 more alpha should now dispatch
    with kb.connect_closing() as conn:
        res2 = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            max_in_progress_per_profile=1,
        )
    assert len(res2.spawned) == 1
    assert len(res2.skipped_per_profile_capped) == 1
    assert res2.spawned[0][0] != spawned_id  # different task this time


def test_max_spawn_does_not_count_running_tasks_twice(isolated_kanban_home_with_profiles):
    """Dispatch's max_spawn clamp for total progress should not include
    already-running tasks in the per-tick spawn counter."""
    import time

    kb = isolated_kanban_home_with_profiles

    now = int(time.time())
    with kb.connect_closing() as conn:
        running = [kb.create_task(conn, title=f"running-{i}", assignee="alpha") for i in range(2)]
        for tid in running:
            conn.execute(
                "UPDATE tasks SET status='running', claim_lock='lock', claim_expires=?, worker_pid=? "
                "WHERE id=?",
                (now + 3600, 42, tid),
            )

        for i in range(3):
            kb.create_task(conn, title=f"ready-{i}", assignee="alpha")

    with kb.connect_closing() as conn:
        result = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            max_in_progress=4,
            max_spawn=3,
            dry_run=False,
        )

    # existing behavior clamped max_spawn to remaining (2) but counted both
    # running and spawned in the loop guard, causing 0 spawns. Fixed behavior
    # spawns 2 tasks this tick (to reach max_in_progress=4).
    assert len(result.spawned) == 2


def test_review_dispatch_respects_max_in_progress_with_running_tasks(
    isolated_kanban_home_with_profiles,
):
    """Review-only ticks should also respect max_in_progress via remaining capacity."""
    import time

    kb = isolated_kanban_home_with_profiles
    now = int(time.time())

    with kb.connect_closing() as conn:
        for i in range(2):
            running = kb.create_task(
                conn,
                title=f"running-review-{i}",
                assignee="alpha",
            )
            conn.execute(
                "UPDATE tasks SET status='running', claim_lock='lock', claim_expires=?, worker_pid=? "
                "WHERE id=?",
                (now + 3600, 99, running),
            )

        for i in range(3):
            review_id = kb.create_task(conn, title=f"review-{i}", assignee="alpha")
            conn.execute(
                "UPDATE tasks SET status='review' WHERE id=?",
                (review_id,),
            )

    with kb.connect_closing() as conn:
        result = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            max_in_progress=4,
            max_spawn=3,
            dry_run=False,
        )

    # Remaining capacity is 2, so only two review tasks may be spawned.
    assert len(result.spawned) == 2



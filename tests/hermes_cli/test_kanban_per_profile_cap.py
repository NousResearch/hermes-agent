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
    affected = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith("hermes_cli")
        or name.startswith("hermes_state")
        or name == "hermes_constants"
    }
    for name in affected:
        sys.modules.pop(name, None)
    from hermes_cli import kanban_db
    try:
        yield kanban_db
    finally:
        for name in list(sys.modules):
            if (
                name.startswith("hermes_cli")
                or name.startswith("hermes_state")
                or name == "hermes_constants"
            ):
                sys.modules.pop(name, None)
        sys.modules.update(affected)


def _fake_spawn(*args, **kwargs):
    return 12345


def test_no_cap_all_tasks_dispatched(isolated_kanban_home_with_profiles):
    """Baseline: with no per-profile cap, all ready tasks dispatch."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(5):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
        for i in range(3):
            kb.create_task(conn, title=f"b{i}", assignee="beta")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert len(res.spawned) == 8
    assert not res.skipped_per_profile_capped


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


def test_pre_existing_running_counts_against_cap(isolated_kanban_home_with_profiles):
    """A task already in 'running' status when dispatch_once starts counts
    toward the per-profile cap. With 1 alpha pre-running and cap=1, NO new
    alpha tasks should spawn; beta is independent so 1 beta spawns."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        running_alpha = kb.create_task(conn, title="running alpha", assignee="alpha")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'running', claim_lock = 'test:1' WHERE id = ?",
                (running_alpha,),
            )
        for i in range(2):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
        for i in range(2):
            kb.create_task(conn, title=f"b{i}", assignee="beta")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile=1,
        )
    spawn_assignees = [s[1] for s in res.spawned]
    capped_assignees = [c[1] for c in res.skipped_per_profile_capped]
    assert spawn_assignees.count("alpha") == 0
    assert spawn_assignees.count("beta") == 1
    assert capped_assignees.count("alpha") == 2
    assert capped_assignees.count("beta") == 1


@pytest.mark.parametrize("cap", [0, -1, "abc", None])
def test_invalid_cap_treated_as_no_cap(isolated_kanban_home_with_profiles, cap):
    """Cap values that don't represent a positive int should be treated as
    'no cap' — silently falling through rather than crashing the dispatcher."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(3):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile=cap,
        )
    assert not res.skipped_per_profile_capped
    assert len(res.spawned) == 3


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


def test_dispatch_result_has_skipped_per_profile_capped_field():
    """Schema-level invariant: DispatchResult exposes the
    skipped_per_profile_capped field as a list of
    (task_id, assignee, current_running) tuples."""
    from hermes_cli.kanban_db import DispatchResult
    r = DispatchResult()
    assert hasattr(r, "skipped_per_profile_capped")
    assert r.skipped_per_profile_capped == []

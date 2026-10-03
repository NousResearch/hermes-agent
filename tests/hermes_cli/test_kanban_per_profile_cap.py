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
        with open(os.path.join(test_home, "profiles", prof, "config.yaml"), "w") as fh:
            fh.write("{}\n")  # identity marker: a bare dir is not a profile
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
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(5):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
        for i in range(3):
            kb.create_task(conn, title=f"b{i}", assignee="beta")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
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
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        ids = [kb.create_task(conn, title=f"a{i}", assignee="alpha") for i in range(3)]

    # First tick: cap=1, only 1 alpha dispatched
    with kbc.connect_closing() as conn:
        res1 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            max_in_progress_per_profile=1,
        )
    assert len(res1.spawned) == 1
    assert len(res1.skipped_per_profile_capped) == 2

    # Simulate the running task completing — set it back to done so the
    # 'running' count drops
    spawned_id = res1.spawned[0][0]
    with kbc.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL WHERE id = ?",
                (spawned_id,),
            )

    # Second tick: 1 more alpha should now dispatch
    with kbc.connect_closing() as conn:
        res2 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            max_in_progress_per_profile=1,
        )
    assert len(res2.spawned) == 1
    assert len(res2.skipped_per_profile_capped) == 1
    assert res2.spawned[0][0] != spawned_id  # different task this time


def test_per_profile_cap_counts_other_boards(isolated_kanban_home_with_profiles):
    """A profile at its cap on ANOTHER board blocks a spawn on this board.

    The per-profile cap bounds the profile across the whole host, not per
    board - without the cross-board count, N boards each run up to the cap
    and one profile holds N*cap workers at once.
    """
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    kb.create_board(slug="second")

    # Three 'alpha' workers already running on the second board.
    with kbc.connect(board="second") as conn:
        for i in range(3):
            tid = kb.create_task(conn, title=f"busy-{i}", assignee="alpha")
            assert kb.claim_task(conn, tid) is not None

    # One ready 'alpha' here: already at the cap (3) elsewhere → deferred.
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wants-to-run", assignee="alpha")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile=3,
        )

    assert [s[0] for s in res.spawned] == []
    assert len(res.skipped_per_profile_capped) == 1
    assert res.skipped_per_profile_capped[0][0] == tid
    assert res.skipped_per_profile_capped[0][1] == "alpha"
    assert res.skipped_per_profile_capped[0][2] == 3


def test_per_profile_cap_partial_budget_across_boards(
    isolated_kanban_home_with_profiles,
):
    """A profile under its cap across boards still spawns - the cross-board
    count adds, it does not over-block other profiles under the global cap."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    kb.create_board(slug="second")

    # Two 'alpha' workers running on the second board.
    with kbc.connect(board="second") as conn:
        for i in range(2):
            tid = kb.create_task(conn, title=f"busy-{i}", assignee="alpha")
            assert kb.claim_task(conn, tid) is not None

    # One ready 'alpha' here: 2 running + 1 spawn = 3, exactly at cap → allowed.
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wants-to-run", assignee="alpha")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile=3,
        )

    assert [s[0] for s in res.spawned] == [tid]
    assert res.skipped_per_profile_capped == []



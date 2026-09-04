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


@pytest.mark.parametrize("dry_run", [True, False])
def test_profile_overrides_share_capacity_across_ready_and_review(
    isolated_kanban_home_with_profiles,
    dry_run,
):
    """An assignee override is one ceiling shared by both dispatch lanes."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        running_alpha = kb.create_task(conn, title="running", assignee="alpha")
        ready_alpha = kb.create_task(conn, title="ready", assignee="alpha")
        review_alpha = kb.create_task(conn, title="review", assignee="alpha")
        beta_ids = [
            kb.create_task(conn, title=f"beta-{i}", assignee="beta")
            for i in range(4)
        ]
        assert kb.claim_task(conn, running_alpha) is not None
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'review' WHERE id = ?",
                (review_alpha,),
            )

    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            dry_run=dry_run,
            max_in_progress_per_profile=3,
            max_in_progress_per_profile_overrides={"alpha": 1},
        )

    spawned_ids = {task_id for task_id, _, _ in res.spawned}
    capped_ids = {task_id for task_id, _, _ in res.skipped_per_profile_capped}
    assert ready_alpha in capped_ids
    assert review_alpha in capped_ids
    assert not {ready_alpha, review_alpha}.intersection(spawned_ids)
    assert not [task_id for task_id, assignee, _ in res.spawned if assignee == "alpha"]
    assert len(spawned_ids.intersection(beta_ids)) == 3


def test_normalize_profile_cap_overrides_warns_and_keeps_valid_entries(caplog):
    """Gateway and CLI share one parser with actionable invalid-entry warnings."""
    import logging

    from hermes_cli import kanban_db as kb

    with caplog.at_level(logging.WARNING, logger="hermes_cli.kanban_db"):
        parsed = kb.normalize_profile_cap_overrides(
            {
                "supervisor": 1,
                "implementer": "3",
                "zero": 0,
                "bad": "many",
                "": 2,
            }
        )

    assert parsed == {"supervisor": 1, "implementer": 3}
    messages = [record.getMessage() for record in caplog.records]
    assert any("zero" in message and "below 1" in message for message in messages)
    assert any("bad" in message and "invalid" in message for message in messages)
    assert any("profile name" in message for message in messages)



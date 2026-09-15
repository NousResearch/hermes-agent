"""Named kanban.capacity_pools: several profiles share one host-wide context.

Implements the admission shape requested in #96299 (shared named pools) without
a separate lease ledger. A pool name is an operator-chosen context (vendor,
local role, person). Running members are counted across every board; extras
stay Ready as skipped_pool_capped. Tests use ``spark`` as one such name.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home_with_profiles(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_capacity_pools_")
    for prof in ("coder", "paul-coder", "reviewer", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345


def _spark_pool():
    from hermes_cli.kanban_db_dispatch import CapacityPool
    return {
        "spark": CapacityPool(
            name="spark", max_in_progress=1, members=("coder", "paul-coder"),
        )
    }


def test_normalize_capacity_pools_ignores_invalid_entries():
    from hermes_cli.kanban_db_dispatch import normalize_capacity_pools
    pools = normalize_capacity_pools({
        "spark": {"max_in_progress": 1, "members": ["coder", "paul-coder"]},
        "": {"max_in_progress": 1, "members": ["x"]},
        "bad-cap": {"max_in_progress": 0, "members": ["y"]},
        "no-members": {"max_in_progress": 2, "members": []},
        "not-a-map": "nope",
    })
    assert set(pools) == {"spark"}
    assert pools["spark"].members == ("coder", "paul-coder")


def test_normalize_capacity_pools_first_pool_keeps_shared_member():
    from hermes_cli.kanban_db_dispatch import normalize_capacity_pools
    pools = normalize_capacity_pools({
        "spark": {"max_in_progress": 1, "members": ["coder"]},
        "other": {"max_in_progress": 2, "members": ["coder", "reviewer"]},
    })
    assert pools["spark"].members == ("coder",)
    assert pools["other"].members == ("reviewer",)


def test_unset_pools_leave_dispatch_uncapped(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(3):
            kb.create_task(conn, title=f"c{i}", assignee="coder")
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert len(res.spawned) == 3
    assert res.skipped_pool_capped == []


def test_pool_caps_two_profiles_on_one_board(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="c0", assignee="coder")
        kb.create_task(conn, title="p0", assignee="paul-coder")
        kb.create_task(conn, title="r0", assignee="reviewer")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            capacity_pools=_spark_pool(),
        )
    spawn_assignees = [s[1] for s in res.spawned]
    assert spawn_assignees.count("coder") + spawn_assignees.count("paul-coder") == 1
    assert spawn_assignees.count("reviewer") == 1
    assert len(res.skipped_pool_capped) == 1
    assert res.skipped_pool_capped[0][1] == "spark"
    assert res.skipped_pool_capped[0][3] == 1


def test_pool_counts_running_on_other_boards(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    kb.create_board("second")
    with kbc.connect(board="second") as conn:
        tid = kb.create_task(conn, title="busy-spark", assignee="coder")
        assert kb.claim_task(conn, tid) is not None
    with kbc.connect() as conn:
        kb.create_task(conn, title="wants-spark", assignee="paul-coder")
        kb.create_task(conn, title="cloud", assignee="reviewer")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            capacity_pools=_spark_pool(),
        )
    spawn_assignees = [s[1] for s in res.spawned]
    assert "paul-coder" not in spawn_assignees
    assert spawn_assignees == ["reviewer"]
    assert res.skipped_pool_capped[0][1] == "spark"
    assert res.skipped_pool_capped[0][2] == 1


def test_pool_deferred_card_stays_ready(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        first = kb.create_task(conn, title="first", assignee="coder")
        second = kb.create_task(conn, title="second", assignee="paul-coder")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            capacity_pools=_spark_pool(),
        )
    assert len(res.spawned) == 1
    spawned_id = res.spawned[0][0]
    deferred_id = second if spawned_id == first else first
    assert res.skipped_pool_capped[0][0] == deferred_id
    with kbc.connect_closing() as conn:
        deferred = kb.get_task(conn, deferred_id)
        assert deferred.status == "ready"
        assert deferred.claim_lock is None


def test_pool_slot_frees_on_complete(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="first", assignee="coder")
        kb.create_task(conn, title="second", assignee="paul-coder")
        res1 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            capacity_pools=_spark_pool(),
        )
    assert len(res1.spawned) == 1
    spawned_id = res1.spawned[0][0]
    with kbc.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL WHERE id = ?",
                (spawned_id,),
            )
        res2 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            capacity_pools=_spark_pool(),
        )
    assert len(res2.spawned) == 1
    assert res2.spawned[0][0] != spawned_id


def test_cli_dispatch_passes_capacity_pools(isolated_kanban_home_with_profiles, monkeypatch):
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_dispatch as kbd
    import argparse

    fake_config = {
        "kanban": {
            "capacity_pools": {
                "spark": {"max_in_progress": 1, "members": ["coder"]},
            }
        }
    }
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)
    captured = {}

    def fake_dispatch_once(conn, **kwargs):
        captured.update(kwargs)
        return kanban_db.DispatchResult()

    monkeypatch.setattr(kbd, "dispatch_once", fake_dispatch_once)
    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)
    pools = captured.get("capacity_pools") or {}
    assert "spark" in pools
    assert pools["spark"].max_in_progress == 1
    assert pools["spark"].members == ("coder",)


def test_two_named_contexts_are_independent(isolated_kanban_home_with_profiles):
    """A full spark pool must not freeze an unrelated grok pool."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli.kanban_db_dispatch import CapacityPool
    pools = {
        "spark": CapacityPool(
            name="spark", max_in_progress=1, members=("coder",),
        ),
        "grok": CapacityPool(
            name="grok", max_in_progress=2, members=("reviewer",),
        ),
    }
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(3):
            kb.create_task(conn, title=f"s{i}", assignee="coder")
        for i in range(3):
            kb.create_task(conn, title=f"g{i}", assignee="reviewer")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            capacity_pools=pools,
        )
    spawn_assignees = [s[1] for s in res.spawned]
    assert spawn_assignees.count("coder") == 1
    assert spawn_assignees.count("reviewer") == 2
    spark_deferred = [c for c in res.skipped_pool_capped if c[1] == "spark"]
    grok_deferred = [c for c in res.skipped_pool_capped if c[1] == "grok"]
    assert len(spark_deferred) == 2
    assert len(grok_deferred) == 1


def test_review_lane_shares_the_same_pool(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        ready_id = kb.create_task(conn, title="ready-spark", assignee="coder")
        review_id = kb.create_task(conn, title="review-spark", assignee="coder")
        claimed = kb.claim_task(conn, review_id)
        assert claimed is not None
        assert kb.request_review(
            conn, review_id, summary="ready",
            expected_run_id=claimed.current_run_id,
        )
        assert kb.get_task(conn, review_id).status == "review"
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            capacity_pools=_spark_pool(),
        )
    spawned_ids = [s[0] for s in res.spawned]
    spark_spawned = [s for s in res.spawned if s[1] == "coder"]
    assert len(spark_spawned) == 1
    assert (ready_id in spawned_ids) ^ (review_id in spawned_ids)
    assert any(c[1] == "spark" for c in res.skipped_pool_capped)


def test_host_cap_still_wins_over_a_roomy_pool(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli.kanban_db_dispatch import CapacityPool
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(4):
            kb.create_task(conn, title=f"r{i}", assignee="reviewer")
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress=2,
            capacity_pools={
                "grok": CapacityPool(
                    name="grok", max_in_progress=8, members=("reviewer",),
                )
            },
        )
    assert len(res.spawned) == 2

"""Regression tests for #111188 — rules-aware parallel worker assignment.

``kanban.parallel_exclusion_groups`` lists profile-name sets that must never
run at once (e.g. two profiles whose models share one GPU). The dispatcher
defers the loser to ``skipped_excluded`` instead of spawning into resource
contention; the task is picked up on a later tick once the sibling finishes.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home_with_profiles(monkeypatch):
    """Fresh HERMES_HOME with live gpu0fast/gpu0dense/gpu1solo profiles."""
    test_home = tempfile.mkdtemp(prefix="kanban_parallel_exclusion_test_")
    for prof in ("gpu0fast", "gpu0dense", "gpu1solo", "default"):
        prof_dir = os.path.join(test_home, "profiles", prof)
        os.makedirs(prof_dir, exist_ok=True)
        with open(os.path.join(prof_dir, "SOUL.md"), "w", encoding="utf-8") as fh:
            fh.write("# test profile\n")
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345


GROUPS = [["gpu0fast", "gpu0dense"]]


def test_same_group_second_task_deferred_same_tick(isolated_kanban_home_with_profiles):
    """One tick, two ready tasks on one GPU: first spawns, second defers."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="research", assignee="gpu0fast")
        kb.create_task(conn, title="coding", assignee="gpu0dense")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            parallel_exclusion_groups=GROUPS,
        )
    assert len(res.spawned) == 1
    assert len(res.skipped_excluded) == 1
    task_id, who, conflict = res.skipped_excluded[0]
    assert who != res.spawned[0][1]
    assert conflict == res.spawned[0][1]


def test_running_sibling_blocks_next_tick_then_releases(isolated_kanban_home_with_profiles):
    """A DB-running sibling blocks the next tick; completion unblocks."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="research", assignee="gpu0fast")
        kb.create_task(conn, title="coding", assignee="gpu0dense")
    with kbc.connect_closing() as conn:
        res1 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            parallel_exclusion_groups=GROUPS,
        )
    assert len(res1.spawned) == 1
    assert res1.spawned[0][1] == "gpu0fast"
    with kbc.connect_closing() as conn:
        res2 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            parallel_exclusion_groups=GROUPS,
        )
    assert len(res2.spawned) == 0
    assert res2.skipped_excluded == [(res2.skipped_excluded[0][0], "gpu0dense", "gpu0fast")]
    spawned_id = res1.spawned[0][0]
    with kbc.connect_closing() as conn:
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'done', claim_lock = NULL WHERE id = ?",
                (spawned_id,),
            )
    with kbc.connect_closing() as conn:
        res3 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            parallel_exclusion_groups=GROUPS,
        )
    assert [s[1] for s in res3.spawned] == ["gpu0dense"]
    assert res3.skipped_excluded == []


def test_unrelated_profiles_still_parallel(isolated_kanban_home_with_profiles):
    """Profiles in no shared group spawn together; empty config is a no-op."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        kb.create_task(conn, title="research", assignee="gpu0fast")
        kb.create_task(conn, title="other", assignee="gpu1solo")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            parallel_exclusion_groups=GROUPS,
        )
    assert sorted(s[1] for s in res.spawned) == ["gpu0fast", "gpu1solo"]
    assert res.skipped_excluded == []
    with kbc.connect_closing() as conn:
        res2 = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=True)
    assert len(res2.spawned) == 2


def test_normalize_exclusion_groups():
    from hermes_cli import kanban_db_dispatch as kbd
    assert kbd.normalize_exclusion_groups(None) == []
    assert kbd.normalize_exclusion_groups([]) == []
    assert kbd.normalize_exclusion_groups(42) == []
    assert kbd.normalize_exclusion_groups([["solo"]]) == []
    assert kbd.normalize_exclusion_groups([["a", "b"], ["b", "a"]]) == [frozenset({"a", "b"})]
    assert kbd.normalize_exclusion_groups("a, b") == [frozenset({"a", "b"})]
    assert kbd.normalize_exclusion_groups([["GPU0Fast", "gpu0dense"]]) == [
        frozenset({"gpu0fast", "gpu0dense"})]

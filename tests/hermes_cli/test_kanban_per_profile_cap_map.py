"""Regression tests for #106784 — map-form per-profile caps + auto_assign.

``kanban.max_in_progress_per_profile`` may be a mapping so local and cloud
profiles can have asymmetric concurrency. ``kanban.auto_assign`` with
``strategy: local-first-overflow`` routes unassigned ready tasks to the first
unsaturated local-pool profile, overflowing to cloud_pool.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home_with_profiles(monkeypatch):
    """Fresh HERMES_HOME with kanban DB + local/cloud/default profiles."""
    test_home = tempfile.mkdtemp(prefix="kanban_per_profile_cap_map_test_")
    for prof in ("local", "cloud", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db


def _fake_spawn(*args, **kwargs):
    return 12345


def _mark_running(kb, conn, task_id: str) -> None:
    """Park a row in ``running`` with claim bookkeeping so orphan reconcile
    does not bounce it back to ``ready`` mid-tick."""
    import time
    now = int(time.time())
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'running', claim_lock = ?, "
            "claim_expires = ? WHERE id = ?",
            (f"test-lock-{task_id}", now + 3600, task_id),
        )


CAP_MAP = {"local": 1, "cloud": 2, "default": 1}

AUTO_ASSIGN = {
    "enabled": True,
    "strategy": "local-first-overflow",
    "local_pool": ["local"],
    "cloud_pool": ["cloud"],
}


def test_cap_map_asymmetric_local_and_cloud(isolated_kanban_home_with_profiles):
    """Map {local:1, cloud:2, default:1}: 2 local + 3 cloud ready, dry_run.

    EXPECT: spawned local==1, cloud==2; extras in skipped_per_profile_capped.
    Pre-fix main ignores/coerces the map, so caps cannot be asymmetric.
    """
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(2):
            kb.create_task(conn, title=f"local-{i}", assignee="local")
        for i in range(3):
            kb.create_task(conn, title=f"cloud-{i}", assignee="cloud")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile=CAP_MAP,
        )
    spawn_assignees = [s[1] for s in res.spawned]
    capped_assignees = [c[1] for c in res.skipped_per_profile_capped]
    assert spawn_assignees.count("local") == 1
    assert spawn_assignees.count("cloud") == 2
    assert capped_assignees.count("local") == 1
    assert capped_assignees.count("cloud") == 1


def test_cap_map_invalid_key_falls_through_to_default(isolated_kanban_home_with_profiles):
    """Non-int map entry is ignored; usable ``default`` still applies."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(2):
            kb.create_task(conn, title=f"local-{i}", assignee="local")
        for i in range(3):
            kb.create_task(conn, title=f"cloud-{i}", assignee="cloud")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile={"local": "nope", "cloud": 2, "default": 1},
        )
    spawn_assignees = [s[1] for s in res.spawned]
    assert spawn_assignees.count("local") == 1
    assert spawn_assignees.count("cloud") == 2


def test_auto_assign_overflows_to_cloud_when_local_at_cap(isolated_kanban_home_with_profiles):
    """local cap=1 with 1 already running: both unassigned ready tasks go cloud."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        running_id = kb.create_task(conn, title="running-local", assignee="local")
        _mark_running(kb, conn, running_id)
        t1 = kb.create_task(conn, title="u1", assignee=None)
        t2 = kb.create_task(conn, title="u2", assignee=None)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile={"local": 1, "cloud": 2},
            auto_assign=AUTO_ASSIGN,
            reconcile_orphans=False,
        )
    assert not res.skipped_unassigned
    assert set(res.auto_assigned_default) == {t1, t2}
    spawn_assignees = [s[1] for s in res.spawned]
    assert spawn_assignees.count("local") == 0
    assert spawn_assignees.count("cloud") == 2


def test_auto_assign_prefers_local_then_overflows(isolated_kanban_home_with_profiles):
    """No running local, cap=1: first unassigned → local, second → cloud."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        t1 = kb.create_task(conn, title="u1", assignee=None)
        t2 = kb.create_task(conn, title="u2", assignee=None)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            max_in_progress_per_profile={"local": 1, "cloud": 2},
            auto_assign=AUTO_ASSIGN,
        )
    assert not res.skipped_unassigned
    assert set(res.auto_assigned_default) == {t1, t2}
    by_id = {s[0]: s[1] for s in res.spawned}
    assert by_id[t1] == "local"
    assert by_id[t2] == "cloud"

    with kbc.connect_closing() as conn:
        rows = {
            row["id"]: row["assignee"]
            for row in conn.execute(
                "SELECT id, assignee FROM tasks WHERE id IN (?, ?)", (t1, t2)
            )
        }
    assert rows[t1] == "local"
    assert rows[t2] == "cloud"

    with kbc.connect_closing() as conn:
        evs = list(conn.execute(
            "SELECT task_id, payload FROM task_events "
            "WHERE kind = 'assigned' AND task_id IN (?, ?)",
            (t1, t2),
        ))
    sources = {row[0]: json.loads(row[1])["source"] for row in evs}
    assert sources[t1] == "kanban.auto_assign"
    assert sources[t2] == "kanban.auto_assign"


def test_auto_assign_honors_cloud_cap_after_overflow(isolated_kanban_home_with_profiles):
    """After local is full, overflow still honors the cloud profile cap."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        running_id = kb.create_task(conn, title="running-local", assignee="local")
        _mark_running(kb, conn, running_id)
        ids = [kb.create_task(conn, title=f"u{i}", assignee=None) for i in range(3)]
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            max_in_progress_per_profile={"local": 1, "cloud": 2},
            auto_assign=AUTO_ASSIGN,
            reconcile_orphans=False,
        )
    spawn_assignees = [s[1] for s in res.spawned]
    assert spawn_assignees.count("local") == 0
    assert spawn_assignees.count("cloud") == 2
    leftover = [tid for tid in ids if tid not in {s[0] for s in res.spawned}]
    assert len(leftover) == 1
    # No unsaturated pool member remains — fall through, do not pin to a
    # saturated cloud profile.
    assert leftover[0] in res.skipped_unassigned or leftover[0] in {
        c[0] for c in res.skipped_per_profile_capped
    }


def test_auto_assign_does_not_reassign_explicit_assignee(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="mine", assignee="cloud")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            auto_assign=AUTO_ASSIGN,
        )
    assert task_id not in res.auto_assigned_default
    assert any(s[0] == task_id and s[1] == "cloud" for s in res.spawned)


def test_auto_assign_disabled_falls_through_to_default_assignee(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="u1", assignee=None)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            default_assignee="default",
            auto_assign={"enabled": False, "strategy": "local-first-overflow",
                         "local_pool": ["local"], "cloud_pool": ["cloud"]},
        )
    assert res.auto_assigned_default == [task_id]
    assert res.spawned[0][1] == "default"
    with kbc.connect_closing() as conn:
        ev = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'assigned'",
            (task_id,),
        ).fetchone()
    assert json.loads(ev[0])["source"] == "kanban.default_assignee"


def test_auto_assign_unknown_strategy_is_ignored(isolated_kanban_home_with_profiles):
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="u1", assignee=None)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            auto_assign={"enabled": True, "strategy": "label-routing",
                         "local_pool": ["local"], "cloud_pool": ["cloud"]},
        )
    assert res.skipped_unassigned == [task_id]
    assert not res.auto_assigned_default


def test_review_unassigned_not_auto_assigned(isolated_kanban_home_with_profiles):
    """Review-lane unassigned rows stay skipped (today's behavior)."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="needs-review", assignee=None)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            auto_assign=AUTO_ASSIGN,
        )
    assert task_id in res.skipped_unassigned
    assert task_id not in res.auto_assigned_default
    assert not any(s[0] == task_id for s in res.spawned)


def test_pick_auto_assign_profile_unit():
    from hermes_cli.kanban_db_dispatch import pick_auto_assign_profile, resolve_per_profile_cap

    caps = {"local": 1, "cloud": 2}

    def cap_for(name: str):
        return resolve_per_profile_cap(caps, name)

    assert pick_auto_assign_profile(
        ["local"], ["cloud"], running={"local": 0}, cap_for=cap_for,
        profile_usable=lambda _n: True,
    ) == "local"
    assert pick_auto_assign_profile(
        ["local"], ["cloud"], running={"local": 1}, cap_for=cap_for,
        profile_usable=lambda _n: True,
    ) == "cloud"
    assert pick_auto_assign_profile(
        ["local"], ["cloud"], running={"local": 1, "cloud": 2}, cap_for=cap_for,
        profile_usable=lambda _n: True,
    ) is None
    assert pick_auto_assign_profile(
        ["missing"], ["cloud"], running={}, cap_for=lambda _n: None,
        profile_usable=lambda n: n == "cloud",
    ) == "cloud"
    # Uncapped pool member stays eligible even when its running count is high.
    assert pick_auto_assign_profile(
        ["uncapped"], ["cloud"], running={"uncapped": 99, "cloud": 0},
        cap_for=lambda n: None if n == "uncapped" else 2,
        profile_usable=lambda _n: True,
    ) == "uncapped"


def test_resolve_per_profile_cap_unit():
    from hermes_cli.kanban_db_dispatch import (
        parse_max_in_progress_per_profile,
        resolve_per_profile_cap,
    )

    assert parse_max_in_progress_per_profile(2) == 2
    assert parse_max_in_progress_per_profile(0) is None
    assert parse_max_in_progress_per_profile(None) is None
    assert parse_max_in_progress_per_profile("nope") is None
    parsed = parse_max_in_progress_per_profile({"local": 1, "cloud": 2, "default": 1})
    assert resolve_per_profile_cap(parsed, "local") == 1
    assert resolve_per_profile_cap(parsed, "cloud") == 2
    assert resolve_per_profile_cap(parsed, "other") == 1
    parsed_no_default = parse_max_in_progress_per_profile({"local": 1})
    assert resolve_per_profile_cap(parsed_no_default, "cloud") is None
    assert resolve_per_profile_cap(2, "anyone") == 2

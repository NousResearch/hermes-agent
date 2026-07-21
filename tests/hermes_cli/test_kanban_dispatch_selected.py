from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home_with_profiles(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_selected_dispatch_test_")
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


def test_dispatch_once_selected_task_ids_only_spawns_requested_ready_tasks(isolated_kanban_home_with_profiles):
    """Manual dashboard nudge with selected rows is closed-world.

    Even with extra ready tasks and a larger max_spawn cap, dispatch_once must
    only consider the operator-selected task IDs.
    """
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        unselected_early = kb.create_task(conn, title="unselected early", assignee="alpha", priority=10)
        selected_one = kb.create_task(conn, title="selected one", assignee="alpha", priority=1)
        selected_two = kb.create_task(conn, title="selected two", assignee="beta", priority=1)
        unselected_late = kb.create_task(conn, title="unselected late", assignee="beta", priority=9)

    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            dry_run=True,
            max_spawn=8,
            selected_task_ids=[selected_one, selected_two],
        )

    assert [tid for tid, _assignee, _workspace in res.spawned] == [selected_one, selected_two]
    assert unselected_early not in [tid for tid, _assignee, _workspace in res.spawned]
    assert unselected_late not in [tid for tid, _assignee, _workspace in res.spawned]


def test_dispatch_once_selected_task_ids_do_not_fallback_when_selected_is_unspawnable(isolated_kanban_home_with_profiles):
    """If a selected task cannot spawn, unselected ready tasks are not substitutes."""
    kb = isolated_kanban_home_with_profiles
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        selected_unassigned = kb.create_task(conn, title="selected unassigned", assignee=None, priority=10)
        unselected_spawnable = kb.create_task(conn, title="unselected spawnable", assignee="alpha", priority=1)

    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            dry_run=True,
            max_spawn=8,
            selected_task_ids=[selected_unassigned],
        )

    assert res.spawned == []
    assert selected_unassigned in res.skipped_unassigned
    assert unselected_spawnable not in res.skipped_unassigned

"""Regression coverage for #106784's asymmetric dispatcher routing."""
from __future__ import annotations

import os
import sys
import tempfile

import pytest


@pytest.fixture()
def kanban_home(monkeypatch):
    home = tempfile.mkdtemp(prefix="kanban_cap_map_")
    for profile in ("local", "cloud"):
        path = os.path.join(home, "profiles", profile)
        os.makedirs(path)
        with open(os.path.join(path, "config.yaml"), "w") as file:
            file.write("{}\n")
    monkeypatch.setenv("HERMES_HOME", home)
    for name in list(sys.modules):
        if name.startswith("hermes_cli") or name in {"hermes_constants", "hermes_state"}:
            del sys.modules[name]
    from hermes_cli import kanban_db
    return kanban_db


def _spawn(*_args, **_kwargs):
    return 12345


def test_map_cap_is_resolved_per_assignee(kanban_home):
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch

    with kbc.connect_closing() as conn:
        kanban_home.create_board(slug="default", name="Test")
        for index in range(2):
            kanban_home.create_task(conn, title=f"local-{index}", assignee="local")
        for index in range(3):
            kanban_home.create_task(conn, title=f"cloud-{index}", assignee="cloud")
    with kbc.connect_closing() as conn:
        result = dispatch.dispatch_once(
            conn, spawn_fn=_spawn, dry_run=True,
            max_in_progress_per_profile={"local": 1, "cloud": 2},
        )
    assert [item[1] for item in result.spawned].count("local") == 1
    assert [item[1] for item in result.spawned].count("cloud") == 2


def test_auto_assignment_uses_local_then_cloud_after_local_cap(kanban_home):
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as dispatch

    with kbc.connect_closing() as conn:
        kanban_home.create_board(slug="default", name="Test")
        first = kanban_home.create_task(conn, title="first", assignee=None)
        second = kanban_home.create_task(conn, title="second", assignee=None)
    with kbc.connect_closing() as conn:
        result = dispatch.dispatch_once(
            conn, spawn_fn=_spawn, dry_run=False,
            max_in_progress_per_profile={"local": 1, "cloud": 2},
            auto_assign={"enabled": True, "strategy": "local-first-overflow",
                         "local_pool": ["local"], "cloud_pool": ["cloud"]},
        )
    assigned = {task_id: assignee for task_id, assignee, _path in result.spawned}
    assert assigned[first] == "local"
    assert assigned[second] == "cloud"


def test_auto_assignment_uses_default_cap_and_skips_unknown_profile(monkeypatch):
    from hermes_cli import kanban_db_dispatch as dispatch

    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda name: name != "gone")
    config = dispatch.parse_auto_assign({
        "enabled": True, "strategy": "local-first-overflow",
        "local_pool": ["gone", "local"], "cloud_pool": ["cloud"],
    })
    assert config is not None
    assert dispatch._choose_unassigned_assignee(
        config, "fallback", running={"local": 1},
        cap_for=lambda name: 1 if name in {"local", "cloud"} else None,
    ) == ("cloud", "kanban.auto_assign")
    assert dispatch.resolve_per_profile_cap({"default": 2}, "other") == 2
    assert dispatch._choose_unassigned_assignee(
        {"local_pool": [], "cloud_pool": []}, "fallback", running={}, cap_for=lambda _name: None,
    ) == ("fallback", "kanban.default_assignee")

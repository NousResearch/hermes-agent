"""Regression tests for #27145 — kanban.default_assignee for unassigned ready tasks.

When the dispatcher hits an unassigned ready task and ``kanban.default_assignee``
is set, the dispatcher applies the assignment and spawns. Without the config,
the task is skipped (existing behavior preserved).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Spin up a fresh HERMES_HOME with a clean kanban DB."""
    test_home = tempfile.mkdtemp(prefix="kanban_default_assignee_test_")
    monkeypatch.setenv("HERMES_HOME", test_home)
    # Force-reimport so the fresh HERMES_HOME is picked up, then restore the
    # collected suite's module identities. Leaving the reimported modules in
    # sys.modules splits later tests between stale and current singletons.
    def is_hermes_module(name):
        return (
            name.startswith("hermes_cli")
            or name.startswith("hermes_state")
            or name == "hermes_constants"
        )

    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if is_hermes_module(name)
    }
    for name in saved_modules:
        del sys.modules[name]
    try:
        from hermes_cli import kanban_db

        yield kanban_db, test_home
    finally:
        for name in list(sys.modules):
            if is_hermes_module(name):
                del sys.modules[name]
        sys.modules.update(saved_modules)


def _fake_spawn(*args, **kwargs):
    """Stand-in for the real worker spawn — returns a fake PID."""
    return 12345




def test_unassigned_task_auto_assigned_with_default_assignee(isolated_kanban_home):
    """Core #27145 contract: with default_assignee set, an unassigned ready
    task gets the assignment applied and dispatched on the same tick. The
    DB row is mutated (assignee column + an 'assigned' event)."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            default_assignee="default",
        )
    assert res.auto_assigned_default == [task_id]
    assert not res.skipped_unassigned
    assert len(res.spawned) == 1
    assert res.spawned[0][0] == task_id
    assert res.spawned[0][1] == "default"

    with kbc.connect_closing() as conn:
        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row["assignee"] == "default"

    # 'assigned' event emitted for the audit trail
    with kbc.connect_closing() as conn:
        evs = list(conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind = 'assigned'",
            (task_id,),
        ))
    assert len(evs) == 1
    payload = json.loads(evs[0][1])
    assert payload["assignee"] == "default"
    assert payload["source"] == "kanban.default_assignee"






def test_explicitly_assigned_task_untouched_by_default_assignee(isolated_kanban_home):
    """A task with an explicit assignee must NOT be touched by the
    default_assignee logic — that fallback only applies to genuinely
    unassigned rows."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee="default")
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            default_assignee="someother",
        )
    assert task_id not in res.auto_assigned_default
    assert any(s[0] == task_id and s[1] == "default" for s in res.spawned)


def test_generated_task_with_removed_profile_uses_default_fallback(
    isolated_kanban_home, monkeypatch,
):
    """Generated cards must not strand when their specialist profile is removed.

    Explicit human/control-plane assignees remain non-spawnable; only known
    automated producers receive the configured fallback.
    """
    kb, _home = isolated_kanban_home
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "default")
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(
            conn,
            title="generated child",
            assignee="removed-specialist",
            created_by="auto-decomposer",
        )

    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn,
            spawn_fn=_fake_spawn,
            dry_run=False,
            default_assignee="default",
        )
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'assigned' "
            "ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()

    assert res.auto_reassigned_invalid == [task_id]
    assert res.skipped_nonspawnable == []
    assert res.spawned[0][0] == task_id
    assert row["assignee"] == "default"
    payload = json.loads(event["payload"])
    assert payload["source"] == "kanban.invalid_assignee_fallback"
    assert payload["previous_assignee"] == "removed-specialist"

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
    # Force-reimport so the fresh HERMES_HOME is picked up.
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    yield kanban_db, test_home
    # Cleanup is best-effort; tempfile dir survives but pytest isolation
    # gives each test its own monkeypatched HERMES_HOME so no cross-test
    # contamination.


def _fake_spawn(*args, **kwargs):
    """Stand-in for the real worker spawn — returns a fake PID."""
    return 12345




@pytest.mark.parametrize("default_assignee", [None, "", "   "])
def test_unassigned_ready_skip_is_durable(isolated_kanban_home, default_assignee):
    """Regression for #100956: unattended ready skips survive reconnection."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="needs routing", assignee=None)
        result = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, default_assignee=default_assignee,
        )
    with kbc.connect_closing() as conn:
        events = conn.execute(
            "SELECT payload FROM task_events "
            "WHERE task_id = ? AND kind = 'skipped_unassigned'",
            (task_id,),
        ).fetchall()
        row = conn.execute("SELECT status, assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()

    assert result.skipped_unassigned == [task_id]
    assert result.spawned == []
    assert tuple(row) == ("ready", None)
    assert [json.loads(event["payload"]) for event in events] == [{"reason": "no_assignee"}]


@pytest.mark.parametrize(
    ("dry_run", "default_assignee"),
    [(True, None), (True, ""), (True, "   "), (True, "default"),
     (True, "missing-profile"), (False, "missing-profile")],
)
def test_unassigned_skip_boundaries_do_not_write(
    isolated_kanban_home, dry_run, default_assignee,
):
    """Dry runs and invalid defaults must not write misleading skip events."""
    kb, _home = isolated_kanban_home
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="needs routing", assignee=None)
        before = list(conn.iterdump())
        result = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=dry_run,
            default_assignee=default_assignee,
        )
    with kbc.connect_closing() as conn:
        assert list(conn.iterdump()) == before
    if default_assignee == "default":
        assert result.auto_assigned_default == [task_id]
    else:
        assert result.skipped_unassigned == [task_id]
        assert result.spawned == []


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



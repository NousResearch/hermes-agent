"""Regression tests for #27145 — kanban.default_assignee for unassigned ready tasks.

When the dispatcher hits an unassigned ready task and ``kanban.default_assignee``
is set, the dispatcher applies the assignment and spawns. Without the config,
the task is skipped (existing behavior preserved).
"""
from __future__ import annotations

import json

import pytest

from hermes_cli import kanban_db


@pytest.fixture()
def isolated_kanban_home(monkeypatch, tmp_path):
    """Use a fresh HERMES_HOME without replacing canonical modules."""
    test_home = tmp_path / ".hermes"
    test_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(test_home))
    yield kanban_db, str(test_home)


def _fake_spawn(*args, **kwargs):
    """Stand-in for the real worker spawn — returns a fake PID."""
    return 12345


def test_unassigned_task_skipped_without_default_assignee(isolated_kanban_home):
    """Baseline: with no default_assignee, an unassigned ready task is
    skipped via the existing `skipped_unassigned` bucket and the DB row
    is untouched."""
    kb, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(conn, spawn_fn=_fake_spawn, dry_run=False)
    assert res.skipped_unassigned == [task_id]
    assert not res.auto_assigned_default
    assert not res.spawned
    with kb.connect_closing() as conn:
        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row["assignee"] is None


def test_unassigned_task_auto_assigned_with_default_assignee(isolated_kanban_home):
    """Core #27145 contract: with default_assignee set, an unassigned ready
    task gets the assignment applied and dispatched on the same tick. The
    DB row is mutated (assignee column + an 'assigned' event)."""
    kb, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            default_assignee="default",
        )
    assert res.auto_assigned_default == [task_id]
    assert not res.skipped_unassigned
    assert len(res.spawned) == 1
    assert res.spawned[0][0] == task_id
    assert res.spawned[0][1] == "default"

    with kb.connect_closing() as conn:
        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()
    assert row["assignee"] == "default"

    # 'assigned' event emitted for the audit trail
    with kb.connect_closing() as conn:
        evs = list(conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? AND kind = 'assigned'",
            (task_id,),
        ))
    assert len(evs) == 1
    payload = json.loads(evs[0][1])
    assert payload["assignee"] == "default"
    assert payload["source"] == "kanban.default_assignee"


def test_dry_run_with_default_assignee_reports_without_mutating(isolated_kanban_home):
    """Dry-run mode: reports what WOULD happen (task in auto_assigned_default,
    spawn entry) but does NOT mutate the DB. Operators using
    `hermes kanban dispatch --dry-run` see the routing decision before
    committing."""
    kb, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            default_assignee="default",
        )
    assert res.auto_assigned_default == [task_id]
    assert len(res.spawned) == 1
    with kb.connect_closing() as conn:
        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (task_id,)).fetchone()
    # DB unchanged — dry_run did not commit the assignment.
    assert row["assignee"] is None


def test_whitespace_default_assignee_treated_as_none(isolated_kanban_home):
    """Empty / whitespace-only default_assignee values must be treated as
    'no fallback set' so a misconfigured kanban.default_assignee=' '
    doesn't surprise operators by silently routing unassigned tasks."""
    kb, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee=None)
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            default_assignee="   ",
        )
    assert task_id in res.skipped_unassigned
    assert not res.auto_assigned_default


def test_explicitly_assigned_task_untouched_by_default_assignee(isolated_kanban_home):
    """A task with an explicit assignee must NOT be touched by the
    default_assignee logic — that fallback only applies to genuinely
    unassigned rows."""
    kb, _home = isolated_kanban_home
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(conn, title="t1", assignee="default")
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            default_assignee="someother",
        )
    assert task_id not in res.auto_assigned_default
    assert any(s[0] == task_id and s[1] == "default" for s in res.spawned)


def test_dispatch_result_has_auto_assigned_default_field():
    """Schema-level invariant: DispatchResult exposes the
    auto_assigned_default field so CLI / dashboard / gateway can surface
    the new routing decisions."""
    from hermes_cli.kanban_db import DispatchResult
    r = DispatchResult()
    assert hasattr(r, "auto_assigned_default")
    assert r.auto_assigned_default == []

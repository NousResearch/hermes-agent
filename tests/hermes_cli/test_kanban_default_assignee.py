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
from pathlib import Path

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


def test_default_assignee_drops_skills_that_profile_disables(isolated_kanban_home):
    """The fallback profile's disabled skills are stripped as it is applied.

    An unassigned card gave ``create_task`` no assignee to filter its pinned
    skills against, so the check lands here instead — otherwise the dispatcher
    spawns a worker whose every pinned skill is missing, which raises before
    any work happens.
    """
    kb, home = isolated_kanban_home
    profile_dir = Path(home) / "profiles" / "researcher"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text(
        "skills:\n  disabled:\n    - arxiv\n", encoding="utf-8"
    )
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(
            conn, title="t1", assignee=None, skills=["arxiv", "translation"]
        )
        # Nothing was known to be disabled at creation: there was no assignee.
        assert kb.get_task(conn, task_id).skills == ["arxiv", "translation"]

    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=False,
            default_assignee="researcher",
        )
    assert res.auto_assigned_default == [task_id]

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    assert task.assignee == "researcher"
    assert task.skills == ["translation"]


def test_dry_run_default_assignee_leaves_skills_alone(isolated_kanban_home):
    """Dry-run reports the routing without stripping anything from the card."""
    kb, home = isolated_kanban_home
    profile_dir = Path(home) / "profiles" / "researcher"
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "config.yaml").write_text(
        "skills:\n  disabled:\n    - arxiv\n", encoding="utf-8"
    )
    with kb.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        task_id = kb.create_task(
            conn, title="t1", assignee=None, skills=["arxiv", "translation"]
        )
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn, spawn_fn=_fake_spawn, dry_run=True,
            default_assignee="researcher",
        )
    assert res.auto_assigned_default == [task_id]
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    assert task.assignee is None
    assert task.skills == ["arxiv", "translation"]





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



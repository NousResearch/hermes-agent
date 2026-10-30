"""Tests for kanban lifecycle plugin hooks.

Verifies that claim/complete/block transitions fire the
kanban_task_claimed / kanban_task_completed / kanban_task_blocked plugin
hooks AFTER the board DB change is committed, with the documented kwargs,
and that a misbehaving hook callback never breaks the transition.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import sqlite3

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def captured_hooks(monkeypatch):
    """Register capturing callbacks for the three kanban lifecycle hooks.

    Patches the plugin manager's _hooks dict directly (the same registry
    invoke_hook reads) and restores it afterward.
    """
    mgr = get_plugin_manager()
    events: list[tuple[str, dict]] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    for hook in ("kanban_task_claimed", "kanban_task_completed", "kanban_task_blocked"):
        mgr._hooks.setdefault(hook, []).append(
            lambda _h=hook, **kw: events.append((_h, kw))
        )
    try:
        yield events
    finally:
        mgr._hooks = saved




def test_claim_fires_hook(kanban_home, captured_hooks):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_claimed"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "worker"
    assert "profile_name" in kw
    assert kw["run_id"] is not None




@pytest.mark.parametrize("kind", ["dependency", "needs_input"])
def test_block_hook_can_write_committed_board(kanban_home, monkeypatch, kind):
    """A board-writing plugin must not deadlock its caller (#145885)."""
    from hermes_cli.kanban_db_notify import add_notify_sub, list_notify_subs

    observations = []
    mgr = get_plugin_manager()
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="child", assignee="worker")
        assert kb.claim_task(conn, task_id) is not None
        running_task = kb.get_task(conn, task_id)
        assert running_task is not None
        run_id = running_task.current_run_id
        parent = kb.create_task(conn, title="parent", assignee="worker")
        kb.link_tasks(conn, parent_id=parent, child_id=task_id, expected_child_run_id=run_id)
        db_path = conn.execute("PRAGMA database_list").fetchone()[2]

        def subscribe(**kwargs):
            # Fail immediately if the transition still owns SQLite's writer lock.
            # The real hook boundary swallows failures, so assert receipts outside it.
            with closing(sqlite3.connect(db_path, timeout=0, isolation_level=None)) as observer:
                observer.row_factory = sqlite3.Row
                observer.execute("BEGIN IMMEDIATE")
                observer.commit()
                task = kb.get_task(observer, task_id)
                assert task is not None
                events = kb.list_events(observer, task_id)
                add_notify_sub(observer, task_id=task_id, platform="telegram", chat_id="123")
                observations.append((task.status, task.current_run_id, events[-1].kind, kwargs))

        monkeypatch.setitem(mgr._hooks, "kanban_task_blocked", [subscribe])
        assert kb.block_task(conn, task_id, reason="waiting", kind=kind, expected_run_id=run_id)
        expected_status, expected_event = (
            ("todo", "dependency_wait") if kind == "dependency" else ("blocked", "blocked")
        )
        assert len(observations) == 1
        status, current_run, event, kwargs = observations[0]
        assert (status, current_run, event) == (expected_status, None, expected_event)
        assert kwargs["task_id"] == task_id
        assert kwargs["run_id"] == run_id
        assert kwargs["assignee"] == "worker"
        assert kwargs["reason"] == "waiting"
        assert len(list_notify_subs(conn, task_id)) == 1
        assert not kb.block_task(conn, task_id, kind=kind, expected_run_id=run_id)
        assert len(observations) == 1


def test_misbehaving_hook_does_not_break_transition(kanban_home, monkeypatch):
    """A hook callback that raises must not break the board transition."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    mgr._hooks.setdefault("kanban_task_completed", []).append(_boom)
    try:
        conn = kbc.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            kb.claim_task(conn, tid)
            # Despite the raising hook, completion succeeds and persists.
            assert kb.complete_task(conn, tid, summary="ok") is True
            assert kb.get_task(conn, tid).status == "done"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved

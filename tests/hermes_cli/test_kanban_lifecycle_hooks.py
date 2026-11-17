"""Tests for kanban lifecycle plugin hooks.

Verifies that claim/complete/block transitions fire the
kanban_task_claimed / kanban_task_completed / kanban_task_blocked plugin
hooks AFTER the board DB change is committed, with the documented kwargs,
and that a misbehaving hook callback never breaks the transition.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

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


@pytest.mark.parametrize(("kind", "status", "event"), [
    ("dependency", "todo", "dependency_wait"),
    ("needs_input", "blocked", "blocked"),  # control: always post-commit
])
def test_block_hook_runs_after_commit_with_write_lock_free(
    kanban_home, captured_hooks, kind, status, event,
):
    """Every block kind fires kanban_task_blocked after its txn commits: a
    second connection sees the transition and can take the write lock."""
    db_path = kb.kanban_db_path()
    seen = []

    def _probe(**_kw):
        other = sqlite3.connect(str(db_path), timeout=0, isolation_level=None)
        try:
            row = other.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
            last = other.execute(
                "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,),
            ).fetchone()
            try:
                other.execute("BEGIN IMMEDIATE")
                other.execute("ROLLBACK")
                lock = "free"
            except sqlite3.OperationalError as exc:
                lock = str(exc)
            seen.append((row[0], last[0], lock))
        finally:
            other.close()

    get_plugin_manager()._hooks["kanban_task_blocked"].append(_probe)
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="child", assignee="worker")
        kb.link_tasks(conn, parent_id=kb.create_task(conn, title="parent"), child_id=tid)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
        assert kb.block_task(conn, tid, reason="waiting", kind=kind)
    finally:
        conn.close()
    assert seen == [(status, event, "free")]


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

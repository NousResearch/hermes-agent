"""Tests for kanban lifecycle plugin hooks.

Verifies that claim/complete/block transitions fire the
kanban_task_claimed / kanban_task_completed / kanban_task_blocked plugin
hooks AFTER the board DB change is committed, with the documented kwargs,
and that a misbehaving hook callback never breaks the transition.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
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
            assert kb.complete_task(conn, tid, summary="ok") is True
            assert kb.get_task(conn, tid).status == "done"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved


def test_pre_create_suppress_prevents_row_write(kanban_home):
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_kanban_task_create", []).append(
        lambda **kw: {"action": "suppress", "reason": "duplicate", "existing_task_id": "existing"}
    )
    try:
        conn = kbc.connect()
        try:
            assert kb.create_task(conn, title="blocked") == "existing"
            assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0
        finally:
            conn.close()
    finally:
        mgr._hooks = saved


def test_pre_dispatch_hold_records_respawn_guard(kanban_home, monkeypatch):
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_kanban_dispatch", []).append(
        lambda **kw: {"action": "hold", "reason": "operator hold"}
    )
    try:
        conn = kbc.connect()
        try:
            monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
            tid = kb.create_task(conn, title="held", assignee="worker")
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
            result = kbd.dispatch_once(conn, board=None)
            assert result.respawn_guarded == [(tid, "operator hold")]
            assert kb.get_task(conn, tid).status == "ready"
            assert conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1", (tid,)
            ).fetchone()[0] == "respawn_guarded"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved


@pytest.mark.parametrize(
    "directive",
    [None, "not-a-dict", {"action": "unknown", "reason": "bad"}, {"action": "suppress"}],
)
def test_pre_create_invalid_directives_are_noops(kanban_home, directive):
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault("pre_kanban_task_create", []).append(lambda **kw: directive)
    try:
        conn = kbc.connect()
        try:
            tid = kb.create_task(conn, title="normal")
            assert tid
            assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1
        finally:
            conn.close()
    finally:
        mgr._hooks = saved


def test_pre_kanban_hooks_are_noops_without_subscriber(kanban_home, monkeypatch):
    conn = kbc.connect()
    try:
        monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)
        tid = kb.create_task(conn, title="normal", assignee="worker")
        result = kbd.dispatch_once(conn, dry_run=True)
        assert result.respawn_guarded == []
        assert result.spawned and result.spawned[0][0] == tid
    finally:
        conn.close()

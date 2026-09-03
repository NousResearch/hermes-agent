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
    """Register capturing callbacks for the kanban lifecycle hooks.

    Patches the plugin manager's _hooks dict directly (the same registry
    invoke_hook reads) and restores it afterward.
    """
    mgr = get_plugin_manager()
    events: list[tuple[str, dict]] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    for hook in (
        "kanban_task_claimed",
        "kanban_task_completed",
        "kanban_task_blocked",
        "on_kanban_task_auto_blocked",
    ):
        mgr._hooks.setdefault(hook, []).append(
            lambda _h=hook, **kw: events.append((_h, kw))
        )
    try:
        yield events
    finally:
        mgr._hooks = saved


def test_circuit_breaker_fires_post_commit_auto_block_hook(
    kanban_home, captured_hooks
):
    """The breaker observer sees the committed blocked state and cause."""
    durable_statuses: list[str] = []
    mgr = get_plugin_manager()

    def _read_durable_state(**kw):
        c2 = kb.connect()
        try:
            task = kb.get_task(c2, kw["task_id"])
            durable_statuses.append(task.status if task else "missing")
        finally:
            c2.close()

    mgr._hooks.setdefault("on_kanban_task_auto_blocked", []).append(
        _read_durable_state
    )
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker", max_retries=1)
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None

        assert kb._record_task_failure(
            conn,
            tid,
            "worker failed",
            outcome="spawn_failed",
            failure_limit=5,
            release_claim=True,
            end_run=True,
        )
    finally:
        conn.close()

    fired = [
        event for event in captured_hooks
        if event[0] == "on_kanban_task_auto_blocked"
    ]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["run_id"] == claimed.current_run_id
    assert kw["outcome"] == "spawn_failed"
    assert kw["error"] == "worker failed"
    assert kw["error_fingerprint"]
    assert kw["consecutive_failures"] == 1
    assert kw["failure_limit"] == 1
    assert kw["retry_status"] == "ready"
    assert kw["status"] == "blocked"
    assert durable_statuses == ["blocked"]




def test_claim_fires_hook(kanban_home, captured_hooks):
    conn = kb.connect()
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
        conn = kb.connect()
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

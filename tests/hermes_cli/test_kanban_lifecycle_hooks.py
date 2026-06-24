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
    """Register capturing callbacks for the three kanban lifecycle hooks.

    Patches the plugin manager's _hooks dict directly (the same registry
    invoke_hook reads) and restores it afterward.
    """
    mgr = get_plugin_manager()
    events: list[tuple[str, dict]] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    for hook in ("kanban_task_claimed", "kanban_task_completed", "kanban_task_blocked"):
        def _capture(_h=hook, **kw):
            # Read through a separate connection so the test proves the hook
            # fires after the transition is durably committed, not while the
            # SQLite write transaction is still open.
            with kb.connect() as check_conn:
                task = kb.get_task(check_conn, kw["task_id"])
            kw["_observed_status"] = task.status if task else None
            events.append((_h, kw))

        mgr._hooks.setdefault(hook, []).append(_capture)
    try:
        yield events
    finally:
        mgr._hooks = saved


def test_hooks_are_registered_as_valid():
    """The three lifecycle hook names are part of VALID_HOOKS."""
    assert "kanban_task_claimed" in VALID_HOOKS
    assert "kanban_task_completed" in VALID_HOOKS
    assert "kanban_task_blocked" in VALID_HOOKS


def test_claim_fires_hook(kanban_home, captured_hooks):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="work on smoke", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_claimed"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["board"] == "default"
    assert kw["assignee"] == "worker"
    assert kw["title"] == "work on smoke"
    assert "profile_name" in kw
    assert kw["run_id"] is not None
    assert kw["_observed_status"] == "running"


def test_complete_fires_hook_with_summary(kanban_home, captured_hooks):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="ship summary", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.complete_task(conn, tid, summary="all done")
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_completed"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["board"] == "default"
    assert kw["title"] == "ship summary"
    assert kw["summary"] == "all done"
    assert kw["assignee"] == "worker"
    assert kw["run_id"] is not None
    assert kw["_observed_status"] == "done"


def test_block_fires_hook_with_reason(kanban_home, captured_hooks):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="ask human", assignee="worker")
        kb.claim_task(conn, tid)
        assert kb.block_task(conn, tid, reason="needs human")
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_blocked"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["board"] == "default"
    assert kw["title"] == "ask human"
    assert kw["reason"] == "needs human"
    assert kw["assignee"] == "worker"
    assert kw["run_id"] is not None
    assert kw["_observed_status"] == "blocked"


def test_lifecycle_title_is_redacted_and_truncated(kanban_home, captured_hooks):
    conn = kb.connect()
    secret = "sk-" + "a" * 48
    bearer = "eyJ" + "b" * 48
    basic = "dXNlcjpwYXNz"
    try:
        tid = kb.create_task(
            conn,
            title=(
                f"deploy Authorization: Basic {basic} Bearer {bearer} "
                f"Basic {basic} token={secret} "
            ) + "x" * 260,
            assignee="worker",
        )
        assert kb.claim_task(conn, tid) is not None
    finally:
        conn.close()
    kw = [e for e in captured_hooks if e[0] == "kanban_task_claimed"][0][1]
    title = kw["title"]
    assert "[REDACTED]" in title
    assert secret not in title
    assert bearer not in title
    assert basic not in title
    assert len(title) <= 200


def test_review_claim_fires_hook_with_title(kanban_home, captured_hooks):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="review this", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
        claimed = kb.claim_review_task(conn, tid)
        assert claimed is not None
    finally:
        conn.close()
    fired = [e for e in captured_hooks if e[0] == "kanban_task_claimed"]
    assert len(fired) == 1
    kw = fired[0][1]
    assert kw["task_id"] == tid
    assert kw["title"] == "review this"
    assert kw["_observed_status"] == "running"


def test_no_hook_on_failed_transition(kanban_home, captured_hooks):
    """complete_task on an unclaimed/nonexistent task fires no hook."""
    conn = kb.connect()
    try:
        # Completing a task that doesn't exist returns False without firing.
        assert kb.complete_task(conn, "t_doesnotexist", summary="x") is False
    finally:
        conn.close()
    assert [e for e in captured_hooks if e[0] == "kanban_task_completed"] == []


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


@pytest.mark.parametrize(
    ("hook_name", "transition"),
    [
        ("kanban_task_claimed", "claim"),
        ("kanban_task_completed", "complete"),
        ("kanban_task_blocked", "block"),
    ],
)
def test_misbehaving_lifecycle_hooks_fail_open_for_every_transition(
    kanban_home,
    hook_name,
    transition,
):
    """A broken observer must never prevent claim, complete, or block."""
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def _boom(**kw):
        raise RuntimeError("plugin exploded")

    mgr._hooks.setdefault(hook_name, []).append(_boom)
    try:
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            if transition == "claim":
                assert kb.claim_task(conn, tid) is not None
                task = kb.get_task(conn, tid)
                assert task is not None
                assert task.status == "running"
            else:
                assert kb.claim_task(conn, tid) is not None
                if transition == "complete":
                    assert kb.complete_task(conn, tid, summary="ok") is True
                    task = kb.get_task(conn, tid)
                    assert task is not None
                    assert task.status == "done"
                else:
                    assert kb.block_task(conn, tid, reason="needs human") is True
                    task = kb.get_task(conn, tid)
                    assert task is not None
                    assert task.status == "blocked"
        finally:
            conn.close()
    finally:
        mgr._hooks = saved

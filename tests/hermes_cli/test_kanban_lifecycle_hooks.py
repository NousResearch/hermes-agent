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
        mgr._hooks.setdefault(hook, []).append(
            lambda _h=hook, **kw: events.append((_h, kw))
        )
    try:
        yield events
    finally:
        mgr._hooks = saved




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




def test_review_claim_fires_hook(kanban_home, captured_hooks):
    """The review lane is dispatched from the same process as the ready lane.

    ``kanban_task_claimed`` is documented as firing "after claim commit, in
    dispatcher process before worker spawn" so a plugin registered in the
    dispatcher observes every transition centrally. ``claim_review_task`` is
    a dispatcher-side claim too, so a reviewer spawn must be observable the
    same way an implementation spawn is.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        implementation = kb.claim_task(conn, tid)
        assert implementation is not None
        assert kb.request_review(
            conn,
            tid,
            reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
        )
        review = kb.claim_review_task(conn, tid, claimer="reviewer:1")
        assert review is not None
    finally:
        conn.close()

    fired = [e for e in captured_hooks if e[0] == "kanban_task_claimed"]
    # One for the implementation claim, one for the review claim.
    assert len(fired) == 2
    kw = fired[1][1]
    assert kw["task_id"] == tid
    assert kw["assignee"] == "reviewer"
    assert "profile_name" in kw
    assert kw["run_id"] == review.current_run_id


def test_failed_review_claim_does_not_fire_hook(kanban_home, captured_hooks):
    """A refused claim is not a spawn, so it must stay silent.

    The second claim loses the race (the task is already ``running`` under
    the first reviewer's lock) and returns None — firing the hook there
    would tell an observer a reviewer started when none did.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="worker")
        implementation = kb.claim_task(conn, tid)
        assert implementation is not None
        assert kb.request_review(
            conn,
            tid,
            reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
        )
        assert kb.claim_review_task(conn, tid, claimer="reviewer:1") is not None
        assert kb.claim_review_task(conn, tid, claimer="reviewer:2") is None
    finally:
        conn.close()

    fired = [e for e in captured_hooks if e[0] == "kanban_task_claimed"]
    # Implementation claim + the one review claim that actually won.
    assert len(fired) == 2


def test_review_claim_demoted_by_reopened_parent_does_not_fire_hook(
    kanban_home, captured_hooks
):
    """The dependency-guard exit is a demotion, not a spawn.

    When a parent is reopened while the child waits in review,
    ``claim_review_task`` sends the child back to ``todo`` and returns None.
    That path writes to the board, so it is easy to mistake for a
    transition worth announcing — but no reviewer process starts, and
    ``claim_task``'s matching ``parents_not_done`` exit is silent too.
    """
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent", assignee="planner")
        assert kb.complete_task(conn, parent)
        tid = kb.create_task(
            conn, title="t", assignee="worker", parents=[parent]
        )
        implementation = kb.claim_task(conn, tid)
        assert implementation is not None
        assert kb.request_review(
            conn,
            tid,
            reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
        )
        # Reopen the parent behind the child's back.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status = 'ready' WHERE id = ?", (parent,)
            )
        assert kb.claim_review_task(conn, tid, claimer="reviewer:1") is None
        demoted = kb.get_task(conn, tid)
        assert demoted is not None
        assert demoted.status == "todo"
    finally:
        conn.close()

    fired = [e for e in captured_hooks if e[0] == "kanban_task_claimed"]
    # Only the implementation claim; the refused review claim stays silent.
    assert len(fired) == 1
    assert fired[0][1]["task_id"] == tid


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

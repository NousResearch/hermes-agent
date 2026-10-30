"""Regression tests for four kanban decomposer / state-transition fixes.

Fix A — decompose_triage_task blocks when root worktree is dirty.
Fix B — decompose_task does NOT reassign the root task's assignee.
Fix C — promote_task(force=True) bypasses the claim gate (parents_not_done).
Fix D — unblock_task accepts 'todo' status and promotes to 'ready' when
         all parents are done.

No real DB connections; all tests use the isolated ``kanban_home`` fixture.
No live git subprocesses; worktree checks are monkey-patched.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _triage(conn, title="rough idea", assignee="original-assignee"):
    return kb.create_task(conn, title=title, assignee=assignee, triage=True)


def _get(conn, tid):
    """Unwrapping helper — raises if task is missing."""
    t = kb.get_task(conn, tid)
    assert t is not None, f"task {tid} not found"
    return t


# ---------------------------------------------------------------------------
# Fix A — dirty worktree blocks decompose, adds audit comment
# ---------------------------------------------------------------------------

class TestFixA:
    """decompose_triage_task refuses to fan out when root worktree is dirty."""

    def test_dirty_worktree_blocks_decompose_and_adds_comment(self, kanban_home, tmp_path):
        """When _worktree_has_active_work returns True, decompose returns None
        and leaves an audit comment on the root task."""
        fake_ws = str(tmp_path / "fake_ws")
        with kb.connect() as conn:
            tid = kb.create_task(
                conn,
                title="a task with worktree",
                assignee="kamiyu",
                triage=True,
                workspace_kind="worktree",
                workspace_path=fake_ws,
            )

        children = [{"title": "child one", "assignee": "worker"}]
        with patch(
            "hermes_cli.kanban_db._worktree_has_active_work",
            return_value=True,
        ):
            with kb.connect() as conn:
                result = kb.decompose_triage_task(
                    conn,
                    tid,
                    root_assignee=None,
                    children=children,
                    author="decomposer",
                )

        assert result is None, "decompose must return None when worktree is dirty"

        with kb.connect() as conn:
            task = _get(conn, tid)
            comments = conn.execute(
                "SELECT body FROM task_comments WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                (tid,),
            ).fetchone()
            events = conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
                (tid,),
            ).fetchone()

        assert task.status == "triage", "root must stay in triage when decompose is blocked"
        assert comments is not None, "an audit comment should have been added"
        assert "Decomposition blocked" in comments["body"]
        assert events is not None
        assert events["kind"] == "decompose_blocked_dirty_worktree"

    def test_clean_worktree_allows_decompose(self, kanban_home, tmp_path):
        """When _worktree_has_active_work returns False, decompose proceeds normally."""
        fake_ws = str(tmp_path / "fake_ws")
        with kb.connect() as conn:
            tid = kb.create_task(
                conn,
                title="a task with clean worktree",
                assignee="kamiyu",
                triage=True,
                workspace_kind="worktree",
                workspace_path=fake_ws,
            )

        children = [{"title": "child one", "assignee": "worker"}]
        with patch(
            "hermes_cli.kanban_db._worktree_has_active_work",
            return_value=False,
        ):
            with kb.connect() as conn:
                result = kb.decompose_triage_task(
                    conn,
                    tid,
                    root_assignee=None,
                    children=children,
                    author="decomposer",
                )

        assert result is not None, "decompose must succeed when worktree is clean"
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Fix B — root assignee is NOT changed by decompose_triage_task(root_assignee=None)
# ---------------------------------------------------------------------------

class TestFixB:
    """Decomposition with root_assignee=None must preserve the existing assignee."""

    def test_root_assignee_unchanged_when_none(self, kanban_home):
        """root_assignee=None leaves the root task's assignee intact."""
        with kb.connect() as conn:
            tid = _triage(conn, assignee="original-kamiyu")

        children = [{"title": "child", "assignee": "worker"}]
        with kb.connect() as conn:
            result = kb.decompose_triage_task(
                conn,
                tid,
                root_assignee=None,
                children=children,
                author="decomposer",
            )

        assert result is not None
        with kb.connect() as conn:
            task = _get(conn, tid)
        assert task.assignee == "original-kamiyu", (
            "root assignee must be preserved when root_assignee=None"
        )

    def test_root_assignee_overwritten_when_given(self, kanban_home):
        """When root_assignee is explicitly provided it DOES overwrite the assignee
        (this behaviour is intentional and used by tests that need it)."""
        with kb.connect() as conn:
            tid = _triage(conn, assignee="original-kamiyu")

        children = [{"title": "child", "assignee": "worker"}]
        with kb.connect() as conn:
            result = kb.decompose_triage_task(
                conn,
                tid,
                root_assignee="amuro",
                children=children,
                author="decomposer",
            )

        assert result is not None
        with kb.connect() as conn:
            task = _get(conn, tid)
        assert task.assignee == "amuro"


# ---------------------------------------------------------------------------
# Fix C — promote_task(force=True) bypasses the claim gate
# ---------------------------------------------------------------------------

class TestFixC:
    """promote --force must set force_promoted so the claim gate lets it through."""

    def test_force_promote_bypasses_parents_not_done_gate(self, kanban_home):
        """A task with an unfinished parent can be force-promoted AND claimed."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent")
            child = kb.create_task(conn, title="child", parents=[parent])
            assert _get(conn, child).status == "todo"

        # force-promote the child
        with kb.connect() as conn:
            ok, err = kb.promote_task(conn, child, actor="operator", force=True)
        assert ok, f"promote --force failed: {err}"

        with kb.connect() as conn:
            t = _get(conn, child)
        assert t.status == "ready"
        assert t.force_promoted == 1

        # claim must succeed (bypassing the parents_not_done guard)
        with kb.connect() as conn:
            claimed = kb.claim_task(conn, child, claimer="dispatcher")
        assert claimed is not None, (
            "claim must succeed when force_promoted=1, even with undone parents"
        )

        # flag must be cleared after the claim
        with kb.connect() as conn:
            assert _get(conn, child).force_promoted == 0

    def test_force_promoted_flag_cleared_after_claim(self, kanban_home):
        """force_promoted is reset to 0 on first claim (one-shot bypass)."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent")
            child = kb.create_task(conn, title="child", parents=[parent])
            kb.promote_task(conn, child, actor="operator", force=True)

        with kb.connect() as conn:
            kb.claim_task(conn, child, claimer="worker-1")
            assert _get(conn, child).force_promoted == 0

    def test_normal_promote_without_force_still_blocked_by_parent_gate(self, kanban_home):
        """Without force=True the claim gate still rejects tasks with undone parents."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent")
            child = kb.create_task(conn, title="child", parents=[parent])
            ok, err = kb.promote_task(conn, child, actor="operator", force=False)
        assert not ok
        assert "unsatisfied" in (err or "")

    def test_force_promote_without_undone_parents_also_succeeds(self, kanban_home):
        """force=True with no undone parents still works fine (no regression)."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent")
            child = kb.create_task(conn, title="child", parents=[parent])
            kb.complete_task(conn, parent, summary="done")
            # child was promoted to ready by recompute_ready; demote manually
            conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (child,))

        with kb.connect() as conn:
            ok, err = kb.promote_task(conn, child, actor="operator", force=True)
        assert ok, err

        with kb.connect() as conn:
            claimed = kb.claim_task(conn, child, claimer="dispatcher")
        assert claimed is not None


# ---------------------------------------------------------------------------
# Fix D — unblock_task accepts 'todo' and promotes to 'ready' when parents done
# ---------------------------------------------------------------------------

class TestFixD:
    """unblock_task must also accept todo-status tasks."""

    def test_unblock_todo_with_all_parents_done_promotes_to_ready(self, kanban_home):
        """A task in 'todo' with all parents done becomes 'ready' on unblock."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent")
            child = kb.create_task(conn, title="child", parents=[parent])
            assert _get(conn, child).status == "todo"
            kb.complete_task(conn, parent, summary="done")
            # After parent completes recompute_ready promotes child; manually
            # demote it back to 'todo' to simulate the claim-gate demotion.
            conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (child,))

        with kb.connect() as conn:
            ok = kb.unblock_task(conn, child)
        assert ok is True

        with kb.connect() as conn:
            assert _get(conn, child).status == "ready", (
                "unblock_task must promote a 'todo' task to 'ready' "
                "when all parents are done"
            )

    def test_unblock_todo_with_pending_parents_stays_todo(self, kanban_home):
        """A task in 'todo' with undone parents stays in 'todo' after unblock."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent")
            child = kb.create_task(conn, title="child", parents=[parent])
            assert _get(conn, child).status == "todo"

        with kb.connect() as conn:
            ok = kb.unblock_task(conn, child)
        assert ok is True

        with kb.connect() as conn:
            assert _get(conn, child).status == "todo"

    def test_unblock_blocked_still_works(self, kanban_home):
        """Existing blocked → ready behaviour is unaffected."""
        with kb.connect() as conn:
            t = kb.create_task(conn, title="plain task")
            kb.claim_task(conn, t, claimer="worker")
            kb.block_task(conn, t, reason="needs input")
            assert _get(conn, t).status == "blocked"

        with kb.connect() as conn:
            ok = kb.unblock_task(conn, t)
        assert ok is True

        with kb.connect() as conn:
            assert _get(conn, t).status == "ready"

"""Regression tests for #22375.

``unblock_task`` was the only path that could land a task in ``ready``
without consulting the parent-dependency gate that ``recompute_ready``
enforces.  Steps to reproduce from the bug report:

1. Create parent task A.
2. Create child task B with ``parents=[A]``; B starts as ``todo``.
3. Force B into ``blocked`` (skipping the normal ``ready -> running ->
   blocked`` path).
4. ``unblock_task(B)`` jumped B to ``ready`` even though A is still
   ``todo``, and the dispatcher could claim B before A was done.

The fix routes ``unblock_task`` through the same parent-status check
used by ``recompute_ready``: land in ``ready`` only when every parent is
``done``, otherwise land in ``todo`` so the next dispatcher tick re-runs
the gate.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _force_blocked(conn, task_id: str) -> None:
    """Drop a task into ``blocked`` directly.

    Mirrors the bug-report scenario where a child task ends up ``blocked``
    while its parent is still ``todo`` — the legitimate API path requires
    ``ready -> running -> blocked``, which itself implies the parent gate
    was already cleared, so we bypass it for the regression case.
    """
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'blocked' WHERE id = ?",
            (task_id,),
        )


class TestUnblockParentGate:
    def test_unblock_with_open_parent_lands_in_todo(self, kanban_home):
        """Bug from #22375: child must NOT jump to ready while parent is open."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent")
            child = kb.create_task(conn, title="child", parents=[parent])
            assert kb.get_task(conn, parent).status == "ready"
            assert kb.get_task(conn, child).status == "todo"

            _force_blocked(conn, child)
            assert kb.get_task(conn, child).status == "blocked"

            assert kb.unblock_task(conn, child) is True
            assert kb.get_task(conn, child).status == "todo"

    def test_unblock_with_done_parent_lands_in_ready(self, kanban_home):
        """All parents done → unblock promotes straight to ready."""
        with kb.connect() as conn:
            parent = kb.create_task(conn, title="parent", assignee="a")
            child = kb.create_task(conn, title="child", parents=[parent])
            kb.claim_task(conn, parent)
            kb.complete_task(conn, parent, result="ok")
            kb.recompute_ready(conn)
            assert kb.get_task(conn, child).status == "ready"

            _force_blocked(conn, child)
            assert kb.unblock_task(conn, child) is True
            assert kb.get_task(conn, child).status == "ready"

    def test_unblock_with_no_parents_lands_in_ready(self, kanban_home):
        """No-parents case keeps the original behaviour."""
        with kb.connect() as conn:
            t = kb.create_task(conn, title="solo", assignee="a")
            kb.claim_task(conn, t)
            assert kb.block_task(conn, t, reason="need input")
            assert kb.unblock_task(conn, t) is True
            assert kb.get_task(conn, t).status == "ready"

    def test_unblock_with_partially_done_parents_lands_in_todo(self, kanban_home):
        """Some parents done, others still open → todo (not ready)."""
        with kb.connect() as conn:
            p1 = kb.create_task(conn, title="p1", assignee="a")
            p2 = kb.create_task(conn, title="p2", assignee="b")
            child = kb.create_task(conn, title="c", parents=[p1, p2])
            kb.claim_task(conn, p1)
            kb.complete_task(conn, p1, result="ok")
            assert kb.get_task(conn, child).status == "todo"

            _force_blocked(conn, child)
            assert kb.unblock_task(conn, child) is True
            assert kb.get_task(conn, child).status == "todo"

    def test_unblock_returns_false_when_not_blocked(self, kanban_home):
        """Original contract preserved: False when row isn't in blocked."""
        with kb.connect() as conn:
            t = kb.create_task(conn, title="x")
            assert kb.unblock_task(conn, t) is False
            assert kb.get_task(conn, t).status == "ready"

"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking.
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each same-cause re-block after an unblock increments ``block_recurrences``,
  and at ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` resets the loop memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


# ---------------------------------------------------------------------------
# Loop breaker
# ---------------------------------------------------------------------------










def test_block_loop_detected_event_emitted(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="capability")
        events = [e for e in kb.list_events(conn, tid)
                  if e.kind == "block_loop_detected"]
        assert events, "expected a block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") == "capability"


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_parentless_dependency_wait_not_auto_promoted(kanban_home: Path) -> None:
    """Regression for the rapid re-dispatch loop (t_cb67c890).

    A parentless worker that calls ``kanban_block(kind="dependency")`` lands
    in ``todo`` with ``block_kind='dependency'`` and no parent links.
    ``recompute_ready`` treated ``all([])`` as vacuously true and promoted
    it on the very next dispatch tick, respawning the worker which would
    block again — six blocked runs, four in ~4 minutes.  The fix parks a
    dependency-wait task with no parents in ``todo`` until a parent link is
    added or an operator deliberately changes/promotes the task.
    """
    with kb.connect_closing() as conn:
        tid = _running_task(conn, title="parentless dep wait")
        assert kb.block_task(conn, tid, reason="waiting on X", kind="dependency")
        t = kb.get_task(conn, tid)
        assert t.status == "todo"
        assert t.block_kind == "dependency"

        # Simulate multiple dispatcher ticks — must NOT promote.
        for _ in range(5):
            promoted = kb.recompute_ready(conn)
            assert promoted == 0, (
                "parentless dependency-wait must not auto-promote "
                "(would respawn immediately and loop)"
            )
            assert kb.get_task(conn, tid).status == "todo"


def test_parentless_dependency_wait_promotes_after_link(kanban_home: Path) -> None:
    """A parentless dependency-wait that later gets a parent link must
    resume normal parent-gated promotion once that parent completes."""
    with kb.connect_closing() as conn:
        child = _running_task(conn, title="child")
        kb.block_task(conn, child, reason="waiting", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"

        # Still no promotion before a parent exists.
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "todo"

        # Add a parent and complete it.
        parent = kb.create_task(conn, title="parent", assignee="worker")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_parentless_dependency_wait_promotes_when_linked_to_completed_parent(
    kanban_home: Path,
) -> None:
    """Linking an already-completed parent also resumes normal promotion."""
    with kb.connect_closing() as conn:
        child = _running_task(conn, title="child")
        kb.block_task(conn, child, reason="waiting", kind="dependency")
        assert kb.recompute_ready(conn) == 0

        parent = _running_task(conn, title="completed parent")
        kb.complete_task(conn, parent, result="done")
        parent_task = kb.get_task(conn, parent)
        assert parent_task is not None
        assert parent_task.status == "done"

        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.recompute_ready(conn)
        child_task = kb.get_task(conn, child)
        assert child_task is not None
        assert child_task.status == "ready"


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------



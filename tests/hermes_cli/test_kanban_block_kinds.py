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
    with kb.connect_closing() as conn:
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
    with kb.connect_closing() as conn:
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


def test_dependency_block_names_sibling_creates_parent_link(kanban_home: Path) -> None:
    """A dependency block naming sibling X links the blocked card to X."""
    with kb.connect_closing() as conn:
        sibling = kb.create_task(conn, title="sibling", assignee="worker")
        child = _running_task(conn, title="child")
        kb.block_task(
            conn, child,
            reason=f"waiting on sibling {sibling}", kind="dependency",
        )
        assert kb.get_task(conn, child).status == "todo"
        link = conn.execute(
            "SELECT parent_id, child_id FROM task_links "
            "WHERE parent_id = ? AND child_id = ?",
            (sibling, child),
        ).fetchone()
        assert link is not None, "expected parent link child -> sibling"
        ev = [e for e in kb.list_events(conn, child)
              if e.kind == "dependency_wait"][-1]
        payload = ev.payload or {}
        assert payload.get("sibling_id") == sibling
        assert payload.get("sibling_resolved") is True
        assert payload.get("link_created") is True


def test_dependency_block_unknown_sibling_fails_safe(kanban_home: Path) -> None:
    """An unresolvable sibling id still blocks, with alert metadata recorded."""
    with kb.connect_closing() as conn:
        child = _running_task(conn, title="child")
        kb.block_task(
            conn, child,
            reason="waiting on t_ffffffff (missing)", kind="dependency",
        )
        assert kb.get_task(conn, child).status == "todo"
        assert conn.execute(
            "SELECT 1 FROM task_links WHERE child_id = ?", (child,)
        ).fetchone() is None
        ev = [e for e in kb.list_events(conn, child)
              if e.kind == "dependency_wait"][-1]
        payload = ev.payload or {}
        assert payload.get("sibling_resolved") is False
        assert payload.get("link_created") is False


def test_dependency_block_sibling_gates_recompute_ready(kanban_home: Path) -> None:
    """A dependency-blocked card stays parked until the named sibling completes."""
    with kb.connect_closing() as conn:
        sibling = kb.create_task(conn, title="sibling", assignee="worker")
        child = _running_task(conn, title="child")
        kb.block_task(
            conn, child,
            reason=f"needs {sibling} first", kind="dependency",
        )
        assert kb.get_task(conn, child).status == "todo"
        # Sibling still incomplete: recompute_ready must NOT re-promote.
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"
        # Sibling completes: the card may now promote.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (sibling,))
        kb.claim_task(conn, sibling, claimer="worker")
        kb.complete_task(conn, sibling, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------



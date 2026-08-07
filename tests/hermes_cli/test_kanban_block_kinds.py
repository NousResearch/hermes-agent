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


# ---------------------------------------------------------------------------
# Vacuous-satisfaction guard (t_4f14b90d)
#
# recompute_ready gates a ``todo`` card on ``all(p in {done,archived} for p in
# parents)``. Python's ``all([])`` is vacuously True, so a card with ZERO
# parent links passes the gate. That is correct for a standalone task, but a
# card that only landed in ``todo`` via ``kanban_block(kind="dependency")``
# relies on its parent link to re-gate it. If that link is transiently dropped
# (re-decompose / unlink->recompute churn), the dependency-waiting card was
# promoted ``todo -> ready`` on an EMPTY parent set (``satisfied_parent_ids:
# []``), got claimed, re-declared the SAME dependency wait, and thrashed on a
# ~5-10 min respawn cadence — burning a worker every cycle with zero forward
# progress (live repro t_7cf95f5a on t_3365a1dd).
# ---------------------------------------------------------------------------


def test_dependency_wait_with_missing_parent_stays_todo(kanban_home: Path) -> None:
    """A dependency-waiting card whose parent link was dropped must NOT be
    promoted on an empty parent set. It stays in ``todo`` until a real
    ``done`` parent link exists. This is the core t_4f14b90d regression:
    it FAILS (card -> ready) without the guard in recompute_ready."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"

        # Simulate link churn dropping the child's parent edge (re-decompose /
        # unlink sweep) while the parent is NOT done — exactly the live repro
        # state where satisfied_parent_ids came back empty.
        with kb.write_txn(conn):
            conn.execute(
                "DELETE FROM task_links WHERE child_id=?", (child,)
            )
        parents = conn.execute(
            "SELECT 1 FROM task_links WHERE child_id=?", (child,)
        ).fetchall()
        assert parents == [], "precondition: child has no parent link"

        promoted = kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo", (
            "dependency wait with no live done parent must stay todo, "
            "never promote on a vacuous empty parent set"
        )
        # And it must not have emitted a bogus promotion for this card.
        kinds = [e.kind for e in kb.list_events(conn, child)]
        assert "promoted" not in kinds, (
            "no promoted event for a vacuously-satisfied dependency wait"
        )
        assert isinstance(promoted, int)


def test_dependency_wait_promotes_only_when_parent_done(kanban_home: Path) -> None:
    """The guard must not over-correct: a dependency wait with an intact
    parent link still promotes exactly once, and only once the parent is
    ``done`` — not while the parent is merely blocked/todo."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"

        # Parent not done yet -> child must stay todo (real link, undone parent).
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"

        # Parent reaches done -> child promotes.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_standalone_todo_still_promotes_on_empty_parents(kanban_home: Path) -> None:
    """No over-correction: an ordinary standalone card (never a dependency
    wait) with zero parents must reach ``ready`` — the guard only fires for
    dependency-waiting cards, never for standalone ones."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="standalone", assignee="worker")
        # A standalone card with no parents is not gated. Force it back to
        # ``todo`` and confirm recompute_ready still promotes it (the guard,
        # which only fires on a dependency_wait, must leave it alone).
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (tid,))
        assert kb.get_task(conn, tid).status == "todo"
        assert not kb._is_dependency_wait(conn, tid)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "ready"


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------



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
* ``resource`` also lands in ``blocked``, but CARRIES ``block_recurrences``
  forward instead of incrementing it: a peer holding a write lease or a
  rate-limited seat is contention, not this task's own failure, so the wait
  must not spend the breaker's budget.
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


def test_resource_block_carries_recurrences_instead_of_incrementing(kanban_home: Path) -> None:
    """A ``resource`` re-block keeps the count it inherited.

    The earlier ``capability`` block leaves the counter at 1, which is what
    makes the carry observable: an incrementing path would reach
    ``BLOCK_RECURRENCE_LIMIT`` on the second ``resource`` cycle and route the
    task to ``triage``.
    """
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="needs a human", kind="capability")
        assert kb.get_task(conn, tid).block_recurrences == 1

        for cycle in (1, 2):
            kb.unblock_task(conn, tid)
            _make_running_again(conn, tid)
            kb.block_task(conn, tid, reason="write lease held by a peer", kind="resource")
            task = kb.get_task(conn, tid)
            assert task.status == "blocked", cycle
            assert task.block_kind == "resource", cycle
            assert task.block_recurrences == 1, cycle

        assert not [e for e in kb.list_events(conn, tid)
                    if e.kind == "block_loop_detected"]


def test_resource_never_trips_the_breaker_that_other_kinds_do(kanban_home: Path) -> None:
    """The same block -> unblock -> re-block cycle, differing only in the kind.

    ``capability`` is the control: at ``BLOCK_RECURRENCE_LIMIT`` it routes to
    ``triage``, which is right for a card nobody is progressing. A card
    queueing behind a write lease must not be escalated to a human for waiting
    its turn.
    """
    with kbc.connect_closing() as conn:
        for kind, expected_status, expected_recurrences in (
            ("capability", "triage", kb.BLOCK_RECURRENCE_LIMIT),
            ("resource", "blocked", 0),
        ):
            tid = _running_task(conn, title=kind)
            kb.block_task(conn, tid, reason="same cause", kind=kind)
            for _ in range(kb.BLOCK_RECURRENCE_LIMIT - 1):
                kb.unblock_task(conn, tid)
                _make_running_again(conn, tid)
                kb.block_task(conn, tid, reason="same cause", kind=kind)
            task = kb.get_task(conn, tid)
            assert task.status == expected_status, kind
            assert task.block_recurrences == expected_recurrences, kind


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


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------



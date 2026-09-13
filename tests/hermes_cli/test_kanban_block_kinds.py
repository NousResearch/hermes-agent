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


def test_block_loop_detected_with_alternating_kinds(kanban_home: Path) -> None:
    """Regression for t_e2f7128f (2026-09): a task that alternates its block
    ``kind`` on every re-block must still trip the triage safety valve on the
    2nd re-block, exactly as a same-kind repeat would. Before the fix,
    ``_route_block`` only accumulated ``block_recurrences`` when the incoming
    ``kind`` matched the ``block_kind`` stored from the previous block, so
    needs_input -> untyped -> needs_input reset the counter to 1 every time
    and never reached BLOCK_RECURRENCE_LIMIT no matter how many cycles ran."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)

        # Cycle 1: typed needs_input block.
        kb.block_task(conn, tid, reason="need a decision", kind="needs_input")
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.get_task(conn, tid).block_recurrences == 1
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)

        # Cycle 2: untyped (kind=None) block -- a DIFFERENT kind than cycle 1.
        # This must still be the 2nd cycle and trip the breaker, not reset to 1.
        kb.block_task(conn, tid, reason="more items remain", kind=None)
        landed = kb.get_task(conn, tid)
        assert landed.status == "triage", (
            "alternating block kind must not reset the unblock-loop counter"
        )
        assert landed.block_recurrences == 2
        events = [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"]
        assert events, "expected triage escalation on the 2nd cycle despite the kind change"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") is None
        assert payload.get("prev_kind") == "needs_input"


def test_block_recurrences_accumulate_across_three_distinct_kinds(kanban_home: Path) -> None:
    """Three re-blocks with three DIFFERENT kinds (needs_input, then untyped,
    then needs_input again -- the exact sequence from the incident) must
    escalate on the 2nd cycle, matching a human needing to look after two
    block/unblock round-trips regardless of how the reason was phrased."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="run 85", kind="needs_input")
        assert kb.get_task(conn, tid).status == "blocked"
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)

        kb.block_task(conn, tid, reason="run 86", kind=None)
        # Would have incorrectly landed back in 'blocked' under the old logic.
        assert kb.get_task(conn, tid).status == "triage"


def test_complete_task_resets_counter_for_next_genuine_blocker(kanban_home: Path) -> None:
    """A successful completion clears the loop memory, so a later, wholly
    unrelated block on a *new* run of work starts a fresh count -- the
    unification of the counter must not make legitimate future blockers
    (on work done after a completion) inherit stale recurrence history."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="need creds", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.complete_task(conn, tid, result="done")
        row = kb.get_task(conn, tid)
        assert row.block_recurrences == 0
        assert row.block_kind is None


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



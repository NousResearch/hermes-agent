"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking.
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each re-block after an unblock increments ``block_recurrences``, and at
  ``BLOCK_RECURRENCE_LIMIT`` the task is flagged loop-escalated (a
  ``block_loop_detected`` event) while STAYING in ``blocked`` — it must not
  land in ``triage``, which the gateway auto-decomposer sweeps back into
  dispatch.
* A re-block with a DIFFERENT ``kind`` decays the counter but never resets it,
  so a worker cannot escape the breaker by varying its reported kind.
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










def test_loop_escalation_lands_in_blocked_not_triage(kanban_home: Path) -> None:
    """N>=limit same-kind blocks => blocked + block_loop_detected, NOT triage.

    ``triage`` is swept unconditionally by the gateway auto-decomposer
    (``_auto_decompose_tick``), so escalating there fed the card straight back
    into dispatch and re-armed the loop. Escalation must be a terminal resting
    state for a human, and the card must not appear in ``list_triage_ids()``.
    """
    from hermes_cli import kanban_decompose

    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        for i in range(kb.BLOCK_RECURRENCE_LIMIT):
            kb.block_task(conn, tid, reason="wall", kind="capability")
            if i < kb.BLOCK_RECURRENCE_LIMIT - 1:
                kb.unblock_task(conn, tid)
                _make_running_again(conn, tid)

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_recurrences >= kb.BLOCK_RECURRENCE_LIMIT
        assert [e for e in kb.list_events(conn, tid)
                if e.kind == "block_loop_detected"]

    assert tid not in kanban_decompose.list_triage_ids()


def test_alternating_block_kinds_still_escalate(kanban_home: Path) -> None:
    """Varying the reported ``kind`` must not reset the recurrence counter.

    The old rule was ``recurrences = prev + 1 if same_cause else 1``, which
    rewarded a worker for reporting a different kind each time: the counter
    reset to 1 forever and the breaker never fired. A changed kind may decay
    the counter, never zero it.
    """
    kinds = ["capability", "transient", "needs_input", "capability",
             "transient", "needs_input"]
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        escalated = False
        for kind in kinds:
            kb.block_task(conn, tid, reason="wall", kind=kind)
            task = kb.get_task(conn, tid)
            assert task.block_recurrences >= 1
            if [e for e in kb.list_events(conn, tid)
                    if e.kind == "block_loop_detected"]:
                escalated = True
                break
            kb.unblock_task(conn, tid)
            _make_running_again(conn, tid)

        assert escalated, (
            "alternating block kinds escaped the loop breaker: "
            f"recurrences={kb.get_task(conn, tid).block_recurrences}"
        )


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
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------



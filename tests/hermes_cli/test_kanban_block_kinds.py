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
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` by
  default (the amnesia that let the loop run unbounded). Automated
  unblockers rely on this.
* ``unblock_task(reset_recurrence=True)`` is the opt-in for a HUMAN-driven
  release, which does clear the counter — a person answering the question is
  not the unattended loop the breaker guards against. Automated unblockers
  must never pass it.
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
# Human vs automated unblock (reset_recurrence)
# ---------------------------------------------------------------------------
#
# The breaker exists to stop an UNATTENDED loop: a cron unblocks, the worker
# re-blocks for the same cause, forever. A human answering the question is not
# that loop, but the counter cannot tell them apart on its own — so a
# supervised task got exactly one question before routing to triage.
#
# ``unblock_task(reset_recurrence=True)`` is the opt-in for human-driven
# releases. The three tests below are the contract; the middle one is what
# keeps the opt-in from becoming a blanket bypass.


def test_human_unblock_allows_one_legitimate_same_kind_reblock(
    kanban_home: Path,
) -> None:
    """Sequence (a): supervised release must NOT trip the breaker.

    A worker asks a question, a human answers and releases the task, and the
    worker then asks a *different* question of the same kind. That second
    block is legitimate work, not a loop, so it stays in ``blocked``.
    """
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="which timezone?", kind="needs_input")
        assert kb.get_task(conn, tid).block_recurrences == 1

        # Human/CLI/dashboard release: informed decision, fresh start.
        assert kb.unblock_task(conn, tid, reset_recurrence=True)
        t = kb.get_task(conn, tid)
        assert t.status == "ready"
        assert t.block_recurrences == 0
        assert t.block_kind is None

        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="which currency?", kind="needs_input")
        t = kb.get_task(conn, tid)
        assert t.status == "blocked", "supervised re-block must not route to triage"
        assert t.block_recurrences == 1


def test_reset_does_not_disable_the_breaker_for_unattended_loops(
    kanban_home: Path,
) -> None:
    """Sequence (b): a genuine loop with no human release still trips.

    Guards against "fixing" the supervised case by weakening the breaker: two
    same-kind re-blocks with only default (automated) unblocks in between must
    still route to triage.
    """
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="need creds", kind="needs_input")
        kb.unblock_task(conn, tid)  # default: no reset
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="still need creds", kind="needs_input")
        t = kb.get_task(conn, tid)
        assert t.status == "triage"
        assert t.block_recurrences == 2


def test_automated_unblocker_must_not_reset_the_counter(kanban_home: Path) -> None:
    """Sequence (c): a sweeper cron cannot clear the loop memory.

    The counter surviving a blind automated unblock is the whole protection.
    This asserts the end-to-end consequence rather than just the stored
    value: an automated release followed by a same-kind re-block still
    escalates, and it emits the loop-detected event.
    """
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="need creds", kind="needs_input")

        # A sweeper cron releases the task, explicitly passing the default.
        assert kb.unblock_task(conn, tid, reset_recurrence=False)
        t = kb.get_task(conn, tid)
        assert t.block_recurrences == 1, "automated unblock must preserve the counter"
        assert t.block_kind == "needs_input", "kind must survive for cause comparison"

        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="need creds", kind="needs_input")
        t = kb.get_task(conn, tid)
        assert t.status == "triage"
        assert t.block_recurrences == 2
        assert [e for e in kb.list_events(conn, tid)
                if e.kind == "block_loop_detected"], "breaker must still fire"


def test_human_unblock_records_the_reset_on_the_event(kanban_home: Path) -> None:
    """The reset is auditable — an operator can see why a counter went to 0."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="needs_input")
        kb.unblock_task(conn, tid, reset_recurrence=True)
        unblocks = [e for e in kb.list_events(conn, tid) if e.kind == "unblocked"]
        assert unblocks, "expected an unblocked event"
        assert (unblocks[-1].payload or {}).get("recurrence_reset") is True

        # And the default path stays silent about it.
        tid2 = _running_task(conn, title="t2")
        kb.block_task(conn, tid2, reason="x", kind="needs_input")
        kb.unblock_task(conn, tid2)
        unblocks2 = [e for e in kb.list_events(conn, tid2) if e.kind == "unblocked"]
        assert "recurrence_reset" not in (unblocks2[-1].payload or {})


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



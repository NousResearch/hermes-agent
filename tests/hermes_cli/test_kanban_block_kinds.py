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


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Transient routing: a wait on time, not on a human
# ---------------------------------------------------------------------------


def _park_transient(conn, tid, reason="waiting on a background job"):
    assert kb.block_task(conn, tid, reason=reason, kind="transient")
    return kb.get_task(conn, tid)


def test_transient_parks_in_scheduled_not_blocked(kanban_home: Path) -> None:
    """``transient`` means "may clear on its own" — that is a wait on time,
    and ``scheduled`` is the column whose whole meaning is exactly that."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        task = _park_transient(conn, tid)
        assert task.status == "scheduled"
        assert task.block_kind == "transient"
        assert task.block_recurrences == 1
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert "transient_wait" in kinds
        assert "blocked" not in kinds, "a transient park is not a human block"


def test_transient_park_records_the_run_as_scheduled(kanban_home: Path) -> None:
    """Attempt history must say "waited", not "stopped for a human"."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        _park_transient(conn, tid)
        outcomes = [
            r["outcome"] for r in conn.execute(
                "SELECT outcome FROM task_runs WHERE task_id = ? ORDER BY id", (tid,),
            )
        ]
    assert outcomes[-1] == "scheduled"


def test_transient_never_reaches_triage(kanban_home: Path) -> None:
    """The regression this whole branch exists for: a machine wait must never
    land in ``triage``, the one column nothing promotes out of."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        seen = []
        for _ in range(kb.BLOCK_RECURRENCE_LIMIT + 2):
            seen.append(_park_transient(conn, tid).status)
            kb.unblock_task(conn, tid)
            _make_running_again(conn, tid)
    assert "triage" not in seen, f"transient escalated into triage: {seen}"
    assert seen[0] == "scheduled", "the first park waits on time"
    assert seen[1] == "blocked", (
        "believed twice and flaky twice — hand it to a human, who has a voice"
    )


def test_transient_escalation_emits_a_plain_block_not_a_loop_break(
    kanban_home: Path,
) -> None:
    """The escalated card must look like an ordinary human block, so every
    consumer that already watches ``blocked`` picks it up unchanged."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        _park_transient(conn, tid)
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        _park_transient(conn, tid)
        kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert "blocked" in kinds
    assert "block_loop_detected" not in kinds


def test_human_kinds_still_escalate_to_triage(kanban_home: Path) -> None:
    """Mutation guard: the transient branch must not swallow the loop breaker
    for the kinds it was built for."""
    for kind in ("needs_input", "capability", None):
        with kbc.connect_closing() as conn:
            tid = _running_task(conn, title=f"k-{kind}")
            kb.block_task(conn, tid, reason="x", kind=kind)
            assert kb.get_task(conn, tid).status == "blocked"
            kb.unblock_task(conn, tid)
            _make_running_again(conn, tid)
            kb.block_task(conn, tid, reason="x", kind=kind)
            assert kb.get_task(conn, tid).status == "triage", (
                f"{kind!r} must still escalate to triage"
            )


def test_scheduled_transient_comes_back_through_unblock(kanban_home: Path) -> None:
    """The park is recoverable by the ordinary door, with no special case."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        _park_transient(conn, tid)
        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"


def test_transient_park_honours_the_expected_run_guard(kanban_home: Path) -> None:
    """A worker that no longer owns the run must not be able to park the card
    out from under its successor."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        live_run = kb.get_task(conn, tid).current_run_id
        assert not kb.block_task(
            conn, tid, reason="stale", kind="transient",
            expected_run_id=(live_run or 0) + 99,
        )
        assert kb.get_task(conn, tid).status == "running"


def test_transient_park_is_not_confused_with_a_different_prior_kind(
    kanban_home: Path,
) -> None:
    """Recurrences count SAME-cause parks. A transient park after a human
    block starts the count over, so one flake does not inherit someone else's
    strike and skip straight to ``blocked``."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="need a human", kind="needs_input")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        task = _park_transient(conn, tid)
    assert task.status == "scheduled"
    assert task.block_recurrences == 1

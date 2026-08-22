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
# Dependency-block loop guard (backoff + escalation, 2026-08-19)
# ---------------------------------------------------------------------------


def test_dependency_block_without_task_link_does_not_repromote_immediately(
    kanban_home: Path,
) -> None:
    """A dependency block with NO formal task_link must not be re-promoted
    to ``ready`` on the very next ``recompute_ready`` tick (the loop
    observed on t_ee5d85a4: 11 runs in 11 minutes, all outcome=blocked).

    Before the fix, ``recompute_ready``'s "all parents done" check is
    vacuously true over an empty parent set, so a task blocked purely by a
    prose-only dependency (no ``kanban_link``) was re-promoted immediately.
    """
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="waiting on X, no link", kind="dependency")
        assert kb.get_task(conn, tid).status == "todo"
        task = kb.get_task(conn, tid)
        assert task.dependency_block_streak == 1
        assert task.dependency_backoff_until is not None
        # Immediate recompute_ready must NOT promote — backoff window active.
        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert kb.get_task(conn, tid).status == "todo"


def test_dependency_block_backoff_expires_and_promotes(kanban_home: Path) -> None:
    """Once the backoff window has elapsed, the task IS eligible again."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="waiting on X, no link", kind="dependency")
        # Force the backoff window into the past instead of sleeping.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET dependency_backoff_until = ? WHERE id = ?",
                (0, tid),
            )
        promoted = kb.recompute_ready(conn)
        assert promoted == 1
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        # Real progress clears the loop-guard streak.
        assert task.dependency_block_streak == 0
        assert task.dependency_backoff_until is None


def test_dependency_block_with_real_link_never_backs_off(kanban_home: Path) -> None:
    """A task with a genuine task_link is correctly parent-gated already —
    it must never accumulate a dependency-loop streak or backoff, even
    across repeated (re-)blocks while the real parent is still working."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        for _ in range(3):
            kb.block_task(conn, child, reason="still waiting", kind="dependency")
            task = kb.get_task(conn, child)
            assert task.status == "todo"
            assert task.dependency_block_streak == 0
            assert task.dependency_backoff_until is None
            # Parent still not done -> recompute_ready must not promote.
            assert kb.recompute_ready(conn) == 0
            # claim_task correctly refuses ready->running while the real
            # parent is undone, so drive the child straight back to
            # running (bypassing claim_task's parent gate) to re-block it
            # and prove the streak/backoff never engage across iterations.
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status='running' WHERE id=?", (child,)
                )


def test_dependency_block_escalates_to_blocked_after_limit(kanban_home: Path) -> None:
    """N consecutive same-cause dependency blocks with no formal link and no
    intervening real promotion must stop returning to ``todo`` (where the
    dispatcher would just re-promote and re-block it again) and instead
    land in the human-visible ``blocked`` bucket."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        for i in range(kb.DEPENDENCY_BLOCK_ESCALATION_LIMIT):
            kb.block_task(conn, tid, reason="phantom dependency", kind="dependency")
            task = kb.get_task(conn, tid)
            if i < kb.DEPENDENCY_BLOCK_ESCALATION_LIMIT - 1:
                assert task.status == "todo", f"iteration {i}"
                # Clear backoff so the next iteration's block_task call can
                # re-claim/re-run without waiting on the exponential window.
                with kb.write_txn(conn):
                    conn.execute(
                        "UPDATE tasks SET dependency_backoff_until = NULL "
                        "WHERE id = ?",
                        (tid,),
                    )
                _make_running_again(conn, tid)
            else:
                assert task.status == "blocked", f"iteration {i}"
        events = [
            e for e in kb.list_events(conn, tid)
            if e.kind == "dependency_loop_detected"
        ]
        assert events, "expected a dependency_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("streak") == kb.DEPENDENCY_BLOCK_ESCALATION_LIMIT
        # Escalated dependency blocks must NOT be silently auto-recovered by
        # recompute_ready (no sticky-block author, but the escalation is a
        # deliberate human handoff) — they still sit in 'blocked' until a
        # human/orchestrator adds the missing link or re-routes the task.
        # (recompute_ready only promotes when parents are actually done;
        # this task still has zero task_links so it stays put.)
        assert kb.get_task(conn, tid).status == "blocked"


def test_dependency_block_streak_resets_on_completion(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="dependency")
        assert kb.get_task(conn, tid).dependency_block_streak == 1
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        kb.claim_task(conn, tid, claimer="worker")
        kb.complete_task(conn, tid, result="done")
        task = kb.get_task(conn, tid)
        assert task.dependency_block_streak == 0
        assert task.dependency_backoff_until is None


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------



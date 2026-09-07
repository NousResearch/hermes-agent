"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks with unfinished linked parents wait in ``todo``;
  missing or completed dependency links remain blocked to prevent retry churn.
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


def test_machine_pr_feedback_block_returns_to_dispatcher(kanban_home: Path) -> None:
    """Recoverable PR/CI worker failures never become human-sticky blocks."""

    import hermes_cli.kanban_db_connect as _hermes_cli_kanban_db_connect
    with _hermes_cli_kanban_db_connect.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="GitHub PR feedback: acme/widgets#17",
            assignee="pr-repair-steward",
            idempotency_key="github-pr-feedback:abc123",
            initial_status="running",
        )
        assert kb.claim_task(conn, tid, claimer="pr-repair-steward") is not None
        kb.block_task(conn, tid, reason="start validation", kind="needs_input")
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        events = [event.kind for event in kb.list_events(conn, tid)]
        assert "machine_handoff" in events


def test_db_accepts_synthetic_assignee_for_internal_worker(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (kanban_home / "profiles").mkdir()
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Review worker", assignee="worker")
        assert task_id.startswith("t_")


def test_intent_review_needs_input_triage_stays_human_gated(kanban_home: Path) -> None:
    import hermes_cli.kanban_db_connect as _hermes_cli_kanban_db_connect
    with _hermes_cli_kanban_db_connect.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="Operator intent required: PR #17",
            body="Record only the operator intent decision.",
            assignee="task-orchestrator",
            idempotency_key="github-pr-feedback:intent-review:repo:17:abc",
            initial_status="running",
        )
        assert kb.claim_task(conn, tid, claimer="task-orchestrator") is not None
        kb.block_task(conn, tid, reason="operator intent", kind="needs_input")
        kb.unblock_task(conn, tid)
        assert kb.claim_task(conn, tid, claimer="task-orchestrator") is not None
        kb.block_task(conn, tid, reason="operator intent", kind="needs_input")
        assert kb.get_task(conn, tid).status == "triage"

        promoted = kb.recompute_ready(conn)

        assert promoted == 0
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "triage"


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("finished_parent", [False, True])
def test_dependency_without_pending_parent_stays_blocked(kanban_home, finished_parent):
    with kbc.connect_closing() as conn:
        child = _running_task(conn, title="missing external prerequisite")
        if finished_parent:
            parent = _running_task(conn, title="finished prerequisite")
            kb.complete_task(conn, parent, result="done")
            kb.link_tasks(conn, parent_id=parent, child_id=child)
        assert kb.block_task(conn, child, reason="source unavailable", kind="dependency")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "blocked"
        assert kb.claim_task(conn, child, claimer="worker") is None


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

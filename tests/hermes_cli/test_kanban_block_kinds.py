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


def test_first_typed_block_lands_in_blocked(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="which key?", kind="needs_input")
        t = kb.get_task(conn, tid)
        assert t.status == "blocked"
        assert t.block_kind == "needs_input"
        assert t.block_recurrences == 1


def test_unblock_does_not_reset_recurrence_counter(kanban_home: Path) -> None:
    """The crux of the fix: unblock must preserve the loop counter."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="needs_input")
        assert kb.get_task(conn, tid).block_recurrences == 1
        assert kb.unblock_task(conn, tid)
        t = kb.get_task(conn, tid)
        assert t.status == "ready"
        assert t.block_recurrences == 1  # NOT reset to 0
        assert t.block_kind == "needs_input"  # kind preserved for comparison


def test_same_cause_reblock_routes_to_triage(kanban_home: Path) -> None:
    """Dale's loop: block → unblock → re-block same kind → triage."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="need creds", kind="needs_input")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="still need creds", kind="needs_input")
        t = kb.get_task(conn, tid)
        assert t.status == "triage"
        assert t.block_recurrences == 2


def test_untyped_block_loop_also_protected(kanban_home: Path) -> None:
    """Legacy un-typed blocks (kind=None) still trip the breaker."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="a")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="a again")
        assert kb.get_task(conn, tid).status == "triage"


def test_different_kinds_do_not_compound(kanban_home: Path) -> None:
    """A re-block for a DIFFERENT reason resets the counter to 1."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="a", kind="needs_input")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="b", kind="capability")
        t = kb.get_task(conn, tid)
        assert t.status == "blocked"
        assert t.block_recurrences == 1


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


def test_dependency_block_routes_to_todo(kanban_home: Path) -> None:
    """Dependency waits never enter the human 'blocked' bucket."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="need X first", kind="dependency")
        t = kb.get_task(conn, tid)
        assert t.status == "todo"
        assert t.block_kind == "dependency"


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


def test_completion_clears_block_memory(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).block_recurrences == 1
        kb.complete_task(conn, tid, result="done")
        t = kb.get_task(conn, tid)
        assert t.status == "done"
        assert t.block_recurrences == 0
        assert t.block_kind is None


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------


def test_invalid_kind_rejected(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        with pytest.raises(ValueError):
            kb.block_task(conn, tid, reason="x", kind="bogus")


def test_review_required_block_rejected_without_task_state_mutation(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        before = kb.get_task(conn, tid)
        assert before is not None

        with pytest.raises(kb.ReviewHandoffBlockError) as exc:
            kb.block_task(
                conn,
                tid,
                reason="review-required: implementation complete",
                kind="needs_input",
                expected_run_id=before.current_run_id,
            )

        after = kb.get_task(conn, tid)
        assert after is not None
        assert exc.value.task_id == tid
        assert after.status == before.status == "running"
        assert after.current_run_id == before.current_run_id
        assert after.block_kind == before.block_kind
        assert after.block_recurrences == before.block_recurrences


def test_review_kind_alias_rejected_before_invalid_kind_error(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)

        with pytest.raises(kb.ReviewHandoffBlockError):
            kb.block_task(conn, tid, reason="done", kind="review_required")

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running"


def test_producer_handoff_successor_is_idempotent_and_not_parent_suppressed(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        producer = _running_task(conn, title="producer")
        key = "review:artifact:sha256:abc123"

        successor_1 = kb.create_task(
            conn,
            title="EVE review exact artifact abc123",
            assignee="eve",
            body="Review immutable artifact sha256:abc123 from producer.",
            idempotency_key=key,
        )
        successor_2 = kb.create_task(
            conn,
            title="duplicate EVE review exact artifact abc123",
            assignee="eve",
            idempotency_key=key,
        )
        assert successor_2 == successor_1

        successor = kb.get_task(conn, successor_1)
        assert successor is not None
        assert successor.status == "ready"

        assert kb.complete_task(
            conn,
            producer,
            summary="Producer complete; exact artifact successor queued.",
            metadata={
                "artifact_sha256": "abc123",
                "successor_task_id": successor_1,
                "successor_status": successor.status,
                "dispatch_receipt": {"status": "queued_capacity"},
            },
        )

        producer_task = kb.get_task(conn, producer)
        successor_task = kb.get_task(conn, successor_1)
        assert producer_task is not None
        assert successor_task is not None
        assert producer_task.status == "done"
        assert successor_task.status == "ready"


def test_block_without_kind_is_backward_compatible(kanban_home: Path) -> None:
    """Existing callers that pass no kind keep the old single-block behaviour."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        assert kb.block_task(conn, tid, reason="legacy")
        t = kb.get_task(conn, tid)
        assert t.status == "blocked"
        assert t.block_kind is None

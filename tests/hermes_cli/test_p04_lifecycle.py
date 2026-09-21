"""P04 lifecycle proof — synthetic traversal through the real kanban engine.

Goes beyond ready/claim/complete + block/unblock. Exercises a realistic
board traversal end-to-end against a disposable SQLite fixture:

  backlog/triage
    -> decompose_triage_task (a real sibling dependency graph: A then B)
    -> claim/work child A
    -> complete_task routes running -> review (full-tier WS-4 gate)
    -> claim_review_task + approve_review_task(terminal) -> done
    -> parent completion promotes child B ready -> claim/work B -> done
    -> root (orchestrator) wakes ready

Plus a forced review/quality failure that leaves recoverable state:
  complete -> review -> claim_review -> reject_review_task -> blocked
    -> unblock_task -> ready (recoverable), re-claim, re-complete,
    re-review, approve -> done.

Only edits kanban_decompose.py / kanban_specify.py if a failing test
proves a source defect. No source edits were needed here — the engine
already implements the contract.

Isolation: temp HERMES_HOME + explicit db_path; live ~/.hermes never
opened. No subprocess, no cron, no activation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb

# A full-tier body that satisfies validate_task_contract (WS-1 gate):
# both ## Acceptance Criteria and ## Test Plan sections present.
FULL_TIER_BODY = """Implement the feature.

## Acceptance Criteria
- behaviour matches spec
- tests green

## Test Plan
- pytest the focused suite
"""


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    return db_path


def _status(conn, tid):
    return kb.get_task(conn, tid).status


def test_full_synthetic_traversal_decompose_review_accept_done(board):
    """backlog/triage -> decompose (A->B graph) -> claim/work/review/
    accept A -> parent-done promotes B -> claim/work/review/accept B
    -> root wakes ready."""
    conn = kb.connect(db_path=board)
    try:
        # 1. triage root (orchestrator-owned, full-tier so review gate fires)
        root = kb.create_task(
            conn, title="triage-root", assignee="orchestrator",
            tier="full", triage=True,
        )
        assert _status(conn, root) == "triage"

        # 2. decompose into a dependency graph: child A (no parents) then
        #    child B (parent=A). A real sibling dependency graph.
        children = [
            {"title": "child-A", "body": FULL_TIER_BODY,
             "assignee": "worker", "parents": []},
            {"title": "child-B", "body": FULL_TIER_BODY,
             "assignee": "worker", "parents": [0]},
        ]
        cids = kb.decompose_triage_task(
            conn, root, root_assignee="orchestrator",
            children=children, author="test",
        )
        assert cids is not None and len(cids) == 2
        a, b = cids
        # Root flipped triage -> todo and waits for children.
        assert _status(conn, root) == "todo"
        # A has no parents -> ready; B has parent A -> todo.
        assert _status(conn, a) == "ready"
        assert _status(conn, b) == "todo"

        # 3. claim + work A
        claimed_a = kb.claim_task(conn, a)
        assert claimed_a is not None and claimed_a.status == "running"

        # 4. complete_task routes running -> review (full-tier WS-4 gate)
        ok = kb.complete_task(conn, a, summary="work done", result="ok")
        assert ok is True
        assert _status(conn, a) == "review"

        # 5. quality gate: claim review + approve (terminal). Approver must
        #    differ from the original worker.
        rev = kb.claim_review_task(conn, a, claimer="reviewer-1")
        assert rev is not None and rev.status == "running"
        appr = kb.approve_review_task(
            conn, a, outcome="terminal",
            approver_profile="reviewer-1", summary="approved",
        )
        assert appr["ok"] is True
        assert appr["status"] == "done"
        assert _status(conn, a) == "done"

        # 6. parent completion promotes B todo -> ready
        assert _status(conn, b) == "ready"

        # 7. claim + work + review + accept B
        claimed_b = kb.claim_task(conn, b)
        assert claimed_b is not None and claimed_b.status == "running"
        kb.complete_task(conn, b, summary="work done", result="ok")
        assert _status(conn, b) == "review"
        kb.claim_review_task(conn, b, claimer="reviewer-2")
        appr_b = kb.approve_review_task(
            conn, b, outcome="terminal",
            approver_profile="reviewer-2", summary="approved",
        )
        assert appr_b["ok"] is True
        assert _status(conn, b) == "done"

        # 8. all children done -> root wakes ready (orchestrator judges)
        assert _status(conn, root) == "ready"
    finally:
        conn.close()


def test_forced_review_failure_leaves_recoverable_state(board):
    """A rejected review blocks the task in a recoverable state: unblock
    re-queues it, and the full claim/work/review/accept cycle runs again
    to completion."""
    conn = kb.connect(db_path=board)
    try:
        tid = kb.create_task(
            conn, title="full-task", assignee="worker",
            tier="full", body=FULL_TIER_BODY,
        )
        assert _status(conn, tid) == "ready"

        # claim + complete -> review
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="first attempt", result="ok")
        assert _status(conn, tid) == "review"

        # reviewer rejects -> blocked (recoverable)
        kb.claim_review_task(conn, tid, claimer="reviewer-1")
        ok = kb.reject_review_task(
            conn, tid, findings={"reasons": ["found bugs"]},
            rejected_by_profile="reviewer-1",
        )
        assert ok is True
        assert _status(conn, tid) == "blocked"

        # recover: unblock -> ready
        ub = kb.unblock_task(conn, tid)
        assert ub is True
        assert _status(conn, tid) == "ready"

        # re-run the full quality gate to completion
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="fixed", result="ok")
        assert _status(conn, tid) == "review"
        kb.claim_review_task(conn, tid, claimer="reviewer-2")
        appr = kb.approve_review_task(
            conn, tid, outcome="terminal",
            approver_profile="reviewer-2", summary="approved",
        )
        assert appr["ok"] is True
        assert _status(conn, tid) == "done"
    finally:
        conn.close()


def test_decompose_graph_cycle_rejected(board):
    """decompose_triage_task must reject a cyclic sibling graph rather
    than deadlock children in todo."""
    conn = kb.connect(db_path=board)
    try:
        root = kb.create_task(
            conn, title="cyclic-root", assignee="orchestrator", triage=True,
        )
        cyclic = [
            {"title": "x", "assignee": "worker", "parents": [1]},  # x <- y
            {"title": "y", "assignee": "worker", "parents": [0]},  # y <- x
        ]
        with pytest.raises(ValueError, match="cyclic"):
            kb.decompose_triage_task(
                conn, root, root_assignee="orchestrator",
                children=cyclic, author="test",
            )
        # Root stays in triage (decompose was rejected atomically).
        assert _status(conn, root) == "triage"
    finally:
        conn.close()


def test_claim_is_atomic_no_double_claim(board):
    """An already-running task cannot be claimed again."""
    conn = kb.connect(db_path=board)
    try:
        tid = kb.create_task(conn, title="atomic-claim", assignee="worker",
                             tier="full", body=FULL_TIER_BODY)
        first = kb.claim_task(conn, tid)
        assert first is not None
        second = kb.claim_task(conn, tid)
        assert second is None
    finally:
        conn.close()


def test_complete_nonexistent_returns_false(board):
    conn = kb.connect(db_path=board)
    try:
        assert kb.complete_task(conn, "t_doesnotexist", summary="x") is False
    finally:
        conn.close()


def test_rollback_disposes_without_touching_production(board, tmp_path):
    """After dropping the disposable DB, a fresh board in a different path
    initialises cleanly with zero tasks — no leakage."""
    conn = kb.connect(db_path=board)
    try:
        tid = kb.create_task(conn, title="rollback-target", assignee="worker")
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    board.unlink()
    assert not board.exists()

    other = tmp_path / "other-board.db"
    kb.init_db(db_path=other)
    conn2 = kb.connect(db_path=other)
    try:
        rows = conn2.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert rows == 0
    finally:
        conn2.close()

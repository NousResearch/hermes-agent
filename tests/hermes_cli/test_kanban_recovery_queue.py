"""Tests for the recovery-queue path in ``recover_blocked_tasks``.

Covers:
1. **Review-marker eligibility** — a blocked task whose latest comment
   contains a review-fix marker (``review-required``, ``needs_fix``,
   ``verdict=needs_fix``) produces a ready successor.
2. **Transient eligibility** — ``block_kind='transient'`` qualifies
   regardless of comment content.
3. **Safety skips** — tasks with ``HUMAN_STOP`` or ``needs_input`` in
   any comment (case-insensitive), or ``block_kind`` in
   ``{'needs_input', 'capability'}``, are unconditionally skipped.
4. **Idempotent second call** — calling ``recover_blocked_tasks`` twice
   for the same source+comment must not create a duplicate successor
   or emit duplicate audit events.
5. **Per-tick cap** — ``max_per_tick`` limits how many successors are
   created in a single pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _blocked_task(
    conn,
    title: str = "task",
    assignee: str = "worker",
    block_kind: str | None = None,
    comment: str | None = None,
) -> str:
    """Create a task, drive it to blocked, optionally add a comment."""
    tid = kb.create_task(conn, title=title, assignee=assignee)
    # Drive to running so block_task can act.
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer=assignee)
    assert claimed is not None
    kb.block_task(conn, tid, reason="test-block", kind=block_kind)
    if comment:
        kb.add_comment(conn, tid, author="reviewer", body=comment)
    return tid


# ---------------------------------------------------------------------------
# Review-marker eligibility
# ---------------------------------------------------------------------------

def test_review_marker_creates_successor(kanban_home: Path) -> None:
    """A blocked task with a review-fix marker in latest comment is recovered."""
    with kb.connect_closing() as conn:
        tid = _blocked_task(conn, title="fix auth", comment="verdict=needs_fix")

        result = kb.recover_blocked_tasks(conn)

        assert len(result.recovered) == 1
        source_id, successor_id = result.recovered[0]
        assert source_id == tid
        successor = kb.get_task(conn, successor_id)
        assert successor is not None
        assert successor.status == "ready"
        assert successor.title == "Recovery: fix auth"
        assert "recovery-queue" in (successor.created_by or "")


def test_review_marker_case_insensitive(kanban_home: Path) -> None:
    """Review markers are matched case-insensitively."""
    with kb.connect_closing() as conn:
        tid = _blocked_task(conn, title="t", comment="REVIEW-REQUIRED please")
        result = kb.recover_blocked_tasks(conn)
        assert len(result.recovered) == 1
        assert result.recovered[0][0] == tid


def test_needs_fix_marker(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _blocked_task(conn, title="t", comment="needs_fix: style issues")
        result = kb.recover_blocked_tasks(conn)
        assert len(result.recovered) == 1
        assert result.recovered[0][0] == tid


# ---------------------------------------------------------------------------
# Transient eligibility
# ---------------------------------------------------------------------------

def test_transient_block_kind_creates_successor(kanban_home: Path) -> None:
    """block_kind='transient' qualifies regardless of comment content."""
    with kb.connect_closing() as conn:
        tid = _blocked_task(
            conn, title="flaky api", block_kind="transient",
            comment="some random comment",
        )
        result = kb.recover_blocked_tasks(conn)
        assert len(result.recovered) == 1
        assert result.recovered[0][0] == tid


def test_transient_without_comment(kanban_home: Path) -> None:
    """Transient block without any comment still qualifies."""
    with kb.connect_closing() as conn:
        tid = _blocked_task(conn, title="retry me", block_kind="transient")
        result = kb.recover_blocked_tasks(conn)
        assert len(result.recovered) == 1


# ---------------------------------------------------------------------------
# Safety skips
# ---------------------------------------------------------------------------

def test_safety_skip_human_stop_comment(kanban_home: Path) -> None:
    """A comment containing HUMAN_STOP (any case) prevents recovery."""
    with kb.connect_closing() as conn:
        tid = _blocked_task(
            conn, title="dangerous", block_kind="transient",
            comment="HUMAN_STOP: do not auto-recover",
        )
        result = kb.recover_blocked_tasks(conn)
        assert len(result.recovered) == 0
        assert tid in result.skipped_safety


def test_safety_skip_needs_input_comment(kanban_home: Path) -> None:
    """A comment containing needs_input (any case) prevents recovery."""
    with kb.connect_closing() as conn:
        tid = _blocked_task(
            conn, title="waiting", block_kind="transient",
            comment="Needs_Input from user",
        )
        result = kb.recover_blocked_tasks(conn)
        assert len(result.recovered) == 0
        assert tid in result.skipped_safety


def test_safety_skip_older_comment_blocks(kanban_home: Path) -> None:
    """Safety markers in ANY comment (not just latest) prevent recovery."""
    with kb.connect_closing() as conn:
        tid = _blocked_task(
            conn, title="multi-comment", block_kind="transient",
        )
        # First comment has the safety marker.
        kb.add_comment(conn, tid, author="human", body="human_stop")
        # Second (latest) comment does NOT have it.
        kb.add_comment(conn, tid, author="bot", body="review-required")

        result = kb.recover_blocked_tasks(conn)
        assert len(result.recovered) == 0
        assert tid in result.skipped_safety


def test_safety_skip_block_kind_needs_input(kanban_home: Path) -> None:
    """block_kind='needs_input' is unconditionally skipped."""
    with kb.connect_closing() as conn:
        tid = _blocked_task(
            conn, title="input needed", block_kind="needs_input",
            comment="review-required",
        )
        result = kb.recover_blocked_tasks(conn)
        assert len(result.recovered) == 0
        assert tid in result.skipped_safety


def test_safety_skip_block_kind_capability(kanban_home: Path) -> None:
    """block_kind='capability' is unconditionally skipped."""
    with kb.connect_closing() as conn:
        tid = _blocked_task(
            conn, title="missing tool", block_kind="capability",
            comment="review-required",
        )
        result = kb.recover_blocked_tasks(conn)
        assert len(result.recovered) == 0
        assert tid in result.skipped_safety


# ---------------------------------------------------------------------------
# Idempotent second call
# ---------------------------------------------------------------------------

def test_idempotent_second_call(kanban_home: Path) -> None:
    """Calling recover_blocked_tasks twice does not create duplicates."""
    with kb.connect_closing() as conn:
        tid = _blocked_task(
            conn, title="once only", block_kind="transient",
        )

        r1 = kb.recover_blocked_tasks(conn)
        assert len(r1.recovered) == 1
        successor_id = r1.recovered[0][1]

        r2 = kb.recover_blocked_tasks(conn)
        assert len(r2.recovered) == 0
        assert tid in r2.skipped_idempotent

        # Verify only one successor exists.
        successor = kb.get_task(conn, successor_id)
        assert successor is not None

        # Verify audit events were only emitted once.
        events = kb.list_events(conn, tid)
        dispatched = [e for e in events if e.kind == "recovery_dispatched"]
        assert len(dispatched) == 1


def test_idempotent_no_duplicate_audit_events(kanban_home: Path) -> None:
    """The successor's recovery_created event must not be duplicated."""
    with kb.connect_closing() as conn:
        tid = _blocked_task(conn, title="audit check", block_kind="transient")

        r1 = kb.recover_blocked_tasks(conn)
        successor_id = r1.recovered[0][1]

        # Second call — idempotent skip.
        kb.recover_blocked_tasks(conn)

        events = kb.list_events(conn, successor_id)
        created_events = [e for e in events if e.kind == "recovery_created"]
        assert len(created_events) == 1


def test_new_comment_gets_new_successor(kanban_home: Path) -> None:
    """A later review comment has a distinct idempotency key and is recoverable."""
    with kb.connect_closing() as conn:
        tid = _blocked_task(conn, title="iterate", comment="review-required: first")
        first = kb.recover_blocked_tasks(conn)
        assert len(first.recovered) == 1

        kb.add_comment(conn, tid, author="reviewer", body="needs_fix: second")
        second = kb.recover_blocked_tasks(conn)
        assert len(second.recovered) == 1
        assert second.recovered[0][0] == tid
        assert second.recovered[0][1] != first.recovered[0][1]


def test_same_finding_text_with_new_comment_is_idempotent(kanban_home: Path) -> None:
    """Finding identity is content-based, not tied to a new comment row id."""
    with kb.connect_closing() as conn:
        tid = _blocked_task(
            conn, title="same finding", comment="review-required: Missing guard",
        )
        first = kb.recover_blocked_tasks(conn)
        assert len(first.recovered) == 1

        kb.add_comment(
            conn, tid, author="reviewer",
            body="  REVIEW-REQUIRED:   missing   guard  ",
        )
        second = kb.recover_blocked_tasks(conn)

        assert second.recovered == []
        assert second.skipped_idempotent == [tid]
        successors = conn.execute(
            "SELECT id FROM tasks WHERE created_by = 'recovery-queue'"
        ).fetchall()
        assert len(successors) == 1


def test_dependency_review_remediation_cycle(kanban_home: Path) -> None:
    """Reviewer waits for one fixer task, then wakes exactly once after it completes."""
    with kb.connect_closing() as conn:
        review_id = kb.create_task(
            conn,
            title="Review change",
            assignee="reviewer",
            classification="review",
        )
        assert kb.claim_task(conn, review_id, claimer="reviewer") is not None
        assert kb.block_task(
            conn,
            review_id,
            reason="finding: missing null guard",
            kind="dependency",
        )
        assert kb.get_task(conn, review_id).status == "blocked"

        first = kb.recover_blocked_tasks(conn, fixer_assignee="implementer")
        assert len(first.recovered) == 1
        source_id, remediation_id = first.recovered[0]
        assert source_id == review_id
        remediation = kb.get_task(conn, remediation_id)
        assert remediation is not None
        assert remediation.assignee == "implementer"
        assert remediation.classification == "remediation"
        assert remediation.status == "ready"
        assert kb.get_task(conn, review_id).status == "todo"
        assert conn.execute(
            "SELECT 1 FROM task_links WHERE parent_id = ? AND child_id = ?",
            (remediation_id, review_id),
        ).fetchone() is not None

        # Further dispatcher/recovery passes cannot wake the review or fan out
        # another remediation while the fixer has not completed.
        assert kb.recompute_ready(conn) == 0
        second = kb.recover_blocked_tasks(conn, fixer_assignee="implementer")
        assert second.recovered == []
        assert kb.get_task(conn, review_id).status == "todo"

        assert kb.claim_task(conn, remediation_id, claimer="implementer") is not None
        assert kb.complete_task(conn, remediation_id, result="fixed")
        assert kb.get_task(conn, review_id).status == "ready"

        # Promotion is edge-triggered: another recompute has nothing to do.
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, review_id).status == "ready"


def test_dispatch_routes_dependency_when_generic_recovery_is_disabled(
    kanban_home: Path,
) -> None:
    """Dependency remediation is autonomous, not gated by generic recovery."""
    with kb.connect_closing() as conn:
        review_id = kb.create_task(
            conn, title="Review", assignee="reviewer", classification="review",
        )
        assert kb.claim_task(conn, review_id, claimer="reviewer") is not None
        assert kb.block_task(
            conn, review_id, reason="finding: stale output", kind="dependency",
        )

        result = kb.dispatch_once(
            conn,
            max_spawn=0,
            recovery_queue_enabled=False,
            recovery_fixer_assignee="configured-fixer",
        )

        assert result.recovery is not None
        assert len(result.recovery.recovered) == 1
        _, remediation_id = result.recovery.recovered[0]
        assert kb.get_task(conn, remediation_id).assignee == "configured-fixer"
        assert kb.get_task(conn, review_id).status == "todo"


# ---------------------------------------------------------------------------
# Per-tick cap
# ---------------------------------------------------------------------------

def test_max_per_tick_cap(kanban_home: Path) -> None:
    """max_per_tick limits recoveries in a single pass."""
    with kb.connect_closing() as conn:
        tids = []
        for i in range(5):
            tid = _blocked_task(
                conn, title=f"task-{i}", block_kind="transient",
            )
            tids.append(tid)

        result = kb.recover_blocked_tasks(conn, max_per_tick=2)
        assert len(result.recovered) == 2

        # A second pass picks up more.
        r2 = kb.recover_blocked_tasks(conn, max_per_tick=2)
        assert len(r2.recovered) == 2

        # Third pass gets the last one.
        r3 = kb.recover_blocked_tasks(conn, max_per_tick=2)
        assert len(r3.recovered) == 1


# ---------------------------------------------------------------------------
# Successor task properties
# ---------------------------------------------------------------------------

def test_successor_inherits_assignee(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _blocked_task(
            conn, title="inherit", assignee="alice",
            block_kind="transient",
        )
        result = kb.recover_blocked_tasks(conn)
        successor = kb.get_task(conn, result.recovered[0][1])
        assert successor is not None
        assert successor.assignee == "alice"


def test_successor_body_contains_source_id(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _blocked_task(conn, title="ref", block_kind="transient")
        result = kb.recover_blocked_tasks(conn)
        successor = kb.get_task(conn, result.recovered[0][1])
        assert successor is not None
        assert tid in (successor.body or "")


def test_ineligible_no_marker_no_transient(kanban_home: Path) -> None:
    """A blocked task without transient kind or review marker is not recovered."""
    with kb.connect_closing() as conn:
        _blocked_task(conn, title="nope", comment="just a normal comment")
        result = kb.recover_blocked_tasks(conn)
        assert len(result.recovered) == 0

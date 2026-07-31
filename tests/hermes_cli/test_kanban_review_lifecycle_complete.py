"""End-to-end regressions for the Kanban review lifecycle.

These tests cover the two review models that must coexist:

* first-class same-card review, including an autonomous reviewer requesting
  changes and routing the task back to the original implementer; and
* legacy downstream review cards, where a sticky ``review-required`` parent
  can silently starve its reviewer child and therefore needs an immediate,
  graph-aware diagnostic.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_diagnostics as kd


@pytest.fixture
def conn(tmp_path: Path):
    db = kb.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _event(events, kind: str):
    return [event for event in events if event.kind == kind][-1]


def _run(runs, outcome: str):
    return [run for run in runs if run.outcome == outcome][-1]


def _claimed_review(
    conn,
    title: str,
    *,
    ttl_seconds: int | None = None,
    max_runtime_seconds: int | None = None,
):
    task_id = kb.create_task(
        conn,
        title=title,
        assignee="builder",
        max_runtime_seconds=max_runtime_seconds,
    )
    implementation = kb.claim_task(conn, task_id, claimer="builder:test")
    assert implementation is not None
    assert kb.request_review(
        conn,
        task_id,
        summary="ready for independent review",
        reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(
        conn,
        task_id,
        ttl_seconds=ttl_seconds,
    )
    assert review is not None
    return task_id, review


def test_same_card_review_supports_changes_and_approval_without_block_loop(conn):
    task_id = kb.create_task(conn, title="Implement guarded export", assignee="builder")
    implementation = kb.claim_task(conn, task_id, claimer="builder:1")
    assert implementation is not None

    assert kb.request_review(
        conn,
        task_id,
        reviewer="reviewer",
        summary="Implementation and focused tests are ready.",
        metadata={"commit": "abc123"},
        expected_run_id=implementation.current_run_id,
    )

    awaiting_review = kb.get_task(conn, task_id)
    assert awaiting_review is not None
    assert awaiting_review.status == "review"
    assert awaiting_review.assignee == "reviewer"
    assert awaiting_review.current_run_id is None

    first_events = kb.list_events(conn, task_id)
    requested = _event(first_events, "review_requested")
    assert requested.payload["implementer"] == "builder"
    assert requested.payload["reviewer"] == "reviewer"
    assert requested.payload["summary"] == "Implementation and focused tests are ready."
    implementation_run = _run(kb.list_runs(conn, task_id), "review_requested")
    assert implementation_run.summary == "Implementation and focused tests are ready."
    assert implementation_run.metadata == {"commit": "abc123"}

    review = kb.claim_review_task(conn, task_id, claimer="reviewer:1")
    assert review is not None
    assert kb.request_changes(
        conn,
        task_id,
        reason="Add a regression for the fallback branch.",
        expected_run_id=review.current_run_id,
    ) == (True, "builder")

    rework = kb.get_task(conn, task_id)
    assert rework is not None
    assert rework.status == "ready"
    assert rework.assignee == "builder"
    assert rework.current_run_id is None
    changes = _event(kb.list_events(conn, task_id), "changes_requested")
    assert changes.payload["reason"] == "Add a regression for the fallback branch."
    assert changes.payload["implementer"] == "builder"
    _run(kb.list_runs(conn, task_id), "changes_requested")

    implementation_2 = kb.claim_task(conn, task_id, claimer="builder:2")
    assert implementation_2 is not None
    assert kb.request_review(
        conn,
        task_id,
        reviewer="reviewer",
        summary="Fallback regression added.",
        expected_run_id=implementation_2.current_run_id,
    )
    review_2 = kb.claim_review_task(conn, task_id, claimer="reviewer:2")
    assert review_2 is not None
    assert kb.complete_task(
        conn,
        task_id,
        summary="Approved after independent verification.",
        expected_run_id=review_2.current_run_id,
    )

    completed = kb.get_task(conn, task_id)
    assert completed is not None
    assert completed.status == "done"
    assert completed.block_recurrences == 0


def test_review_changes_reapply_parent_gate(conn):
    parent_id = kb.create_task(conn, title="Upstream prerequisite", assignee="planner")
    task_id = kb.create_task(
        conn,
        title="Dependent implementation",
        assignee="builder",
        parents=[parent_id],
    )

    # Move the task through review while its parent is temporarily terminal,
    # then make the parent non-terminal again before changes are requested.
    assert kb.complete_task(conn, parent_id)
    implementation = kb.claim_task(conn, task_id, claimer="builder:1")
    assert implementation is not None
    assert kb.request_review(
        conn,
        task_id,
        reviewer="reviewer",
        summary="Ready for review.",
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(conn, task_id, claimer="reviewer:1")
    assert review is not None
    conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (parent_id,))
    conn.commit()

    assert kb.request_changes(
        conn,
        task_id,
        reason="Parent contract changed; rework after it lands.",
        expected_run_id=review.current_run_id,
    ) == (True, "builder")
    regated = kb.get_task(conn, task_id)
    assert regated is not None
    assert regated.status == "todo"


def test_request_changes_fails_closed_on_malformed_review_provenance(conn):
    task_id = kb.create_task(conn, title="Malformed handoff", assignee="builder")
    implementation = kb.claim_task(conn, task_id, claimer="builder:1")
    assert implementation is not None
    assert kb.request_review(
        conn,
        task_id,
        reviewer="reviewer",
        summary="Ready.",
        expected_run_id=implementation.current_run_id,
    )
    conn.execute(
        "UPDATE task_events SET payload = ? "
        "WHERE task_id = ? AND kind = 'review_requested'",
        ("{malformed-json", task_id),
    )
    conn.commit()
    review = kb.claim_review_task(conn, task_id, claimer="reviewer:1")
    assert review is not None

    ok, detail = kb.request_changes(
        conn,
        task_id,
        reason="Needs changes.",
        expected_run_id=review.current_run_id,
    )
    assert ok is False
    assert "implementer provenance" in (detail or "")
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "running"
    assert task.assignee == "reviewer"
    assert task.current_run_id == review.current_run_id


@pytest.mark.parametrize(
    "reclaim_kind",
    ["spawn_failure", "expired_claim", "manual_reclaim", "stale_heartbeat"],
)
def test_interrupted_review_runs_retry_in_review_phase(
    conn,
    reclaim_kind: str,
) -> None:
    task_id, review = _claimed_review(
        conn,
        f"Retry review after {reclaim_kind}",
        ttl_seconds=-1 if reclaim_kind == "expired_claim" else None,
    )

    if reclaim_kind == "spawn_failure":
        assert not kb._record_spawn_failure(
            conn,
            task_id,
            "reviewer process failed to spawn",
            failure_limit=3,
        )
    elif reclaim_kind == "expired_claim":
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET claim_expires = ? WHERE id = ?",
                (int(time.time()) - 1, task_id),
            )
        assert kb.release_stale_claims(conn) == 1
    elif reclaim_kind == "manual_reclaim":
        assert kb.reclaim_task(conn, task_id, reason="operator retry")
    else:
        old = int(time.time()) - 1_000
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET started_at = ?, last_heartbeat_at = NULL "
                "WHERE id = ?",
                (old, task_id),
            )
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE id = ?",
                (old, review.current_run_id),
            )
        assert kb.detect_stale_running(conn, stale_timeout_seconds=1) == [task_id]

    retried = kb.get_task(conn, task_id)
    assert retried is not None
    assert retried.status == "review"
    assert retried.current_run_id is None
    event = kb.list_events(conn, task_id=task_id)[-1]
    assert event.payload is not None
    assert event.payload.get("retry_status") == "review"


def test_review_retry_still_trips_the_failure_breaker(conn) -> None:
    task_id, _review = _claimed_review(conn, "Reviewer repeatedly fails")
    assert kb._record_spawn_failure(
        conn,
        task_id,
        "reviewer cannot start",
        failure_limit=1,
    )
    blocked = kb.get_task(conn, task_id)
    assert blocked is not None
    assert blocked.status == "blocked"
    gave_up = _event(kb.list_events(conn, task_id), "gave_up")
    assert gave_up.payload is not None
    assert gave_up.payload["retry_status"] == "review"
    assert kb.unblock_task(conn, task_id)
    unblocked = kb.get_task(conn, task_id)
    assert unblocked is not None
    assert unblocked.status == "review"


def test_review_escalation_unblocks_back_to_review(conn) -> None:
    task_id, review = _claimed_review(conn, "External review escalation")
    assert kb.block_task(
        conn,
        task_id,
        reason="needs_input: maintainer decision required",
        kind="needs_input",
        expected_run_id=review.current_run_id,
    )
    blocked_event = _event(kb.list_events(conn, task_id), "blocked")
    assert blocked_event.payload is not None
    assert blocked_event.payload["source_status"] == "review"
    assert kb.unblock_task(conn, task_id)
    resumed = kb.get_task(conn, task_id)
    assert resumed is not None
    assert resumed.status == "review"


def test_review_dependency_wait_reenters_review_after_parent_finishes(conn) -> None:
    parent_id = kb.create_task(conn, title="Parent", assignee="planner")
    assert kb.complete_task(conn, parent_id)
    task_id = kb.create_task(
        conn,
        title="Review after dependency refresh",
        assignee="builder",
        parents=[parent_id],
    )
    implementation = kb.claim_task(conn, task_id)
    assert implementation is not None
    assert kb.request_review(
        conn,
        task_id,
        summary="ready",
        reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(conn, task_id)
    assert review is not None
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (parent_id,))
    assert kb.block_task(
        conn,
        task_id,
        reason="dependency: parent contract is being refreshed",
        kind="dependency",
        expected_run_id=review.current_run_id,
    )
    waiting = kb.get_task(conn, task_id)
    assert waiting is not None
    assert waiting.status == "todo"
    assert kb.complete_task(conn, parent_id)
    resumed = kb.get_task(conn, task_id)
    assert resumed is not None
    assert resumed.status == "review"


def test_crashed_and_timed_out_review_runs_retry_in_review_phase(
    conn,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kb, "_classify_worker_exit", lambda _pid: ("nonzero_exit", 1))
    old = int(time.time()) - 1_000

    timed_out_id, timed_out_run = _claimed_review(
        conn,
        "Timeout during review",
        max_runtime_seconds=1,
    )
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, started_at = ? WHERE id = ?",
            (999_998, old, timed_out_id),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid = ?, started_at = ? WHERE id = ?",
            (999_998, old, timed_out_run.current_run_id),
        )
    assert timed_out_id in kb.enforce_max_runtime(conn, signal_fn=lambda *_: None)
    timed_out = kb.get_task(conn, timed_out_id)
    assert timed_out is not None
    assert timed_out.status == "review"

    crashed_id, crashed_run = _claimed_review(conn, "Crash during review")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, started_at = ? WHERE id = ?",
            (999_999, old, crashed_id),
        )
        conn.execute(
            "UPDATE task_runs SET worker_pid = ?, started_at = ? WHERE id = ?",
            (999_999, old, crashed_run.current_run_id),
        )
    assert crashed_id in kb.detect_crashed_workers(conn)
    crashed = kb.get_task(conn, crashed_id)
    assert crashed is not None
    assert crashed.status == "review"


def test_legacy_review_child_deadlock_is_reported_immediately(conn):
    implementation_id = kb.create_task(
        conn,
        title="Implement export",
        assignee="builder",
    )
    reviewer_id = kb.create_task(
        conn,
        title="Review export",
        assignee="reviewer",
        parents=[implementation_id],
    )
    implementation = kb.claim_task(conn, implementation_id, claimer="builder:1")
    assert implementation is not None
    assert kb.block_task(
        conn,
        implementation_id,
        reason="review-required: implementation ready for independent review",
        expected_run_id=implementation.current_run_id,
    )
    reviewer_task = kb.get_task(conn, reviewer_id)
    assert reviewer_task is not None
    assert reviewer_task.status == "todo"
    assert kb.recompute_ready(conn) == 0

    task = kb.get_task(conn, implementation_id)
    diagnostics = kd.compute_task_diagnostics(
        task,
        kb.list_events(conn, implementation_id),
        kb.list_runs(conn, implementation_id),
        graph={
            "children": [
                {
                    "id": reviewer_id,
                    "title": "Review export",
                    "status": "todo",
                }
            ]
        },
    )

    deadlocks = [d for d in diagnostics if d.kind == "review_dependency_deadlock"]
    assert len(deadlocks) == 1
    deadlock = deadlocks[0]
    assert deadlock.severity == "error"
    assert deadlock.data["blocked_parent_id"] == implementation_id
    assert deadlock.data["waiting_child_ids"] == [reviewer_id]
    assert any(action.kind == "cli_hint" for action in deadlock.actions)


def test_hard_block_with_waiting_child_is_not_mislabeled_as_review_deadlock(conn):
    implementation_id = kb.create_task(
        conn, title="Implement export", assignee="builder"
    )
    child_id = kb.create_task(
        conn,
        title="Publish export",
        assignee="release",
        parents=[implementation_id],
    )
    implementation = kb.claim_task(conn, implementation_id, claimer="builder:1")
    assert implementation is not None
    assert kb.block_task(
        conn,
        implementation_id,
        reason="needs_input: production credentials unavailable",
        expected_run_id=implementation.current_run_id,
    )

    diagnostics = kd.compute_task_diagnostics(
        kb.get_task(conn, implementation_id),
        kb.list_events(conn, implementation_id),
        kb.list_runs(conn, implementation_id),
        graph={
            "children": [{"id": child_id, "title": "Publish export", "status": "todo"}]
        },
    )
    assert not any(d.kind == "review_dependency_deadlock" for d in diagnostics)

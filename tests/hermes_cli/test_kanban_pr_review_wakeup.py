"""The fm #37 gap directly: a task's declared GitHub PR (`completion_contract`)
gets human review approval while the kanban card is still `running` and
nobody is told. `detect_stale_waiting` only watches kanban's OWN
`review`/`blocked` status; this is the missing GitHub-PR-specific piece.

All tests inject a fake `query_fn` -- no real network access, no `gh` CLI
dependency, deterministic.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


PR_URL = "https://github.com/qwickapps/fm/pull/37"


def _running_task_with_pr(conn, pr_url: str = PR_URL) -> str:
    task_id = kb.create_task(
        conn, title="t", assignee="primex", completion_contract=pr_url,
    )
    kb.claim_task(conn, task_id)
    return task_id


def test_disabled_by_default_makes_no_query(conn):
    calls = []
    _running_task_with_pr(conn)
    result = kbd.detect_stale_pr_review_ready(
        conn, enabled=False, query_fn=lambda url: calls.append(url) or {"review_decision": "APPROVED"},
    )
    assert result == []
    assert calls == [], "enabled=False must never call query_fn (no network by default)"


def test_approved_review_escalates_once(conn):
    task_id = _running_task_with_pr(conn)
    query_fn = lambda url: {"review_decision": "APPROVED", "state": "OPEN", "merged": False}  # noqa: E731

    first = kbd.detect_stale_pr_review_ready(conn, enabled=True, query_fn=query_fn)
    assert first == [{"id": task_id, "pr_url": PR_URL, "review_decision": "APPROVED"}]

    events = [e.kind for e in kb.list_events(conn, task_id)]
    assert events.count("pr_review_ready_escalated") == 1
    assert "pr_review_checked" in events

    # Second call is rate-limited by min_check_interval_seconds (default 900s,
    # nothing backdated) -- no new query, no re-escalation.
    second = kbd.detect_stale_pr_review_ready(conn, enabled=True, query_fn=query_fn)
    assert second == []
    assert [e.kind for e in kb.list_events(conn, task_id)].count("pr_review_ready_escalated") == 1


def test_not_yet_approved_does_not_escalate(conn):
    task_id = _running_task_with_pr(conn)
    for decision in (None, "REVIEW_REQUIRED", "CHANGES_REQUESTED"):
        query_fn = lambda url, d=decision: {"review_decision": d, "state": "OPEN", "merged": False}
        result = kbd.detect_stale_pr_review_ready(
            conn, enabled=True, min_check_interval_seconds=0, query_fn=query_fn,
        )
        assert result == [], f"must not escalate on review_decision={decision!r}"
    assert kb.get_task(conn, task_id).status == "running"


def test_merged_pr_does_not_escalate(conn):
    """Already merged -- the card should complete normally; escalating here
    would be a duplicate, unhelpful nudge."""
    _running_task_with_pr(conn)
    query_fn = lambda url: {"review_decision": "APPROVED", "state": "MERGED", "merged": True}  # noqa: E731
    result = kbd.detect_stale_pr_review_ready(conn, enabled=True, query_fn=query_fn)
    assert result == []


def test_query_error_does_not_escalate_or_crash(conn):
    _running_task_with_pr(conn)
    query_fn = lambda url: {"error": "GitHub API unreachable"}  # noqa: E731
    result = kbd.detect_stale_pr_review_ready(conn, enabled=True, query_fn=query_fn)
    assert result == []


def test_non_pr_completion_contract_is_never_queried(conn):
    calls = []
    kb.create_task(conn, title="local", assignee="anika", completion_contract="local-only")
    kb.create_task(conn, title="repo-only", assignee="anika", completion_contract="qwickapps/fm")
    result = kbd.detect_stale_pr_review_ready(
        conn, enabled=True, query_fn=lambda url: calls.append(url) or {"review_decision": "APPROVED"},
    )
    assert result == []
    assert calls == []


def test_reapproval_after_change_escalates_again(conn):
    """A PR that was approved, then had changes requested, then got
    re-approved is genuinely new information -- escalate again."""
    task_id = _running_task_with_pr(conn)
    approved = lambda url: {"review_decision": "APPROVED", "state": "OPEN", "merged": False}  # noqa: E731
    changes = lambda url: {"review_decision": "CHANGES_REQUESTED", "state": "OPEN", "merged": False}  # noqa: E731

    first = kbd.detect_stale_pr_review_ready(
        conn, enabled=True, min_check_interval_seconds=0, query_fn=approved,
    )
    assert len(first) == 1
    second = kbd.detect_stale_pr_review_ready(
        conn, enabled=True, min_check_interval_seconds=0, query_fn=changes,
    )
    assert second == []
    third = kbd.detect_stale_pr_review_ready(
        conn, enabled=True, min_check_interval_seconds=0, query_fn=approved,
    )
    assert len(third) == 1
    assert [e.kind for e in kb.list_events(conn, task_id)].count("pr_review_ready_escalated") == 2


def test_fetch_pr_review_state_rejects_malformed_url():
    from hermes_cli.kanban_pr_acceptance import fetch_pr_review_state

    result = fetch_pr_review_state("not-a-url")
    assert "error" in result


def test_active_pr_recovery_routes_terminal_pr_to_recovery_block(conn):
    """A closed linked PR is authoritative lifecycle evidence, unlike prose."""
    task_id = kb.create_task(conn, title="terminal PR", assignee="builder")
    kb.add_comment(conn, task_id, author="builder", body=f"Opened {PR_URL}")

    recovered = kbd.reconcile_active_pr_recoveries(
        conn,
        enabled=True,
        query_fn=lambda _url: {
            "state": "CLOSED", "merged": False, "review_decision": None,
            "head_sha": "a" * 40,
        },
    )

    assert recovered == [{"id": task_id, "reason": "pr_terminal", "pr_url": PR_URL}]
    task = kb.get_task(conn, task_id)
    assert task.status == "blocked"
    assert task.block_kind == "needs_input"
    assert "active_pr_recovered" in [event.kind for event in kb.list_events(conn, task_id)]


def test_active_pr_recovery_routes_distinct_exact_sha_verdict_to_review(conn):
    """Only a distinct reviewer plus the matching recorded head can release it."""
    task_id = kb.create_task(conn, title="reviewed PR", assignee="builder")
    implementation = kb.claim_task(conn, task_id)
    assert implementation is not None
    assert kb.request_review(
        conn,
        task_id,
        reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
        metadata={"head_sha": "a" * 40},
    )
    review = kb.claim_review_task(conn, task_id)
    assert review is not None
    assert kb.request_changes(
        conn, task_id, reason="reconcile current PR", expected_run_id=review.current_run_id,
    ) == (True, "builder")
    kb.add_comment(conn, task_id, author="builder", body=f"Pushed {PR_URL}")

    recovered = kbd.reconcile_active_pr_recoveries(
        conn,
        enabled=True,
        query_fn=lambda _url: {
            "state": "OPEN", "merged": False, "review_decision": "CHANGES_REQUESTED",
            "head_sha": "a" * 40,
        },
    )

    assert recovered == [{
        "id": task_id, "reason": "distinct_exact_sha_review_verdict", "pr_url": PR_URL,
    }]
    task = kb.get_task(conn, task_id)
    assert (task.status, task.assignee) == ("review", "reviewer")
    assert kbd.check_respawn_guard(conn, task_id, lane="review") is None


def test_active_pr_recovery_binds_verdict_to_matching_review_round(conn):
    """A current older SHA must not adopt a later review round's reviewer."""
    task_id = kb.create_task(conn, title="two review rounds", assignee="builder")
    first_implementation = kb.claim_task(conn, task_id)
    assert first_implementation is not None
    assert kb.request_review(
        conn,
        task_id,
        reviewer="reviewer_a",
        expected_run_id=first_implementation.current_run_id,
        metadata={"head_sha": "a" * 40},
    )
    first_review = kb.claim_review_task(conn, task_id)
    assert first_review is not None
    assert kb.request_changes(
        conn, task_id, reason="first round", expected_run_id=first_review.current_run_id,
    ) == (True, "builder")

    second_implementation = kb.claim_task(conn, task_id)
    assert second_implementation is not None
    assert kb.request_review(
        conn,
        task_id,
        reviewer="reviewer_b",
        expected_run_id=second_implementation.current_run_id,
        metadata={"head_sha": "b" * 40},
    )
    second_review = kb.claim_review_task(conn, task_id)
    assert second_review is not None
    assert kb.request_changes(
        conn, task_id, reason="second round", expected_run_id=second_review.current_run_id,
    ) == (True, "builder")
    kb.add_comment(conn, task_id, author="builder", body=f"Pushed {PR_URL}")

    recovered = kbd.reconcile_active_pr_recoveries(
        conn,
        enabled=True,
        query_fn=lambda _url: {
            "state": "OPEN", "merged": False, "review_decision": "CHANGES_REQUESTED",
            "head_sha": "a" * 40,
        },
    )

    assert recovered == [{
        "id": task_id, "reason": "distinct_exact_sha_review_verdict", "pr_url": PR_URL,
    }]
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert (task.status, task.assignee) == ("review", "reviewer_a")


def test_active_pr_recovery_rejects_stale_review_sha(conn):
    """A review attached to an old head cannot unlock a newer PR head."""
    task_id = kb.create_task(conn, title="stale review", assignee="builder")
    implementation = kb.claim_task(conn, task_id)
    assert implementation is not None
    assert kb.request_review(
        conn,
        task_id,
        reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
        metadata={"head_sha": "a" * 40},
    )
    review = kb.claim_review_task(conn, task_id)
    assert review is not None
    assert kb.request_changes(
        conn, task_id, reason="old head", expected_run_id=review.current_run_id,
    ) == (True, "builder")
    kb.add_comment(conn, task_id, author="builder", body=f"Pushed {PR_URL}")

    assert kbd.reconcile_active_pr_recoveries(
        conn,
        enabled=True,
        query_fn=lambda _url: {
            "state": "OPEN", "merged": False, "review_decision": "APPROVED",
            "head_sha": "b" * 40,
        },
    ) == []
    assert kb.get_task(conn, task_id).status == "ready"
    assert kbd.check_respawn_guard(conn, task_id) == "active_pr"

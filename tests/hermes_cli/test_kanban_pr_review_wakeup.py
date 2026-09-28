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

"""A successful handoff supersedes a proven failure stamp, not a fresh auth error.

Regression for upstream #138618; builds on the read-side approach in #137833.
"""
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_pr_acceptance_store import record_acceptance

QUOTA = "pid 42 exited rate-limited (quota wall) — requeued without counting a failure"


@pytest.fixture(params=["rate_limited", "crashed"])
def recovered(tmp_path, monkeypatch, request):
    error = QUOTA if request.param == "rate_limited" else ('pid 42 exited with code 1: billing in old output; ' + 'x' * 600)
    monkeypatch.setattr(kb.time, "time", lambda: 5_000_000)
    with kbc.connect(tmp_path / "kanban.db") as conn:
        tid = kb.create_task(conn, title="Recovered failure", assignee="builder")
        kb.claim_task(conn, tid)
        kb._end_run(conn, tid, outcome=request.param, error=error)
        conn.execute(
            "UPDATE tasks SET status='ready', claim_lock=NULL, claim_expires=NULL, "
            "worker_pid=NULL, last_failure_error=?, consecutive_failures=2, "
            "block_recurrences=1 WHERE id=?", (error[:500], tid),
        )
        conn.commit()
        monkeypatch.setattr(kb.time, "time", lambda: 5_000_001)
        yield conn, tid, error[:500]


@pytest.mark.parametrize("reviewer,same_second", [("builder", False), ("reviewer", False), ("reviewer", True)])
def test_successful_handoff_releases_stale_failure_without_changing_accounting(recovered, reviewer, same_second, monkeypatch):
    conn, tid, error = recovered
    if same_second:
        monkeypatch.setattr(kb.time, "time", lambda: 5_000_000)
    claimed = kb.claim_task(conn, tid)
    assert kb.request_review(conn, tid, reviewer=reviewer, summary="Ready for review",
                             expected_run_id=claimed.current_run_id)
    before = tuple(conn.execute(
        "SELECT consecutive_failures, block_recurrences, last_failure_error FROM tasks WHERE id=?", (tid,),
    ).fetchone())
    assert before == (2, 1, error)
    # The newer run owns current state, including equal-timestamp handoffs.
    assert kbd.check_respawn_guard(conn, tid, lane="review") is None
    assert tuple(conn.execute(
        "SELECT consecutive_failures, block_recurrences, last_failure_error FROM tasks WHERE id=?", (tid,),
    ).fetchone()) == before


@pytest.mark.parametrize("outcome,expected", [
    ("review_requested", None), ("changes_requested", None),
    ("completed", "recent_success"), ("spawn_failed", "blocker_auth"),
    ("failed", "blocker_auth"), ("blocked", "blocker_auth"),
])
def test_only_successful_newer_run_supersedes_matching_failure(recovered, outcome, expected):
    conn, tid, _error = recovered
    kb.claim_task(conn, tid)
    kb._end_run(conn, tid, outcome=outcome)
    assert kbd.check_respawn_guard(conn, tid) == expected


@pytest.mark.parametrize("current_error", ["acceptance", "unmatched", "unproven"])
def test_successful_handoff_does_not_suppress_current_auth_error(recovered, current_error):
    conn, tid, _error = recovered
    claimed = kb.claim_task(conn, tid)
    assert kb.request_review(conn, tid, reviewer="reviewer", summary="Ready",
                             expected_run_id=claimed.current_run_id)
    if current_error == "acceptance":
        task = kb.get_task(conn, tid)
        assert not record_acceptance(conn, tid, (
            (None, "review", task.completion_contract),
            {"ok": False, "classification": "auth", "detail": "authentication failed", "recovery": "login"},
        ))
    elif current_error == "unproven":
        conn.execute("UPDATE task_runs SET error=NULL WHERE task_id=? AND outcome IN ('rate_limited', 'crashed')", (tid,))
    else:
        conn.execute("UPDATE tasks SET last_failure_error='provider authentication failed' WHERE id=?", (tid,))
    conn.commit()
    assert kbd.check_respawn_guard(conn, tid, lane="review") == "blocker_auth"

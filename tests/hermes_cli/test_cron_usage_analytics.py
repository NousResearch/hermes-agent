from __future__ import annotations

from datetime import datetime, timedelta, timezone

from cron import executions
from hermes_state import SessionDB


def _seed_cron_session(
    db: SessionDB,
    session_id: str,
    *,
    started_at: float,
    input_tokens: int,
    output_tokens: int,
    estimated_cost: float | None,
    actual_cost: float | None,
    end_reason: str = "cron_complete",
) -> None:
    db.create_session(session_id=session_id, source="cron", model="test-model")
    conn = db._conn
    assert conn is not None
    with db._lock:
        conn.execute(
            """UPDATE sessions
               SET started_at=?, ended_at=?, end_reason=?, input_tokens=?, output_tokens=?,
                   estimated_cost_usd=?, actual_cost_usd=?
               WHERE id=?""",
            (
                started_at,
                started_at + 1,
                end_reason,
                input_tokens,
                output_tokens,
                estimated_cost,
                actual_cost,
                session_id,
            ),
        )
        conn.commit()


def test_cron_job_usage_analytics_scopes_period_and_prefers_actual_cost(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        now = datetime.now(timezone.utc).timestamp()
        _seed_cron_session(
            db,
            "cron_job-alpha_20260909_100000",
            started_at=now - 60,
            input_tokens=100,
            output_tokens=20,
            estimated_cost=0.30,
            actual_cost=0.25,
        )
        _seed_cron_session(
            db,
            "cron_job-alpha_20260909_110000",
            started_at=now - 30,
            input_tokens=200,
            output_tokens=40,
            estimated_cost=0.50,
            actual_cost=None,
            end_reason="cron_incomplete_no_output",
        )
        _seed_cron_session(
            db,
            "cron_job-alpha_20250909_100000",
            started_at=now - 100 * 86400,
            input_tokens=999,
            output_tokens=999,
            estimated_cost=9.99,
            actual_cost=None,
        )
        _seed_cron_session(
            db,
            "cron_job-beta_20260909_100000",
            started_at=now - 10,
            input_tokens=999,
            output_tokens=999,
            estimated_cost=9.99,
            actual_cost=None,
        )

        result = db.cron_job_usage_analytics("job-alpha", since=now - 30 * 86400)

        assert result == {
            "usage_runs": 2,
            "total_tokens": 360,
            "avg_tokens_per_run": 180.0,
            "total_cost_usd": 0.75,
            "avg_cost_usd_per_run": 0.375,
            "actual_cost_runs": 1,
            "estimated_cost_runs": 1,
            "unknown_cost_runs": 0,
            "incomplete_runs": 1,
        }
    finally:
        db.close()


def test_execution_analytics_identifies_manual_failures_and_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    scheduled = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    first = executions.create_execution("job-alpha", source="builtin", scheduled_instant=scheduled)
    executions.mark_execution_running(first["id"])
    executions.finish_execution(first["id"], success=False, error="first attempt failed")

    retry = executions.create_execution("job-alpha", source="builtin", scheduled_instant=scheduled)
    executions.mark_execution_running(retry["id"])
    executions.finish_execution(retry["id"], success=True)

    manual = executions.create_execution("job-alpha", source="direct")
    executions.mark_execution_running(manual["id"])
    executions.finish_execution(manual["id"], success=True)

    result = executions.execution_analytics(
        "job-alpha",
        since=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
    )

    assert result == {
        "attempts": 3,
        "completed": 2,
        "failed": 1,
        "unknown": 0,
        "running": 0,
        "manual_runs": 1,
        "scheduled_runs": 1,
        "retry_attempts": 1,
    }

from __future__ import annotations

import pytest
from fastapi import HTTPException

from hermes_cli.web_routers import cron as cron_router


class _FakeAnalyticsDB:
    def __init__(self, token_total: int):
        self.token_total = token_total

    def cron_job_usage_analytics(self, job_id: str, *, since: float):
        return {
            "usage_runs": 1,
            "total_tokens": self.token_total,
            "avg_tokens_per_run": float(self.token_total),
            "total_cost_usd": 0.1,
            "avg_cost_usd_per_run": 0.1,
            "actual_cost_runs": 1,
            "estimated_cost_runs": 0,
            "unknown_cost_runs": 0,
            "incomplete_runs": 0,
        }

    def close(self):
        pass


def test_dashboard_analytics_is_profile_scoped_and_validates_period(monkeypatch):
    jobs = [
        {"id": "default-job", "profile": "default"},
        {"id": "worker-job", "profile": "worker"},
    ]
    monkeypatch.setattr(cron_router, "_list_cron_jobs_sync", lambda profile: jobs)
    monkeypatch.setattr(
        cron_router,
        "_open_session_db_for_profile",
        lambda profile, read_only: _FakeAnalyticsDB(
            10 if profile == "default" else 20
        ),
    )
    monkeypatch.setattr(
        cron_router,
        "_cron_execution_analytics_for_profile",
        lambda profile, job_id, since: {
            "attempts": 1,
            "completed": 1,
            "failed": 0,
            "unknown": 0,
            "running": 0,
            "manual_runs": 0,
            "scheduled_runs": 1,
            "retry_attempts": 0,
        },
    )

    result = cron_router._list_cron_analytics_sync("all", 7)

    assert result["period_days"] == 7
    assert {(item["profile"], item["total_tokens"]) for item in result["jobs"]} == {
        ("default", 10),
        ("worker", 20),
    }
    with pytest.raises(HTTPException) as exc_info:
        cron_router._list_cron_analytics_sync("all", 14)
    assert exc_info.value.status_code == 400

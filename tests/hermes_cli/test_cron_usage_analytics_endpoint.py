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
            "history_limited": False,
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



@pytest.mark.parametrize("worker_store", ["initialized", "missing", "empty"])
def test_analytics_http_reads_real_profile_stores_and_reports_limited_history(tmp_path, monkeypatch, worker_store):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cron import executions
    from hermes_cli import web_server_cron
    from hermes_constants import (
        get_hermes_home_override, reset_hermes_home_override, set_hermes_home_override,
    )
    from hermes_state import SessionDB

    homes = {profile: tmp_path / profile for profile in ("default", "worker")}
    monkeypatch.setattr(web_server_cron, "_cron_profile_home", lambda profile: (profile, homes[profile]))
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 2)
    for profile, tokens in (("default", 10), ("worker", 20)):
        home = homes[profile]
        home.mkdir()
        if profile == "worker" and worker_store != "initialized":
            if worker_store == "empty":
                (home / "state.db").touch()
        else:
            with SessionDB(home / "state.db") as db:
                sid = "cron_shared_20261026_100000"
                db.create_session(sid, source="cron")
                db.update_token_counts(sid, input_tokens=tokens, estimated_cost_usd=0.1)
                db.end_session(sid, "cron_complete")
        token = set_hermes_home_override(str(home))
        try:
            row = executions.create_execution("shared", source="direct")
            executions.mark_execution_running(row["id"])
            executions.finish_execution(row["id"], success=profile == "worker")
            if profile == "default":
                for _ in range(2):
                    noisy = executions.create_execution("noisy", source="direct")
                    executions.mark_execution_running(noisy["id"])
                    executions.finish_execution(noisy["id"], success=True)
        finally:
            reset_hermes_home_override(token)

    monkeypatch.setattr(cron_router, "_list_cron_jobs_sync", lambda profile: [
        {"id": "shared", "profile": name} for name in homes
    ])
    app = FastAPI()
    app.include_router(cron_router.router)
    previous_home = get_hermes_home_override()
    with TestClient(app) as client:
        response = client.get("/api/cron/analytics?profile=all&days=30")
        assert response.status_code == 200
        entries = {row["profile"]: row for row in response.json()["jobs"]}
        assert entries["default"]["history_limited"] is True
        assert entries["default"]["attempts"] == 0
        assert entries["default"]["usage_runs"] == 1
        assert entries["default"]["total_tokens"] == 10
        assert entries["worker"]["history_limited"] is False
        assert entries["worker"]["completed"] == 1
        assert entries["worker"]["total_tokens"] == (20 if worker_store == "initialized" else 0)
        assert entries["worker"]["usage_runs"] == (1 if worker_store == "initialized" else 0)
        if worker_store != "initialized":
            assert entries["worker"]["avg_tokens_per_run"] is None
            assert entries["worker"]["avg_cost_usd_per_run"] is None
            path = homes["worker"] / "state.db"
            assert not path.exists() if worker_store == "missing" else path.stat().st_size == 0
        assert client.get("/api/cron/analytics?days=14").status_code == 400
    assert get_hermes_home_override() == previous_home



@pytest.mark.parametrize("status, detail", [
    (503, "Session store is busy (disk I/O or lock). Retry; the list was not cleared."),
    (503, {"code": "corrupt_store", "message": "Run hermes doctor"}),
    (500, "Session store is not initialized."),
])
def test_analytics_preserves_non_initialization_store_errors(monkeypatch, status, detail):
    error = HTTPException(status_code=status, detail=detail)
    monkeypatch.setattr(cron_router, "_list_cron_jobs_sync", lambda profile: [
        {"id": "shared", "profile": "worker"},
    ])

    def refuse(profile, *, read_only):
        raise error

    monkeypatch.setattr(cron_router, "_open_session_db_for_profile", refuse)
    with pytest.raises(HTTPException) as caught:
        cron_router._list_cron_analytics_sync("all", 30)
    assert caught.value is error

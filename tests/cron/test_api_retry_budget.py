"""Per-job retry budget survives the public tool/store path."""
import json

import pytest

from cron import jobs
from tools import cronjob_tools
from tools.registry import registry


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "_compute_provider_model_snapshots", lambda **kw: (None, None))
    # No scheduler is launched; only registration is replaced, not persistence.
    monkeypatch.setattr("cron.scheduler.create_job_with_scheduler_registration", jobs.create_job)
    return registry.get_entry("cronjob_manage").handler


def test_tool_create_update_reload_and_clear(store):
    created = json.loads(store({"action": "create", "prompt": "ping", "schedule": "every 1h",
        "api_max_retries": 8}))
    assert created["success"], created
    job = jobs.load_jobs()[0]
    assert job["api_max_retries"] == 8
    for value, expected in [(10, 10), (1, 1), ("", None)]:
        updated = json.loads(store({"action": "update", "job_id": job["id"], "api_max_retries": value}))
        assert updated["success"], updated
        assert jobs.get_job(job["id"]).get("api_max_retries") == expected


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "bad", float("inf")])
def test_invalid_retry_budget_rejected_before_store(store, value):
    result = json.loads(store({"action": "create", "prompt": "ping", "schedule": "every 1h",
        "api_max_retries": value}))
    assert not result["success"]
    assert jobs.load_jobs() == []

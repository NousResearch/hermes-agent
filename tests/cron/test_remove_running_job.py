"""Regression for #123045: removal must not silently cancel a paid in-flight run."""

import json
from datetime import timedelta

import pytest

from cron import jobs
from tools.cronjob_tools import cronjob


@pytest.mark.parametrize("claim_field", ["fire_claim", "run_claim"])
def test_remove_refuses_live_claim_without_losing_run(tmp_path, claim_field):
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(
            prompt="fixture", schedule=(jobs._hermes_now() + timedelta(hours=1)).isoformat(),
            name="running fixture")
        claimed = jobs.claim_job_for_fire(job["id"], manual=True, return_job=True)
        if claim_field == "run_claim":
            claimed["run_claim"] = claimed.pop("fire_claim")
            jobs.save_jobs([claimed])
        output = jobs.save_job_output(job["id"], "existing output")
        result = json.loads(cronjob(action="remove", job_id=job["id"]))
        assert result["success"] is False
        assert "running" in result["error"].lower()
        assert jobs.get_job(job["id"])[claim_field] == claimed[claim_field]
        assert output.read_text() == "existing output"
        heartbeat = jobs.heartbeat_fire_claim if claim_field == "fire_claim" else jobs.heartbeat_run_claim
        assert heartbeat(job["id"], expected_owner=claimed[claim_field]["by"])
        assert jobs.mark_job_run(job["id"], success=True)
        assert json.loads(cronjob(action="remove", job_id=job["id"]))["success"] is True
        # Expired leases must not make abandoned jobs impossible to remove.
        stale = jobs.create_job(prompt="fixture", schedule="every 1h")
        stale[claim_field] = {
            "at": (jobs._hermes_now() - timedelta(days=1)).isoformat(), "by": "fixture-owner"}
        jobs.save_jobs([stale])
        assert jobs.remove_job(stale["id"])
        assert not jobs.remove_job(stale["id"])


def test_self_removal_preserves_delivery_exception(tmp_path):
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="fixture", schedule="every 1h")
        assert jobs.claim_job_for_fire(job["id"], manual=True)
        with jobs.self_removal_delivery_scope(job["id"]):
            assert json.loads(cronjob(action="remove", job_id=job["id"]))["success"] is True
            assert jobs.self_removal_delivery_allowed(job["id"])

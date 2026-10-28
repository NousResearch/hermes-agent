"""Pre-fire provider usage-limit gate for cron (cron/quota_hold.py)."""

import time

import pytest

from agent import usage_hold
from cron import jobs as cron_jobs
from cron.quota_hold import STATE_KEY, hold_due_jobs_for_provider_limits

_CFG = {"model": {"provider": "custom", "base_url": "http://127.0.0.1:9/v1"}}


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cron_jobs, "CRON_DIR", tmp_path / "cron", raising=False)
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json", raising=False)
    # The gate resolves routes like run_job does, from the profile's persisted config.
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: custom\n  base_url: http://127.0.0.1:9/v1\n  default: m\n", encoding="utf-8")
    return tmp_path


def _due(name, **extra):
    job = cron_jobs.create_job(prompt="hi", schedule="every 30m", name=name, **extra)
    cron_jobs.update_job(job["id"], {"next_run_at": cron_jobs._hermes_now().isoformat()})
    return cron_jobs.get_job(job["id"])


def test_held_provider_parks_due_jobs_wait_and_skip(home):
    wait_job, skip_job = _due("w"), _due("s")
    cron_jobs.update_job(skip_job["id"], {"on_limit": "skip"})
    until = time.time() + 3600
    usage_hold.record_hold("custom", base_url="http://127.0.0.1:9/v1", until=until)

    runnable = hold_due_jobs_for_provider_limits(
        [cron_jobs.get_job(wait_job["id"]), cron_jobs.get_job(skip_job["id"])], _CFG,
        lambda job, cfg: None)

    assert runnable == []
    waited = cron_jobs.get_job(wait_job["id"])
    skipped = cron_jobs.get_job(skip_job["id"])
    waited_at = cron_jobs._parse_aware(waited["next_run_at"]).timestamp()
    assert until <= waited_at <= until + 120  # one run right after the reset
    assert waited[STATE_KEY] == waited["next_run_at"]
    assert skipped["last_status"] == "skipped_usage_limit"
    assert cron_jobs._parse_aware(skipped["next_run_at"]).timestamp() > waited_at


def test_unheld_provider_or_fallback_chain_still_fires(home):
    job = _due("j")
    usage_hold.record_hold("anthropic", until=time.time() + 3600)
    assert hold_due_jobs_for_provider_limits([job], _CFG, lambda j, c: None) == [job]
    usage_hold.record_hold("custom", base_url="http://127.0.0.1:9/v1", until=time.time() + 3600)
    assert hold_due_jobs_for_provider_limits([job], _CFG, lambda j, c: ["fallback"]) == [job]


def test_bare_custom_pin_meets_the_hold_its_failing_run_recorded(home):
    """A job pinned to ``custom`` with no base_url runs on the configured endpoint and records
    ``custom:<url>``; the gate must look up that same key."""
    job = _due("pinned", provider="custom")
    usage_hold.record_hold("custom", base_url="http://127.0.0.1:9/v1", until=time.time() + 3600)
    assert hold_due_jobs_for_provider_limits([cron_jobs.get_job(job["id"])], _CFG, lambda j, c: None) == []


def test_parking_settles_the_dispatch_stamp(home):
    """The due scan stamped ``pending_slot`` before the gate; a parked job must not leave it
    behind or the next scan restores the old due time and the job re-parks every tick."""
    job = _due("stamped")
    cron_jobs.update_job(job["id"], {"pending_slot": {
        "scheduled_at": cron_jobs._hermes_now().isoformat(), "at": cron_jobs._hermes_now().isoformat(),
        "by": cron_jobs._machine_id()}})
    usage_hold.record_hold("custom", base_url="http://127.0.0.1:9/v1", until=time.time() + 3600)
    assert hold_due_jobs_for_provider_limits([cron_jobs.get_job(job["id"])], _CFG, lambda j, c: None) == []
    assert "pending_slot" not in cron_jobs.get_job(job["id"])

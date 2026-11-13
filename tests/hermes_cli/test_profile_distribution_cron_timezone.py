"""A cron job shipped in a profile distribution keeps its pinned timezone for every installer.

The operator shape (managed crons): one ``cron/jobs.json`` in a profile distribution, installed by
users whose profile ``timezone:`` differs. A job that pins ``timezone: America/New_York`` must run
on New York's wall clock for all of them; a job without the key follows each user's own zone.
These tests go through the real install/update path (``install_distribution`` /
``update_distribution``) and the real store API, never a hand-built intermediate value.
"""

from __future__ import annotations

import json
from datetime import datetime, UTC
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("croniter")

import hermes_time
from cron import jobs
from hermes_cli.profile_distribution import (
    DistributionError, DistributionManifest, install_distribution, update_distribution,
    write_manifest)

NY = ZoneInfo("America/New_York")
PRE_MARKET = "0 9 * * 1-5"


@pytest.fixture
def operator_env(tmp_path, monkeypatch):
    """An isolated Hermes root plus a frozen clock (Wed 2026-11-04 18:00 UTC)."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / ".hermes").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_TIMEZONE", raising=False)
    instant = datetime(2026, 11, 4, 18, 0, tzinfo=UTC)
    monkeypatch.setattr(jobs, "_hermes_now",
                        lambda: instant.astimezone(hermes_time.get_timezone() or UTC))
    hermes_time.reset_cache()
    yield tmp_path
    hermes_time.reset_cache()


def _ship(root, records):
    """A distribution whose ``cron/jobs.json`` holds *records* exactly as an operator wrote them."""
    staged = root / "dist"
    (staged / "cron").mkdir(parents=True)
    (staged / "SOUL.md").write_text("Desk agent.\n", encoding="utf-8")
    (staged / "cron" / "jobs.json").write_text(json.dumps({"jobs": records}), encoding="utf-8")
    write_manifest(staged, DistributionManifest(name="desk", version="1.0.0"))
    return staged


def _pre_market(**overrides):
    record = {"id": "premarket0001", "name": "pre-market", "prompt": "Pre-market update",
              "schedule": PRE_MARKET, "timezone": "America/New_York", "deliver": "local"}
    record.update(overrides)
    return record


def _installed(plan):
    with jobs.use_cron_store(plan.target_dir):
        return {job["id"]: job for job in jobs.load_jobs()}


class TestShippedTimezoneIsValidated:
    """A shipped record reaches the store through ``merge_job_definition``, not ``create_job``,
    so it skipped the IANA validation every other door applies. An invalid zone was stored and
    the job silently fell back to each installer's profile zone, firing at a different hour for
    every user with nothing but a once-per-process log warning."""

    def test_an_unknown_zone_is_refused_naming_the_job(self, operator_env):
        staged = _ship(operator_env, [_pre_market(timezone="America/NewYork")])
        with pytest.raises(DistributionError, match="cron job 'pre-market'.*America/NewYork"):
            install_distribution(str(staged), name="desk")

    def test_an_unknown_zone_on_update_leaves_the_live_job_untouched(self, operator_env):
        staged = _ship(operator_env, [_pre_market()])
        plan = install_distribution(str(staged), name="desk")
        store = staged / "cron" / "jobs.json"
        store.write_text(json.dumps({"jobs": [_pre_market(timezone="US/NewYork")]}), encoding="utf-8")
        with pytest.raises(DistributionError, match="US/NewYork"):
            update_distribution("desk")
        assert _installed(plan)["premarket0001"]["timezone"] == "America/New_York"

    def test_the_zone_is_stored_canonically(self, operator_env):
        staged = _ship(operator_env, [_pre_market(timezone=" America/New_York ")])
        plan = install_distribution(str(staged), name="desk")
        assert _installed(plan)["premarket0001"]["timezone"] == "America/New_York"

    def test_an_empty_zone_means_inherit_and_drops_the_key(self, operator_env):
        staged = _ship(operator_env, [_pre_market(timezone="")])
        plan = install_distribution(str(staged), name="desk")
        assert "timezone" not in _installed(plan)["premarket0001"]


@pytest.mark.parametrize("user_zone", ["Europe/London", "Asia/Hong_Kong", "America/Los_Angeles"])
def test_a_shipped_pin_fires_on_new_york_time_for_every_user(operator_env, user_zone):
    """The whole operator flow: install, the user resumes, the pinned job is due 09:00 New York
    whatever the user's zone; the unpinned job follows the user's own 07:00."""
    staged = _ship(operator_env, [
        _pre_market(),
        {"id": "morningbrf01", "name": "morning-brief", "prompt": "Brief", "schedule": "0 7 * * *",
         "deliver": "local"}])
    plan = install_distribution(str(staged), name="desk")
    (plan.target_dir / "config.yaml").write_text(f"timezone: {user_zone}\n", encoding="utf-8")
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(str(plan.target_dir))
    try:
        hermes_time.reset_cache()
        with jobs.use_cron_store(plan.target_dir):
            pre = jobs.resume_job("premarket0001")
            brief = jobs.resume_job("morningbrf01")
    finally:
        reset_hermes_home_override(token)
    assert datetime.fromisoformat(pre["next_run_at"]) == datetime(2026, 11, 5, 9, 0, tzinfo=NY)
    brief_at = datetime.fromisoformat(brief["next_run_at"]).astimezone(ZoneInfo(user_zone))
    assert (brief_at.hour, brief_at.minute) == (7, 0)

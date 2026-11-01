"""Regression tests for the leaked cron claim (Part A).

``tools/cronjob_tools._run_claimed_job`` registered its in-flight claim under
the active cron-scope home but released it without ``home=``. Any scope
change between register and release (delivery rebinds the profile scope)
made the release miss the key, and the owner guard turned it into a no-op:
the claim leaked permanently.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

import cron.scheduler as sched
import tools.cronjob_tools as tools
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


class _Scope:
    """Bind the cron home scope the way ``_profile_cron_scope`` does for a tick."""

    def __init__(self, home):
        self._home = str(home)

    def __enter__(self):
        self._token = set_hermes_home_override(self._home)
        return self

    def __exit__(self, *_exc):
        reset_hermes_home_override(self._token)
        return False


@pytest.fixture(autouse=True)
def _clean_inflight():
    """Reset the in-memory running sets so tests are isolated."""
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._running_since.clear()
    sched._running_futures.clear()
    sched._running_registration_owners.clear()
    sched._running_allowance_s.clear()
    sched._forced_releases.clear()
    sched._forced_release_count = 0
    yield
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._running_since.clear()
    sched._running_futures.clear()
    sched._running_registration_owners.clear()
    sched._running_allowance_s.clear()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def test_claimed_run_releases_under_claiming_home(tmp_path):
    """(1) A job claimed under a non-default profile home completes via the
    cronjob_tools path; afterwards both registries are empty.

    RED without fix A: the release resolves the rebound (launch) home, misses
    the profile key, and the owner guard turns it into a no-op.
    """
    launch_home, profile_home = tmp_path / "launch", tmp_path / "secretary"
    launch_home.mkdir()
    profile_home.mkdir()
    job_id = "secretary-evening"
    leaked_tokens = []

    def _fake_run(job, **kwargs):
        # Whatever runs between register and release rebinds the home scope
        # (delivery resolves another profile) and never restores it.
        leaked_tokens.append(set_hermes_home_override(str(launch_home)))
        return True

    try:
        with _Scope(profile_home):
            with patch("cron.scheduler.run_one_job", side_effect=_fake_run):
                tools._run_claimed_job({"id": job_id, "name": job_id})
        assert sched._running_job_ids == set()
        assert sched._running_fire_owners == {}
        assert sched.get_running_job_ids() == frozenset()
    finally:
        for token in reversed(leaked_tokens):
            with contextlib.suppress(Exception):
                reset_hermes_home_override(token)
        sched.release_running_job(job_id, home=launch_home)
        sched.release_running_job(job_id, home=profile_home)


def test_default_home_claimed_run_registers_and_releases(tmp_path):
    """(2) Neighbour: a default-home job registers and releases correctly
    through the same cronjob_tools path (no scope switch involved)."""
    home = tmp_path / "launch"
    home.mkdir()
    job_id = "default-home-job"
    with _Scope(home), patch("cron.scheduler.run_one_job", return_value=True):
        tools._run_claimed_job({"id": job_id, "name": job_id})
    assert sched._running_job_ids == set()
    assert sched._running_fire_owners == {}
    assert sched.get_running_job_ids() == frozenset()

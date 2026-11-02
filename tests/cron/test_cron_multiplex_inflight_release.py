"""One process ticks every profile: a claim must be released under the key it was registered with.

The cron home scope is a ContextVar the ticker binds per profile. ``_submit_with_guard`` registers
the claim on the TICKER thread inside that scope, but the pool worker's ``finally`` release sits
OUTSIDE ``ctx.run``, where the worker thread resolves the LAUNCH home instead. Releasing without
the registering home therefore discarded a key nothing holds and leaked every secondary profile's
claim: the job skipped its next fire window until the force-release backstop swept it, and the
shutdown drain plus ``hermes.cron.jobs.running`` saw phantom work.
"""

from __future__ import annotations

import concurrent.futures

import cron.scheduler as sched
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


def test_pool_worker_release_clears_the_ticked_profiles_claim(tmp_path):
    """Dispatch a secondary profile's job through the real guard: no claim may survive the run."""
    launch_home, profile_b = tmp_path / "launch", tmp_path / "profile-b"
    launch_home.mkdir()
    profile_b.mkdir()
    job = {"id": "daily-brief", "name": "daily-brief", "schedule": {"kind": "interval"}}
    ran_in = {}

    def _process_job(j):
        # Runs via ctx.run, so it still sees profile B — only the release is unscoped.
        ran_in["home"] = str(sched._get_hermes_home())

    # The launch home is what an unscoped pool worker thread resolves to.
    with _Scope(launch_home), concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        with _Scope(profile_b):
            fut = sched._submit_with_guard(job, pool, _process_job)
        assert fut is not None, "the job must be dispatched"
        fut.result(timeout=30)

        assert ran_in["home"] == str(profile_b)
        assert sched.get_running_job_ids() == frozenset(), (
            "profile B's in-flight claim leaked: the worker released the LAUNCH home's key"
        )
        assert sched._inflight_key(job["id"], profile_b) not in sched._running_job_ids


def test_a_profiles_release_never_clears_another_profiles_claim(tmp_path):
    """Control: the scope-defaulted release only ever reaches its OWN profile's claim."""
    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    job_id = "daily-brief"
    try:
        with _Scope(home_b):
            assert sched.try_register_running_job(job_id) is True
        with _Scope(home_a):
            sched.release_running_job(job_id)
        assert sched._inflight_key(job_id, home_b) in sched._running_job_ids
        sched.release_running_job(job_id, home=home_b)
        assert sched.get_running_job_ids() == frozenset()
    finally:
        sched.release_running_job(job_id, home=home_a)
        sched.release_running_job(job_id, home=home_b)

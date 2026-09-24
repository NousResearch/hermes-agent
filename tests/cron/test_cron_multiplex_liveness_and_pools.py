"""Per-home cron liveness and worker-pool lifetime under one multiplexing process.

Two invariants that a host-wide, process-lifetime view of cron state gets wrong:

(a) Job liveness is per profile. One process ticks every profile, so the bare-id union that the
    shutdown drain needs would make profile A's running ``daily-brief`` answer for profile B's
    idle one — reporting B's job as running and keeping B's stale one-shot alive as a "possibly
    live run".
(b) A worker pool belongs to a home that is still ticked. Pools were created per home and dropped
    only on ``max_workers`` change or at ``atexit``, so every home ever ticked kept a
    ThreadPoolExecutor and its live ``cron-parallel`` threads for the process's lifetime.
"""

from __future__ import annotations

import cron.scheduler as sched
from hermes_constants import (
    hermes_home_key,
    reset_hermes_home_override,
    set_hermes_home_override,
)


class _Scope:
    def __init__(self, home):
        self._home = str(home)

    def __enter__(self):
        self._token = set_hermes_home_override(self._home)
        return self

    def __exit__(self, *_exc):
        reset_hermes_home_override(self._token)
        return False


def _two_homes(tmp_path):
    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    return home_a, home_b


class TestPerProfileJobLiveness:
    def test_is_job_running_answers_for_the_asking_profile_only(self, tmp_path):
        """(a) A's run must not make B's same-named job report as running."""
        home_a, home_b = _two_homes(tmp_path)
        job_id = "daily-brief"
        try:
            with _Scope(home_a):
                assert sched.try_register_running_job(job_id) is True
                assert sched.is_job_running(job_id) is True
            with _Scope(home_b):
                assert sched.is_job_running(job_id) is False, (
                    "profile B's idle job must not inherit profile A's run"
                )
            # The host-wide union still reports it for the shutdown drain.
            assert job_id in sched.get_running_job_ids()
        finally:
            with _Scope(home_a):
                sched.release_running_job(job_id)

    def test_stale_oneshot_liveness_is_scoped_to_its_own_home(self, tmp_path):
        """The store's stale-entry recovery consumer asks per home, not host-wide."""
        from cron.jobs import _job_running_in_this_process

        home_a, home_b = _two_homes(tmp_path)
        job_id = "daily-brief"
        try:
            with _Scope(home_a):
                assert sched.try_register_running_job(job_id) is True
                assert _job_running_in_this_process(job_id) is True
            with _Scope(home_b):
                assert _job_running_in_this_process(job_id) is False, (
                    "B's stale one-shot would be kept alive forever by A's run"
                )
        finally:
            with _Scope(home_a):
                sched.release_running_job(job_id)


class TestPoolReaping:
    def test_pool_of_an_unticked_home_is_reaped(self, tmp_path):
        """(b) Republishing the ticked set without a home drops that home's pool."""
        from cron.scheduler_ownership import register_ticked_homes

        home_a, home_b = _two_homes(tmp_path)
        try:
            register_ticked_homes([home_a, home_b])
            with _Scope(home_a):
                sched._get_parallel_pool(1)
            with _Scope(home_b):
                sched._get_parallel_pool(1)
            assert len(sched._parallel_pools) == 2

            # Profile B goes away (deleted / gated out); the served set is republished without it.
            register_ticked_homes([home_a])
            assert hermes_home_key(home_b) not in sched._parallel_pools, (
                "an unticked home's ThreadPoolExecutor leaked for the process's lifetime"
            )
            assert hermes_home_key(home_b) not in sched._parallel_pool_max_workers
            # The home still being ticked keeps its pool.
            assert hermes_home_key(home_a) in sched._parallel_pools
        finally:
            register_ticked_homes([])
            sched._shutdown_parallel_pool()

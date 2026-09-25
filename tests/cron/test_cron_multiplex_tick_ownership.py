"""One host process ticks every profile: cron ownership and state must be per profile.

Three invariants, each broken before the multiplex-scope fix:

(a) Two profiles carrying an identically named job, ticked by ONE process, are two jobs: the
    in-flight dedupe guard keyed them by bare job id, so the second profile's run was skipped as
    a duplicate and its release cleared the first profile's claim.
(b) The stale-code yield gate is asked per served profile. It resolved ownership through the
    process-global runtime-lock boolean and the LAUNCH home's lock, so one answer covered every
    profile the ticker visited.
(c) The worker pool is sized by the profile being ticked, not by whichever profile ticked first.
"""

from __future__ import annotations

import cron.scheduler as sched
from hermes_constants import (
    hermes_home_key,
    reset_hermes_home_override,
    set_hermes_home_override,
)


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


def _two_homes(tmp_path):
    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    home_a.mkdir()
    home_b.mkdir()
    return home_a, home_b


class TestPerProfileInflightState:
    def test_same_job_id_in_two_profiles_does_not_collide(self, tmp_path):
        """(a) Both profiles' ``daily-brief`` register, and releasing one leaves the other."""
        home_a, home_b = _two_homes(tmp_path)
        job_id = "daily-brief"
        try:
            with _Scope(home_a):
                assert sched.try_register_running_job(job_id) is True
            with _Scope(home_b):
                assert sched.try_register_running_job(job_id) is True, (
                    "a second profile's same-named job must not read as a duplicate"
                )
                # Within one profile the dedupe guard still holds.
                assert sched.try_register_running_job(job_id) is False
            with _Scope(home_a):
                sched.release_running_job(job_id)
            assert sched._inflight_key(job_id, home_b) in sched._running_job_ids, (
                "profile A's release must not drop profile B's claim"
            )
            # The host-wide accessor keeps reporting bare job ids for the shutdown drain.
            assert job_id in sched.get_running_job_ids()
        finally:
            for home in (home_a, home_b):
                with _Scope(home):
                    sched.release_running_job(job_id)


class TestPerProfileYieldGate:
    def test_yield_gate_is_answered_per_served_profile(self, tmp_path, monkeypatch):
        """(b) A fresher gateway serving only profile A: A yields, B keeps ticking."""
        # Imported per test, not at module scope: a module-level import of a module that does not
        # exist on the base commit turns the whole file into a COLLECTION ERROR, which is not
        # behavioural red-on-base evidence for the invariants below.
        import cron.scheduler_ownership as ownership

        home_a, home_b = _two_homes(tmp_path)
        boot, disk = "a" * 40, "b" * 40
        monkeypatch.setattr(
            sched, "_detect_gateway_code_skew", lambda: (boot[:10], disk[:10]))
        monkeypatch.setattr(sched, "_current_gateway_code_sha", lambda: disk)
        # A live foreign gateway holding the runtime lock, serving profile A's home only.
        from gateway import status as gateway_status

        monkeypatch.setattr(gateway_status, "owns_gateway_runtime_lock", lambda: False)
        monkeypatch.setattr(
            gateway_status, "is_gateway_runtime_lock_active", lambda lock_path=None: True)
        monkeypatch.setattr(gateway_status, "get_running_pid", lambda **_k: 4321)
        monkeypatch.setattr(gateway_status, "runtime_status_is_stale", lambda _r: False)
        monkeypatch.setattr(
            gateway_status,
            "read_runtime_status",
            lambda: {"pid": 4321, "code_sha": disk, "hermes_home": str(home_a)},
        )
        ownership.register_ticked_homes([home_a, home_b])
        try:
            with _Scope(home_a):
                assert sched._should_yield_tick_to_fresh_gateway() is not None
            with _Scope(home_b):
                assert sched._should_yield_tick_to_fresh_gateway() is None, (
                    "profile B has no fresher ticker; yielding would stop its cron entirely"
                )
        finally:
            ownership.register_ticked_homes([])

    def test_host_multiplexer_never_yields_a_home_it_ticks(self, tmp_path, monkeypatch):
        """Owning the runtime lock only excuses the homes this process actually ticks."""
        import cron.scheduler_ownership as ownership

        home_a, home_b = _two_homes(tmp_path)
        boot, disk = "a" * 40, "b" * 40
        monkeypatch.setattr(
            sched, "_detect_gateway_code_skew", lambda: (boot[:10], disk[:10]))
        monkeypatch.setattr(sched, "_current_gateway_code_sha", lambda: disk)
        from gateway import status as gateway_status

        monkeypatch.setattr(gateway_status, "owns_gateway_runtime_lock", lambda: True)
        ownership.register_ticked_homes([home_a])
        try:
            with _Scope(home_a):
                assert ownership.owns_cron_tick_for(home_a) is True
                assert sched._should_yield_tick_to_fresh_gateway() is None
            with _Scope(home_b):
                assert ownership.owns_cron_tick_for(home_b) is False
        finally:
            ownership.register_ticked_homes([])


class TestPerProfileWorkerPool:
    def test_pool_is_sized_by_the_ticked_profile(self, tmp_path):
        """(c) The first profile ticked must not size every other profile's pool."""
        home_a, home_b = _two_homes(tmp_path)
        try:
            with _Scope(home_a):
                pool_a = sched._get_parallel_pool(1)
            with _Scope(home_b):
                pool_b = sched._get_parallel_pool(4)
            assert pool_a is not pool_b
            assert pool_a._max_workers == 1
            assert pool_b._max_workers == 4
            with _Scope(home_a):
                # Profile A keeps its own pool; B's larger budget did not replace it.
                assert sched._get_parallel_pool(1) is pool_a
        finally:
            sched._shutdown_parallel_pool()

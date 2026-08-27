"""Stale-code cron tick yield gate.

A long-lived process whose checkout was updated underneath it (hot git pull /
interrupted ``hermes update``) serves mixed ``sys.modules``; when such a
process races a fresh gateway for the cron tick lock and wins, every agent
job it dispatches can die on ImportErrors whose real cause is staleness.

The gate in ``cron.scheduler.tick`` (before lock acquisition) yields — raises
``CronTickYielded`` — only when ALL of:

1. this process is provably stale (boot fingerprint ≠ disk revision),
2. it does NOT own the gateway runtime lock, and
3. some other process currently holds that lock.

Any probe failure or missing fingerprint means "proceed" (fail-open). Yielded
ticks must surface as failed ticks (``record_ticker_error`` + heartbeat
``success=False``), never as healthy ones.
"""

from __future__ import annotations

import logging
import threading
import time
from unittest.mock import patch

import pytest

import cron.scheduler as scheduler_mod


SKEW = ("boot0123abcd", "disk4567efgh")


def _wait_until(predicate, timeout=10.0, interval=0.005):
    """Block until ``predicate()`` is truthy or ``timeout`` elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


def _gate_mocks(monkeypatch, *, owns: bool, active: bool, skew=SKEW):
    """Point the yield gate's three probes at fixed answers."""
    from gateway import status as gateway_status

    monkeypatch.setattr(scheduler_mod, "_detect_gateway_code_skew", lambda: skew)
    monkeypatch.setattr(gateway_status, "owns_gateway_runtime_lock", lambda: owns)
    monkeypatch.setattr(
        gateway_status, "is_gateway_runtime_lock_active", lambda lock_path=None: active
    )


class TestTickYieldGate:
    def test_skew_plus_foreign_gateway_yields_before_lock(self, monkeypatch, tmp_path):
        """(a) skew + a fresh foreign gateway holds the runtime lock → the
        tick yields WITHOUT acquiring the tick lock or dispatching."""
        _gate_mocks(monkeypatch, owns=False, active=True)

        def _no_lock_allowed(*_a, **_kw):
            raise AssertionError("tick lock must not be acquired after a yield")

        monkeypatch.setattr(scheduler_mod, "_get_lock_paths", _no_lock_allowed)

        with pytest.raises(scheduler_mod.CronTickYielded) as excinfo:
            scheduler_mod.tick(verbose=False)
        # The exception carries the forensics for the ticker-error record.
        assert SKEW[0] in str(excinfo.value)
        assert SKEW[1] in str(excinfo.value)

    def test_skew_plus_self_owned_lock_proceeds(self, monkeypatch, tmp_path):
        """(b) skew + this process IS the gateway → proceed. The delivery
        path's stale-code hint (part 3) stays the surface for this case."""
        _gate_mocks(monkeypatch, owns=True, active=True)
        # Reaching the lock paths proves the gate let the tick proceed.
        called = []
        real = scheduler_mod._get_lock_paths

        def _spy():
            called.append(1)
            return real()

        monkeypatch.setattr(scheduler_mod, "_get_lock_paths", _spy)
        assert scheduler_mod.tick(verbose=False) == 0  # empty store: no jobs
        assert called, "tick must proceed when this process owns the gateway lock"

    def test_skew_plus_no_gateway_proceeds(self, monkeypatch, tmp_path):
        """(c) skew + no gateway alive (desktop-standalone) → proceed:
        yielding would silently kill the user's only ticker."""
        _gate_mocks(monkeypatch, owns=False, active=False)
        assert scheduler_mod.tick(verbose=False) == 0

    def test_no_fingerprint_proceeds(self, monkeypatch, tmp_path):
        """(d) skew is None (non-git install, no boot fingerprint, probe
        failure) → always proceed."""
        _gate_mocks(monkeypatch, owns=False, active=True, skew=None)
        assert scheduler_mod.tick(verbose=False) == 0

    def test_lock_probe_exception_proceeds(self, monkeypatch, tmp_path):
        """Fail-open: the lock probe raising must not yield or crash the
        tick."""
        from gateway import status as gateway_status

        monkeypatch.setattr(scheduler_mod, "_detect_gateway_code_skew", lambda: SKEW)

        def _boom(lock_path=None):
            raise OSError("lock probe failed")

        monkeypatch.setattr(gateway_status, "is_gateway_runtime_lock_active", _boom)
        assert scheduler_mod.tick(verbose=False) == 0

    def test_yield_logs_once_per_episode(self, monkeypatch, caplog):
        """The yield log is throttled: repeated yields with the same skew
        signature log once, not once per tick interval."""
        _gate_mocks(monkeypatch, owns=False, active=True)
        with caplog.at_level(logging.ERROR, logger="cron.scheduler"):
            scheduler_mod._log_tick_yield_once("boot=a disk=b")
            scheduler_mod._log_tick_yield_once("boot=a disk=b")
            scheduler_mod._log_tick_yield_once("boot=a disk=b")
        yield_logs = [
            r for r in caplog.records if "Cron tick yielded" in r.getMessage()
        ]
        assert len(yield_logs) == 1
        # A NEW skew signature (the checkout moved again) is a new episode.
        with caplog.at_level(logging.ERROR, logger="cron.scheduler"):
            scheduler_mod._log_tick_yield_once("boot=a disk=c")
        yield_logs = [
            r for r in caplog.records if "Cron tick yielded" in r.getMessage()
        ]
        assert len(yield_logs) == 2


class TestYieldedTickIsAFailedTick:
    """(e) A yielded tick must not be recorded as a healthy tick."""

    def test_provider_records_error_and_unsuccessful_heartbeat(self):
        from cron.scheduler_provider import InProcessCronScheduler

        beats: list[bool] = []
        errors: list[str] = []
        stop = threading.Event()
        prov = InProcessCronScheduler()

        with patch(
            "cron.scheduler.tick",
            side_effect=scheduler_mod.CronTickYielded(SKEW[0], SKEW[1]),
        ), patch(
            "cron.jobs.record_ticker_heartbeat",
            side_effect=lambda success=False: beats.append(success),
        ), patch(
            "cron.jobs.record_ticker_error",
            side_effect=lambda msg: errors.append(msg),
        ), patch("cron.jobs.clear_ticker_error") as clear:
            t = threading.Thread(
                target=prov.start, args=(stop,), kwargs={"interval": 0}, daemon=True
            )
            t.start()
            assert _wait_until(lambda: len(beats) >= 3), "ticker did not keep beating"
            stop.set()
            t.join(timeout=5)

        assert not t.is_alive(), "ticker must keep yielding, not die"
        assert errors, "yield reason must be persisted for `hermes cron status`"
        assert "CronTickYielded" in errors[0] or "yielded" in errors[0]
        assert beats[-1] is False, "a yielded tick is not a successful tick"
        clear.assert_not_called()

    def test_ticker_takes_over_when_gateway_lock_releases(self):
        """Self-healing: once the fresh gateway dies (lock inactive), the
        yielding ticker's next tick proceeds normally."""
        from cron.scheduler_provider import InProcessCronScheduler

        beats: list[bool] = []
        state = {"foreign_gateway_alive": True}
        stop = threading.Event()
        prov = InProcessCronScheduler()

        def _tick(*args, **kwargs):
            # Gate logic is exercised for real; emulate the two probe arms.
            if state["foreign_gateway_alive"]:
                raise scheduler_mod.CronTickYielded(SKEW[0], SKEW[1])
            return 0

        with patch("cron.scheduler.tick", side_effect=_tick), patch(
            "cron.jobs.record_ticker_heartbeat",
            side_effect=lambda success=False: beats.append(success),
        ), patch("cron.jobs.record_ticker_error"), patch("cron.jobs.clear_ticker_error"):
            t = threading.Thread(
                target=prov.start, args=(stop,), kwargs={"interval": 0}, daemon=True
            )
            t.start()
            assert _wait_until(lambda: len(beats) >= 2)
            state["foreign_gateway_alive"] = False  # gateway lock released
            assert _wait_until(lambda: True in beats), "ticker never took over"
            stop.set()
            t.join(timeout=5)

        assert not t.is_alive()


    def test_multiplex_yield_in_one_profile_does_not_cancel_siblings(self):
        """Multiplex loop: a yield for profile A (its fresh gateway owns the
        runtime lock) must not cancel profile B's tick in the same cycle —
        B may have no other ticker. B's heartbeat stays healthy; A's records
        the yield."""
        from cron.scheduler_provider import InProcessCronScheduler

        home_a = "/tmp/home-a"
        home_b = "/tmp/home-b"
        ticked: list[str] = []
        per_home_beats: dict[str, list[bool]] = {home_a: [], home_b: []}
        per_home_errors: dict[str, list[str]] = {home_a: [], home_b: []}
        state = {"cycles": 0}
        stop = threading.Event()
        prov = InProcessCronScheduler()

        def _tick(*args, **kwargs):
            from hermes_constants import get_hermes_home

            home = str(get_hermes_home())
            ticked.append(home)
            if home == home_a:
                # Profile A: stale process, fresh foreign gateway → yield.
                raise scheduler_mod.CronTickYielded(SKEW[0], SKEW[1])
            # Profile B ticks fine.
            return 0

        def _beat(success=False):
            from hermes_constants import get_hermes_home

            per_home_beats[str(get_hermes_home())].append(success)

        def _err(msg):
            from hermes_constants import get_hermes_home

            per_home_errors[str(get_hermes_home())].append(msg)

        with patch("cron.scheduler.tick", side_effect=_tick), patch(
            "cron.jobs.record_ticker_heartbeat", side_effect=_beat
        ), patch("cron.jobs.record_ticker_error", side_effect=_err), patch(
            "cron.jobs.clear_ticker_error"
        ):
            t = threading.Thread(
                target=prov.start,
                args=(stop,),
                kwargs={
                    "interval": 0,
                    "profile_homes": [home_a, home_b],
                },
                daemon=True,
            )
            t.start()
            assert _wait_until(lambda: len(ticked) >= 6), "multiplex loop did not run"
            stop.set()
            t.join(timeout=5)

        assert not t.is_alive()
        # Every cycle ticked BOTH homes — B was never cancelled by A's yield.
        assert home_b in ticked and home_a in ticked
        # A recorded unsuccessful beats + its yield reason; B stayed healthy.
        # (The first beat per home is the pre-loop initial heartbeat, always
        # unsuccessful — skip it when judging cycle outcomes.)
        assert per_home_beats[home_a], "profile A must still beat (liveness)"
        assert all(b is False for b in per_home_beats[home_a][1:])
        assert per_home_errors[home_a], "profile A must record the yield reason"
        assert per_home_beats[home_b][1:] and all(
            b is True for b in per_home_beats[home_b][1:]
        ), "profile B must stay healthy"
        assert not per_home_errors[home_b], "profile B must not inherit A's yield"


class TestGatewayLockOwnershipProbe:
    """The self-ownership accessor backing the gate, against the real lock."""

    def test_ownership_follows_acquire_and_release(self, tmp_path, monkeypatch):
        from gateway import status as gateway_status

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert gateway_status.owns_gateway_runtime_lock() is False
        assert gateway_status.acquire_gateway_runtime_lock() is True
        try:
            assert gateway_status.owns_gateway_runtime_lock() is True
            # Self-ownership is distinguishable even though the shared
            # liveness probe also reports active for the owner.
            assert gateway_status.is_gateway_runtime_lock_active() is True
        finally:
            gateway_status.release_gateway_runtime_lock()
        assert gateway_status.owns_gateway_runtime_lock() is False

    def test_ownership_probe_is_profile_scoped_by_process_home(
        self, tmp_path, monkeypatch
    ):
        """The gateway lock is a process-level identity file (#56986): under
        a multiplex profile-home override the probe must still resolve the
        launch home's lock, not the overridden profile's."""
        from gateway import status as gateway_status
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        other_profile = tmp_path / "profiles" / "other"
        other_profile.mkdir(parents=True)
        token = set_hermes_home_override(str(other_profile))
        try:
            assert gateway_status.acquire_gateway_runtime_lock() is True
            try:
                assert gateway_status.owns_gateway_runtime_lock() is True
                # The lock file lives in the launch home, not the override.
                assert (tmp_path / "gateway.lock").exists()
                assert not (other_profile / "gateway.lock").exists()
            finally:
                gateway_status.release_gateway_runtime_lock()
        finally:
            reset_hermes_home_override(token)

"""The ticker's own contract — "an exception must not silently kill the daemon thread" —
must hold for every step that runs on the cron-scheduler thread, not just the tick body:
startup recovery, per-cycle profile enumeration/gating, and the status-marker writes.
The gateway starts this thread without a supervisor, so any escaping exception ends cron
silently while the gateway keeps running (#111010).
"""

import threading
import time
from unittest.mock import patch


def _wait_until(predicate, timeout=10.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    return predicate()


def test_ticker_survives_startup_recovery_crash(tmp_path):
    """A store that blows up during startup recovery must not kill the ticker thread:
    the gateway keeps running and cron silently stops firing forever (#111010)."""
    from cron.scheduler_provider import InProcessCronScheduler

    ticks = []

    def fake_tick(*args, **kwargs):
        ticks.append(1)
        return 0

    def broken_recover(self):
        raise RuntimeError("sqlite3.DatabaseError: database disk image is malformed")

    stop = threading.Event()
    provider = InProcessCronScheduler()
    with (
        patch("cron.scheduler.tick", side_effect=fake_tick),
        patch("cron.jobs.record_ticker_heartbeat", lambda **kw: None),
        patch("cron.jobs.record_ticker_error", lambda *a, **kw: None),
        patch.object(InProcessCronScheduler, "recover_interrupted", broken_recover),
    ):
        thread = threading.Thread(
            target=provider.start,
            args=(stop,),
            kwargs={"interval": 0},
            daemon=True,
            name="cron-scheduler",
        )
        thread.start()
        _wait_until(lambda: len(ticks) >= 1)
        alive_after_first_tick = thread.is_alive()
        stop.set()
        thread.join(timeout=5)

    assert alive_after_first_tick, "ticker thread died during startup recovery"
    assert ticks, "ticker never ticked after startup recovery crashed"
    assert not thread.is_alive()


def test_multiplex_ticker_survives_profile_gate_crash(tmp_path):
    """A raising profile_gate callable runs on the ticker thread outside the guarded
    tick body; its failure must degrade to a failed cycle, not a dead thread."""
    from cron.scheduler_provider import InProcessCronScheduler

    home = tmp_path / "default"
    (home / "cron").mkdir(parents=True)

    ticks = []

    def fake_tick(*args, **kwargs):
        ticks.append(1)
        return 0

    def broken_gate(name, home):
        raise RuntimeError("gate boom")

    stop = threading.Event()
    provider = InProcessCronScheduler()
    with (
        patch("cron.scheduler.tick", side_effect=fake_tick),
        patch("cron.jobs.record_ticker_heartbeat", lambda **kw: None),
    ):
        thread = threading.Thread(
            target=provider.start,
            args=(stop,),
            kwargs={
                "interval": 0,
                "profile_homes": [("default", home)],
                "profile_gate": broken_gate,
            },
            daemon=True,
            name="cron-scheduler",
        )
        thread.start()
        time.sleep(0.5)
        alive_after_failed_cycles = thread.is_alive()
        stop.set()
        thread.join(timeout=5)

    assert alive_after_failed_cycles, "ticker thread died when profile_gate raised"
    assert not thread.is_alive()


def test_ticker_survives_marker_write_crash(tmp_path):
    """A misbehaving marker write (SystemExit from the store layer) must not end the
    ticker — the loop body already swallows exactly this class of failure from the
    provider SDKs (#32612); the marker writes deserve the same protection."""
    from cron.scheduler_provider import InProcessCronScheduler

    ticks = []
    heartbeat_calls = {"n": 0}

    def fake_tick(*args, **kwargs):
        ticks.append(1)
        return 0

    def flaky_heartbeat(**kwargs):
        heartbeat_calls["n"] += 1
        if heartbeat_calls["n"] == 2:
            raise SystemExit("misbehaving marker write")

    stop = threading.Event()
    provider = InProcessCronScheduler()
    with (
        patch("cron.scheduler.tick", side_effect=fake_tick),
        patch("cron.jobs.record_ticker_heartbeat", side_effect=flaky_heartbeat),
        patch.object(InProcessCronScheduler, "recover_interrupted", lambda self: 0),
        patch("cron.jobs.record_ticker_error", lambda *a, **kw: None),
    ):
        thread = threading.Thread(
            target=provider.start,
            args=(stop,),
            kwargs={"interval": 0},
            daemon=True,
            name="cron-scheduler",
        )
        thread.start()
        _wait_until(lambda: len(ticks) >= 2)
        alive_past_crash = thread.is_alive()
        stop.set()
        thread.join(timeout=5)

    assert heartbeat_calls["n"] >= 2, "flaky heartbeat write never fired"
    assert alive_past_crash, "ticker thread died on a marker-write SystemExit"
    assert len(ticks) >= 2, "ticker stopped ticking after the marker-write crash"
    assert not thread.is_alive()

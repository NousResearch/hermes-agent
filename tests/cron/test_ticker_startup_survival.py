"""The ticker's contract — "an exception must not silently end the daemon thread" — must hold
for everything that runs on the cron-scheduler thread, not just the tick body: startup
recovery and the status-marker writes included. The gateway starts this thread without a
supervisor, so any escaping exception ends cron silently while the gateway keeps running
(#111010).
"""

import threading
from unittest.mock import patch

# Generous wall-clock bound: the loop runs with interval=0, so a live ticker satisfies this
# within milliseconds; only a dead thread ever hits the timeout.
_BOUND = 5.0


def _run_ticker(provider, stop, ticked):
    """Drive ``provider.start`` on a real daemon thread until ``ticked`` is set (or the bound
    elapses), then return ``(alive_before_stop, thread)``."""
    thread = threading.Thread(
        target=provider.start, args=(stop,), kwargs={"interval": 0},
        daemon=True, name="cron-scheduler",
    )
    thread.start()
    ticked.wait(_BOUND)
    alive = thread.is_alive()
    stop.set()
    thread.join(timeout=_BOUND)
    return alive, thread


def test_ticker_survives_startup_recovery_crash():
    """A store that blows up during startup recovery must not end the ticker thread —
    the loop has to reach its first tick regardless."""
    from cron.scheduler_provider import InProcessCronScheduler

    ticked = threading.Event()

    def fake_tick(*args, **kwargs):
        ticked.set()
        return 0

    def broken_recover(self):
        raise RuntimeError("sqlite3.DatabaseError: database disk image is malformed")

    stop = threading.Event()
    with (
        patch("cron.scheduler.tick", side_effect=fake_tick),
        patch("cron.jobs.record_ticker_heartbeat", lambda **kw: None),
        patch("cron.jobs.record_ticker_error", lambda *a, **kw: None),
        patch.object(InProcessCronScheduler, "recover_interrupted", broken_recover),
    ):
        alive, thread = _run_ticker(InProcessCronScheduler(), stop, ticked)

    assert ticked.is_set(), "ticker never ticked after startup recovery crashed"
    assert alive, "ticker thread died during startup recovery"
    assert not thread.is_alive()


def test_ticker_survives_marker_write_crash():
    """A heartbeat write that raises (SystemExit from the store layer — the class the tick
    body already swallows, #32612) must not end the ticker: the NEXT tick still happens."""
    from cron.scheduler_provider import InProcessCronScheduler

    ticks = 0
    ticked_twice = threading.Event()
    heartbeats = 0

    def fake_tick(*args, **kwargs):
        nonlocal ticks
        ticks += 1
        if ticks >= 2:
            ticked_twice.set()
        return 0

    def flaky_heartbeat(**kwargs):
        nonlocal heartbeats
        heartbeats += 1
        # The 2nd write is the tail of the first loop iteration (the 1st is the startup beat).
        if heartbeats == 2:
            raise SystemExit("misbehaving marker write")

    stop = threading.Event()
    with (
        patch("cron.scheduler.tick", side_effect=fake_tick),
        patch("cron.jobs.record_ticker_heartbeat", side_effect=flaky_heartbeat),
        patch("cron.jobs.record_ticker_error", lambda *a, **kw: None),
        patch.object(InProcessCronScheduler, "recover_interrupted", lambda self: 0),
    ):
        alive, thread = _run_ticker(InProcessCronScheduler(), stop, ticked_twice)

    assert heartbeats >= 2, "flaky heartbeat write never fired"
    assert ticked_twice.is_set(), "ticker stopped ticking after the marker-write crash"
    assert alive, "ticker thread died on a marker-write SystemExit"
    assert not thread.is_alive()

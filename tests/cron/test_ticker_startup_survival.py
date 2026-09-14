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


def test_multiplex_marker_crash_for_one_profile_does_not_starve_the_next(tmp_path):
    """Multiplex: a heartbeat write that raises for profile A must neither end the ticker nor
    skip profile B's beat on that same cycle — otherwise B's heartbeat goes stale while the
    ticker is alive and status blames the wrong profile."""
    from cron.scheduler_provider import InProcessCronScheduler

    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()

    ticks = 0
    ticked_twice = threading.Event()
    beats: list[str] = []

    def fake_tick(*args, **kwargs):
        nonlocal ticks
        ticks += 1
        if ticks >= 4:  # two full cycles over two homes
            ticked_twice.set()
        return 0

    def flaky_heartbeat(**kwargs):
        from hermes_constants import get_hermes_home

        home = get_hermes_home()
        beats.append(str(home))
        if str(home) == str(home_a) and kwargs:  # in-cycle beats only; the startup beat has no kwargs
            raise SystemExit("misbehaving marker write")

    stop = threading.Event()
    thread = threading.Thread(
        target=InProcessCronScheduler().start, args=(stop,),
        kwargs={"interval": 0, "profile_homes": [home_a, home_b]},
        daemon=True, name="cron-scheduler",
    )
    with (
        patch("cron.scheduler.tick", side_effect=fake_tick),
        patch("cron.jobs.record_ticker_heartbeat", side_effect=flaky_heartbeat),
        patch("cron.jobs.record_ticker_error", lambda *a, **kw: None),
        patch("cron.jobs.clear_ticker_error", lambda *a, **kw: None),
        patch.object(InProcessCronScheduler, "recover_interrupted", lambda self: 0),
    ):
        thread.start()
        ticked_twice.wait(_BOUND)
        alive = thread.is_alive()
        stop.set()
        thread.join(timeout=_BOUND)

    assert ticked_twice.is_set(), "multiplex ticker stopped ticking after profile A's marker crash"
    assert alive, "multiplex ticker thread died on a marker-write SystemExit"
    # Startup beats are one per home; every completed cycle must then beat B despite A raising.
    in_cycle_b = beats.count(str(home_b)) - 1
    assert in_cycle_b >= 2, f"profile B's heartbeat was skipped after A's crash: {beats}"
    assert not thread.is_alive()

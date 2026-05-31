"""Tests for the cron ticker supervisor (gateway.run._cron_supervisor_loop)."""

import threading
import time

import pytest

import gateway.run as run


class _FakeWorker:
    """Stands in for the worker thread; the supervisor only calls is_alive()."""

    def __init__(self, *alive):
        self._alive = iter(alive) if alive else iter(())

    def is_alive(self):
        return next(self._alive, False)


def _state(worker):
    return {
        "worker": worker,
        "cron_stop": threading.Event(),
        "adapters": None,
        "loop": None,
        "interval": 60,
    }


def _patch_common(monkeypatch):
    monkeypatch.setattr(run, "write_cron_ticker_status", lambda **kw: None)
    monkeypatch.setattr(run, "SUPERVISOR_CHECK_INTERVAL", 0)
    monkeypatch.setattr(run, "_CRON_BACKOFF_BASE", 0)
    monkeypatch.setattr(run, "_CRON_BACKOFF_CAP", 0)


def test_supervisor_respawns_dead_worker(monkeypatch):
    spawned = {"n": 0}
    sup_stop = threading.Event()

    def factory(*args, **kwargs):
        spawned["n"] += 1
        sup_stop.set()  # one respawn proves the behavior
        return _FakeWorker()

    _patch_common(monkeypatch)
    monkeypatch.setattr(run, "_spawn_cron_worker", factory)
    run._cron_supervisor_loop(sup_stop, _state(_FakeWorker()))
    assert spawned["n"] >= 1


def test_supervisor_never_gives_up(monkeypatch):
    """An always-dead worker is respawned forever — no terminal give-up state."""
    spawned = {"n": 0}
    sup_stop = threading.Event()

    def factory(*args, **kwargs):
        spawned["n"] += 1
        if spawned["n"] >= 3:
            sup_stop.set()
        return _FakeWorker()

    _patch_common(monkeypatch)
    monkeypatch.setattr(run, "_spawn_cron_worker", factory)
    run._cron_supervisor_loop(sup_stop, _state(_FakeWorker()))
    assert spawned["n"] >= 3


def test_supervisor_survives_spawn_failure(monkeypatch):
    """A transient _spawn_cron_worker failure (e.g. thread/fd exhaustion) must
    NOT kill the supervisor — it logs and retries next cycle. Otherwise the
    watchdog itself dies silently, recreating the un-watched dead-ticker bug
    the supervisor exists to prevent."""
    spawned = {"n": 0}
    sup_stop = threading.Event()

    def factory(*args, **kwargs):
        spawned["n"] += 1
        if spawned["n"] == 1:
            raise RuntimeError("can't start new thread")  # resource limit hit
        sup_stop.set()  # the retry succeeded → done
        return _FakeWorker()

    _patch_common(monkeypatch)
    monkeypatch.setattr(run, "_spawn_cron_worker", factory)
    run._cron_supervisor_loop(sup_stop, _state(_FakeWorker()))
    assert spawned["n"] >= 2  # raised once, supervisor survived and retried


def test_supervisor_alerts_but_does_not_respawn_a_hung_worker(monkeypatch):
    """Alive but stale-heartbeat worker is recorded as stalled, never respawned
    (CPython cannot force-kill a thread)."""
    sup_stop = threading.Event()
    stalled = {"n": 0}

    def on_status(**kw):
        if kw.get("state") == "stalled":
            stalled["n"] += 1
            sup_stop.set()

    monkeypatch.setattr(run, "write_cron_ticker_status", on_status)
    monkeypatch.setattr(run, "SUPERVISOR_CHECK_INTERVAL", 0)
    monkeypatch.setattr(run, "HEARTBEAT_STALE_SECONDS", 1)
    monkeypatch.setattr(run, "_cron_last_tick_monotonic", time.monotonic() - 100)
    monkeypatch.setattr(run, "_spawn_cron_worker",
                        lambda *a, **k: pytest.fail("must not respawn a hung worker"))

    run._cron_supervisor_loop(sup_stop, _state(_FakeWorker(True, True)))
    assert stalled["n"] >= 1


def test_supervisor_backoff_escalates_then_caps(monkeypatch):
    """Consecutive deaths grow the respawn delay exponentially, capped."""
    backoffs = []
    sup_stop = threading.Event()
    real_wait = sup_stop.wait

    def tracking_wait(timeout=None):
        if timeout:
            backoffs.append(timeout)
            if len(backoffs) >= 5:
                sup_stop.set()
        return real_wait(0)

    monkeypatch.setattr(run, "write_cron_ticker_status", lambda **kw: None)
    monkeypatch.setattr(run, "SUPERVISOR_CHECK_INTERVAL", 0)
    monkeypatch.setattr(run, "_CRON_BACKOFF_BASE", 1)
    monkeypatch.setattr(run, "_CRON_BACKOFF_CAP", 4)
    monkeypatch.setattr(run, "_spawn_cron_worker", lambda *a, **k: _FakeWorker())
    monkeypatch.setattr(sup_stop, "wait", tracking_wait)

    run._cron_supervisor_loop(sup_stop, _state(_FakeWorker()))
    assert backoffs == [1, 2, 4, 4, 4]


def test_supervisor_shutdown_during_backoff_skips_respawn(monkeypatch):
    """If shutdown arrives while waiting out the backoff, the worker is NOT
    respawned — teardown stays responsive."""
    sup_stop = threading.Event()
    real_wait = sup_stop.wait
    spawned = {"n": 0}

    def tracking_wait(timeout=None):
        if timeout:  # the backoff wait — simulate shutdown arriving mid-wait
            sup_stop.set()
            return True
        return real_wait(0)

    monkeypatch.setattr(run, "write_cron_ticker_status", lambda **kw: None)
    monkeypatch.setattr(run, "SUPERVISOR_CHECK_INTERVAL", 0)
    monkeypatch.setattr(run, "_CRON_BACKOFF_BASE", 5)
    monkeypatch.setattr(run, "_CRON_BACKOFF_CAP", 120)
    monkeypatch.setattr(run, "_spawn_cron_worker",
                        lambda *a, **k: (spawned.__setitem__("n", spawned["n"] + 1), _FakeWorker())[1])
    monkeypatch.setattr(sup_stop, "wait", tracking_wait)

    run._cron_supervisor_loop(sup_stop, _state(_FakeWorker()))
    assert spawned["n"] == 0


def test_supervisor_resets_backoff_after_recovery(monkeypatch):
    """A healthy interval resets the backoff so a transient blip doesn't keep
    the delay permanently inflated."""
    backoffs = []
    sup_stop = threading.Event()
    real_wait = sup_stop.wait
    # First spawn comes back alive-then-dead: it recovers (resetting backoff)
    # before dying again, so its death must use the BASE delay, not an escalated one.
    respawns = iter([_FakeWorker(True, False), _FakeWorker(), _FakeWorker()])

    def tracking_wait(timeout=None):
        if timeout:
            backoffs.append(timeout)
            if len(backoffs) >= 2:
                sup_stop.set()
        return real_wait(0)

    monkeypatch.setattr(run, "write_cron_ticker_status", lambda **kw: None)
    monkeypatch.setattr(run, "SUPERVISOR_CHECK_INTERVAL", 0)
    monkeypatch.setattr(run, "_CRON_BACKOFF_BASE", 1)
    monkeypatch.setattr(run, "_CRON_BACKOFF_CAP", 100)
    monkeypatch.setattr(run, "_cron_last_tick_monotonic", time.monotonic())  # healthy
    monkeypatch.setattr(run, "_spawn_cron_worker", lambda *a, **k: next(respawns))
    monkeypatch.setattr(sup_stop, "wait", tracking_wait)

    run._cron_supervisor_loop(sup_stop, _state(_FakeWorker()))
    assert backoffs[:2] == [1, 1]  # second death back to base → reset happened

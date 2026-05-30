"""Tests for the gateway cron ticker worker loop (gateway.run._start_cron_ticker)."""

import logging
import threading

import gateway.run as run


def test_transient_exception_logs_warning_and_continues(monkeypatch, caplog):
    """A failing tick is logged at WARNING (visible at default level) and the
    loop survives to tick again — the core #32612/#20302 visibility fix."""
    calls = {"n": 0}
    stop = threading.Event()

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        stop.set()

    monkeypatch.setattr(run, "cron_tick", flaky, raising=False)
    with caplog.at_level(logging.WARNING, logger=run.logger.name):
        run._start_cron_ticker(stop, interval=0)

    assert calls["n"] >= 2  # survived the ValueError
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_baseexception_honors_stop_and_does_not_propagate(monkeypatch):
    """A BaseException (e.g. SystemExit) must not escape the worker: when stop
    is set it breaks cleanly, never re-raising into the thread runner."""
    stop = threading.Event()

    def fatal(*args, **kwargs):
        stop.set()
        raise SystemExit(1)

    monkeypatch.setattr(run, "cron_tick", fatal, raising=False)
    run._start_cron_ticker(stop, interval=0)  # returns, does not propagate


def test_heartbeat_stamped_each_tick(monkeypatch):
    """The in-process heartbeat advances once the ticker runs (None until then)."""
    calls = {"n": 0}
    stop = threading.Event()

    def tick(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] >= 2:
            stop.set()

    monkeypatch.setattr(run, "cron_tick", tick, raising=False)
    monkeypatch.setattr(run, "_cron_last_tick_monotonic", 0.0)
    assert run._cron_heartbeat_age() is None  # never stamped yet

    run._start_cron_ticker(stop, interval=0)
    assert run._cron_heartbeat_age() is not None


def test_tick_error_recorded_in_status(monkeypatch):
    """A tick failure surfaces in the persisted cron_ticker block as last_error
    while state stays "running" (the worker is alive, just erroring)."""
    writes = []
    monkeypatch.setattr(run, "write_cron_ticker_status", lambda **kw: writes.append(kw))
    stop = threading.Event()

    def flaky(*args, **kwargs):
        stop.set()
        raise RuntimeError("kaboom")

    monkeypatch.setattr(run, "cron_tick", flaky, raising=False)
    run._start_cron_ticker(stop, interval=0)

    errored = [w for w in writes if w.get("last_error")]
    assert errored and "kaboom" in errored[-1]["last_error"]
    assert errored[-1]["state"] == "running"  # alive, just erroring


def test_recovered_tick_clears_last_error(monkeypatch):
    """A tick that succeeds after a failure must persist last_error=None, so a
    transient fault does not leave a stale error stuck in the status block."""
    writes = []
    monkeypatch.setattr(run, "write_cron_ticker_status", lambda **kw: writes.append(kw))
    calls = {"n": 0}
    stop = threading.Event()

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        stop.set()

    monkeypatch.setattr(run, "cron_tick", flaky, raising=False)
    run._start_cron_ticker(stop, interval=0)

    per_tick = [w for w in writes if "last_error" in w and "interval_seconds" not in w]
    assert per_tick[-1]["last_error"] is None  # cleared on the recovering tick

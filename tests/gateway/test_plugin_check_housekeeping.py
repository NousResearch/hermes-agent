"""Plugin checks are due-gated from startup, not delayed for an hour."""

import threading

from gateway import run
from hermes_cli import plugins_cadence


def test_plugin_cadence_runs_at_startup_and_each_housekeeping_tick(monkeypatch):
    stop = threading.Event()
    calls = []

    def check(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            stop.set()

    # Stop an unfixed loop after its second sleep, without waiting for tick 60.
    sleeps = []

    def wait(timeout=None):
        sleeps.append(timeout)
        if len(sleeps) == 2:
            stop.set()
        return stop.is_set()

    monkeypatch.setattr(stop, "wait", wait)
    monkeypatch.setattr(plugins_cadence, "maybe_run_gateway_check", check)
    monkeypatch.setattr(run, "_housekeeping_deferred_fts_retry", lambda: None)
    monkeypatch.setattr(run, "_housekeeping_memory_trim", lambda: None)
    run._start_gateway_housekeeping(stop)
    assert len(calls) == 2

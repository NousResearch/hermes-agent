"""The ticker contract — "an exception must not silently kill the cron thread" — holds for every
step that runs on the cron-scheduler thread, not only the tick body: startup recovery, the
multiplex per-cycle enumeration/gate, and the status-marker writes. And when a ticker HAS ended,
gateway housekeeping respawns it (#111010).
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


def _run_ticker(provider, stop, **kwargs):
    thread = threading.Thread(target=provider.start, args=(stop,), kwargs={"interval": 0, **kwargs},
                              daemon=True, name="cron-scheduler")
    thread.start()
    return thread


def test_ticker_survives_a_corrupt_ledger_at_startup(tmp_path, monkeypatch):
    """A real corrupt ``executions.db`` (sqlite ``file is not a database``) used to escape
    ``start()`` before the guarded loop and end the thread: gateway up, no job ever fires."""
    from cron.scheduler_provider import InProcessCronScheduler

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "cron").mkdir()
    (tmp_path / "cron" / "executions.db").write_bytes(b"not a sqlite database" * 64)
    ticks = []
    stop = threading.Event()
    with patch("cron.scheduler.tick", side_effect=lambda **kw: ticks.append(1)):
        thread = _run_ticker(InProcessCronScheduler(), stop)
        _wait_until(lambda: len(ticks) >= 2)
        alive = thread.is_alive()
        stop.set()
        thread.join(timeout=5)
    assert alive and len(ticks) >= 2, "ticker died on the pre-loop recovery scan"
    assert (tmp_path / "cron" / "ticker_heartbeat").exists()


def test_multiplex_ticker_survives_a_raising_gate_without_ticking_ungated(tmp_path, caplog):
    """A ``profile_gate`` that raises (Desktop stand-down probe) must fail the CYCLE: thread alive,
    error logged, and — critically — zero ticks, since an unfiltered home list would tick the very
    profiles the gate stands down for (#100489)."""
    from cron.scheduler_provider import InProcessCronScheduler

    home = tmp_path / "default"
    (home / "cron").mkdir(parents=True)
    ticks = []
    stop = threading.Event()

    def broken_gate(name, h):
        raise RuntimeError("gate boom")

    with patch("cron.scheduler.tick", side_effect=lambda **kw: ticks.append(1)), caplog.at_level("ERROR"):
        thread = _run_ticker(InProcessCronScheduler(), stop,
                             profile_homes=[("default", home)], profile_gate=broken_gate)
        _wait_until(lambda: sum("gate boom" in r.getMessage() for r in caplog.records) >= 2)
        alive = thread.is_alive()
        stop.set()
        thread.join(timeout=5)
    assert alive, "ticker thread died when profile_gate raised"
    assert ticks == [], "a raising gate must not tick the ungated home list"


def test_housekeeping_restarts_a_dead_ticker(monkeypatch):
    """The supervisor is the outer layer: a ticker that ended without a stop request is respawned
    on the next housekeeping tick; one that ended BECAUSE of the stop request is not."""
    import gateway.run as gateway_run
    from cron.scheduler_thread import SupervisedTickerThread

    class _OneTick:
        def __init__(self):
            self.waited = False

        def is_set(self):
            return self.waited

        def wait(self, timeout=None):
            self.waited = True
            return True

    starts = []
    stop = threading.Event()

    def dying_ticker(stop_event):
        starts.append(threading.current_thread().name)
        raise RuntimeError("escaped")

    ticker = SupervisedTickerThread(dying_ticker, args=(stop,), stop_event=stop)
    with patch("threading.excepthook", lambda args: None):
        ticker.start()
        _wait_until(lambda: not ticker.is_alive())
        gateway_run._start_gateway_housekeeping(_OneTick(), interval=0, cron_thread=ticker)
        _wait_until(lambda: len(starts) == 2 and not ticker.is_alive())
        assert len(starts) == 2 and ticker.restarts == 1
        stop.set()
        gateway_run._start_gateway_housekeeping(_OneTick(), interval=0, cron_thread=ticker)
        assert len(starts) == 2, "a stopped ticker must not be respawned"

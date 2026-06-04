"""Tests for the cron-ticker drain guard (issue #37858).

When ``hermes gateway stop`` initiates shutdown, the gateway sets
``runner._draining = True`` before the drain loop runs.  If the cron ticker
background thread fires a tick at that exact moment it can launch a new
outbound agent API call — making an LLM request and (on cron-deliver jobs)
sending a platform message — after the operator has already asked the gateway
to stop.

The fix passes the ``runner`` reference into ``_start_cron_ticker`` so the
loop can skip ``cron_tick()`` calls whenever ``runner._draining`` is True.
"""

import threading
import time
from unittest.mock import MagicMock, patch


from gateway.run import _start_cron_ticker


class _FakeRunner:
    """Minimal stand-in for GatewayRunner that exposes only the _draining flag."""

    def __init__(self, *, draining: bool = False):
        self._draining = draining


def test_cron_ticker_skips_tick_when_runner_is_draining():
    """While runner._draining is True the ticker must NOT call cron_tick()."""
    tick_calls = []

    stop_event = threading.Event()
    runner = _FakeRunner(draining=True)

    # Patch cron_tick so we can count invocations without running real jobs.
    with patch("cron.scheduler.tick", side_effect=lambda **kw: tick_calls.append(1)) as _mock_tick:
        # Run the ticker in a background thread for a couple of intervals.
        thread = threading.Thread(
            target=_start_cron_ticker,
            args=(stop_event,),
            kwargs={"interval": 0, "runner": runner},
            daemon=True,
        )
        thread.start()
        time.sleep(0.15)  # enough for several zero-interval ticks
        stop_event.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive(), "Ticker thread did not exit cleanly"
    assert tick_calls == [], (
        f"cron_tick() was called {len(tick_calls)} time(s) while runner._draining=True — "
        "outbound agent calls must not fire after stop is initiated (#37858)"
    )


def test_cron_ticker_runs_tick_when_runner_is_not_draining():
    """Normal operation: cron_tick() fires when the gateway is NOT draining."""
    tick_calls = []

    stop_event = threading.Event()
    runner = _FakeRunner(draining=False)

    with patch("cron.scheduler.tick", side_effect=lambda **kw: tick_calls.append(1)):
        thread = threading.Thread(
            target=_start_cron_ticker,
            args=(stop_event,),
            kwargs={"interval": 0, "runner": runner},
            daemon=True,
        )
        thread.start()
        time.sleep(0.15)
        stop_event.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive(), "Ticker thread did not exit cleanly"
    assert tick_calls, (
        "cron_tick() was never called when runner._draining=False — "
        "normal tick operation is broken"
    )


def test_cron_ticker_skips_tick_without_runner():
    """When runner=None (legacy call sites), the ticker must still call cron_tick()
    unchanged — the drain guard is a no-op when no runner reference is provided."""
    tick_calls = []

    stop_event = threading.Event()

    with patch("cron.scheduler.tick", side_effect=lambda **kw: tick_calls.append(1)):
        thread = threading.Thread(
            target=_start_cron_ticker,
            args=(stop_event,),
            kwargs={"interval": 0, "runner": None},
            daemon=True,
        )
        thread.start()
        time.sleep(0.15)
        stop_event.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert tick_calls, (
        "cron_tick() was never called when runner=None — "
        "backwards-compat with callers that omit runner is broken"
    )


def test_cron_ticker_resumes_after_drain_clears():
    """Once runner._draining reverts to False, ticks should resume normally.

    This covers the case where the gateway runner temporarily sets _draining
    during a restart then clears it (edge-case drain flag lifecycle).
    """
    tick_calls = []
    stop_event = threading.Event()
    runner = _FakeRunner(draining=True)

    with patch("cron.scheduler.tick", side_effect=lambda **kw: tick_calls.append(1)):
        thread = threading.Thread(
            target=_start_cron_ticker,
            args=(stop_event,),
            kwargs={"interval": 0, "runner": runner},
            daemon=True,
        )
        thread.start()
        time.sleep(0.1)
        # Simulate drain completing (e.g. the runner resets the flag internally)
        runner._draining = False
        time.sleep(0.15)
        stop_event.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert tick_calls, (
        "cron_tick() should fire once _draining is cleared, but never did"
    )

"""Regression tests for the gateway cron-ticker watchdog (#32612).

Before this fix the ticker thread silently died if ``cron.scheduler.tick``
raised: the error was logged at DEBUG (invisible at the default INFO
level) and a ``BaseException`` would tear the thread down entirely. The
operator had no way of knowing because ``hermes cron status`` only
checked whether the gateway *process* was alive, not whether the ticker
thread was still ticking.

The fix:
- Promotes per-tick exception logging to WARNING with a traceback.
- Wraps the tick body in an outer ``except BaseException`` so a transient
  fault cannot permanently kill the thread.
- Writes a heartbeat after every tick.
- ``hermes cron status`` reads that heartbeat and reports the ticker as
  dead when it stops being refreshed.
"""

import logging
import threading
import time

import pytest


def _drive_ticker_until(*, tick_impl, monkeypatch, caplog,
                        min_calls=2, max_seconds=2.0):
    """Run ``_start_cron_ticker`` for a couple of iterations with stubs.

    The real ticker calls ``cron.scheduler.tick`` plus a handful of
    housekeeping helpers (image-cache cleanup, paste sweep, curator). The
    test only cares about exception recovery + heartbeat semantics, so we
    monkeypatch the heavy collaborators with no-ops and inject ``tick_impl``
    as the tick body.
    """
    # Import lazily; gateway/run.py is a giant module and we want test
    # collection to stay cheap.
    from gateway import run as gateway_run

    # Replace ``cron.scheduler.tick`` (the symbol the ticker imports at
    # call time) and the cheap housekeeping helpers so we exercise only the
    # exception-handling and heartbeat paths.
    monkeypatch.setattr("cron.scheduler.tick", tick_impl)
    monkeypatch.setattr("gateway.platforms.base.cleanup_image_cache",
                        lambda *a, **kw: 0)
    monkeypatch.setattr("gateway.platforms.base.cleanup_document_cache",
                        lambda *a, **kw: 0)
    monkeypatch.setattr("hermes_cli.debug._sweep_expired_pastes",
                        lambda *a, **kw: (0, 0))

    stop_event = threading.Event()

    def runner():
        # interval=0 so the loop iterates as fast as the wait/stop_event
        # permits, instead of sleeping 60 seconds between ticks.
        gateway_run._start_cron_ticker(stop_event, interval=0)

    thread = threading.Thread(target=runner, daemon=True)
    with caplog.at_level(logging.DEBUG, logger=gateway_run.logger.name):
        thread.start()
        deadline = time.monotonic() + max_seconds
        while time.monotonic() < deadline:
            if tick_impl.call_count >= min_calls:
                break
            time.sleep(0.02)
        stop_event.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive(), "cron ticker thread did not stop"
    return tick_impl


class _CountingTick:
    """Callable that records its call count and optionally raises."""

    def __init__(self, raises=None, raise_on=()):
        self._raises = raises
        self._raise_on = set(raise_on)
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        if self._raises is not None and self.call_count in self._raise_on:
            raise self._raises


class TestTickerSurvivesExceptions:
    def test_transient_exception_logged_at_warning_and_loop_continues(
        self, monkeypatch, caplog
    ):
        """A regular ``Exception`` inside ``cron_tick`` must NOT kill the
        ticker. Before the fix it was logged at DEBUG (invisible by
        default) and any repeat of the same failure was effectively
        silent. The fix promotes it to WARNING + ``exc_info``."""
        tick = _CountingTick(
            raises=RuntimeError("boom"),
            raise_on={1},  # first call raises, subsequent calls succeed
        )

        _drive_ticker_until(
            tick_impl=tick, monkeypatch=monkeypatch, caplog=caplog,
        )

        assert tick.call_count >= 2, "ticker did not retry after exception"

        warning_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "Cron tick error" in r.message
        ]
        assert warning_records, (
            "Expected WARNING-level 'Cron tick error' record; got: "
            f"{[(r.levelname, r.message) for r in caplog.records]}"
        )

    def test_base_exception_caught_and_logged_and_ticker_recovers(
        self, monkeypatch, caplog
    ):
        """A ``BaseException`` (e.g. ``SystemExit`` smuggled out by a C
        extension) historically killed the ticker thread silently. The
        outer ``except BaseException`` must log it at ERROR and let the
        loop continue."""
        tick = _CountingTick(
            raises=SystemExit("hostile c-extension"),
            raise_on={1},
        )

        _drive_ticker_until(
            tick_impl=tick, monkeypatch=monkeypatch, caplog=caplog,
        )

        assert tick.call_count >= 2, (
            "ticker died on BaseException; silent-failure regression #32612"
        )

        error_records = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR
            and "Cron ticker recovered from fatal error" in r.message
        ]
        assert error_records, (
            "Expected ERROR-level recovery log for BaseException; got: "
            f"{[(r.levelname, r.message) for r in caplog.records]}"
        )


class TestHeartbeatVisibility:
    def test_ticker_writes_heartbeat_each_tick(self, monkeypatch, caplog):
        """The heartbeat file must be refreshed after every tick; that is
        the signal ``hermes cron status`` reads to tell the ticker thread
        apart from the gateway process."""
        tick = _CountingTick()
        _drive_ticker_until(
            tick_impl=tick, monkeypatch=monkeypatch, caplog=caplog,
        )

        from gateway.status import read_cron_ticker_heartbeat

        beat = read_cron_ticker_heartbeat()
        assert beat is not None, "ticker never wrote a heartbeat"
        assert "last_tick_at" in beat

    def test_cron_status_reports_ticker_dead_when_heartbeat_stale(
        self, monkeypatch, capsys
    ):
        """``hermes cron status`` must NOT report ``cron jobs will fire
        automatically`` when the heartbeat is stale, even if a gateway
        process is alive. This is the misleading-status bug from #32612."""
        import os

        from gateway.status import _get_cron_ticker_heartbeat_path
        from hermes_cli import cron as cron_cli

        # Fake "a gateway process is running" so the only signal of
        # trouble is the stale heartbeat.
        monkeypatch.setattr(
            "hermes_cli.gateway.find_gateway_pids", lambda: [os.getpid()]
        )

        path = _get_cron_ticker_heartbeat_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"last_tick_at": "2000-01-01T00:00:00+00:00"}')

        cron_cli.cron_status()
        out = capsys.readouterr().out

        assert "cron ticker is NOT alive" in out, (
            f"status did not warn about dead ticker; got:\n{out}"
        )
        assert "cron jobs will fire automatically" not in out, (
            f"status falsely claimed cron jobs would fire; got:\n{out}"
        )

    def test_cron_status_happy_path_when_heartbeat_fresh(
        self, monkeypatch, capsys
    ):
        """When the heartbeat is fresh the status command should keep its
        existing green message; no regression for the healthy case."""
        import os
        from datetime import datetime, timezone

        from gateway.status import _get_cron_ticker_heartbeat_path
        from hermes_cli import cron as cron_cli

        monkeypatch.setattr(
            "hermes_cli.gateway.find_gateway_pids", lambda: [os.getpid()]
        )

        path = _get_cron_ticker_heartbeat_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f'{{"last_tick_at": "{datetime.now(timezone.utc).isoformat()}"}}'
        )

        cron_cli.cron_status()
        out = capsys.readouterr().out

        assert "cron jobs will fire automatically" in out
        assert "cron ticker is NOT alive" not in out

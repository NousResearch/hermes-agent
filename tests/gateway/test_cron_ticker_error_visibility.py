"""Tests for cron-ticker error visibility in gateway/run.py.

Covers #32612: the inner ``cron_tick`` exception handler used to log at DEBUG
(invisible at default INFO level) and only matched ``Exception``, so any
BaseException-subclass error killed the ticker thread silently. The thread
stayed dead for 15+ hours while ``hermes cron status`` reported healthy.
"""

import logging
import threading
from unittest.mock import patch

import pytest


def _run_one_tick(monkeypatch, side_effect):
    """Drive a single iteration of ``_start_cron_ticker`` with a mocked tick.

    Returns the LogCapture handler attached to ``gateway.run``'s logger. The
    test sets ``stop_event`` from inside the mocked ``cron_tick`` so the loop
    exits after exactly one iteration.
    """
    from gateway import run as gw_run

    stop_event = threading.Event()
    seen = {"count": 0}

    def _fake_tick(*args, **kwargs):
        seen["count"] += 1
        stop_event.set()
        raise side_effect

    monkeypatch.setattr("cron.scheduler.tick", _fake_tick)

    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=logging.DEBUG)
    gw_run.logger.addHandler(handler)
    prior_level = gw_run.logger.level
    gw_run.logger.setLevel(logging.DEBUG)
    try:
        # interval=0 so the post-tick stop_event.wait returns immediately
        try:
            gw_run._start_cron_ticker(stop_event, interval=0)
        except BaseException as exc:  # noqa: BLE001 — the BaseException case re-raises
            if isinstance(exc, Exception):
                raise
            return records, exc
        return records, None
    finally:
        gw_run.logger.removeHandler(handler)
        gw_run.logger.setLevel(prior_level)
        assert seen["count"] >= 1, "fake cron_tick was never called"


class TestCronTickerErrorVisibility:
    def test_exception_logged_at_warning_with_traceback(self, monkeypatch):
        records, raised = _run_one_tick(monkeypatch, RuntimeError("boom"))
        assert raised is None
        tick_logs = [r for r in records if "Cron tick error" in r.getMessage()]
        assert tick_logs, "Cron tick error was never logged"
        rec = tick_logs[0]
        assert rec.levelno == logging.WARNING, (
            "tick errors must be logged at WARNING so they surface at default INFO"
        )
        assert rec.exc_info is not None, (
            "tick errors must include exc_info=True so the traceback is visible"
        )

    def test_base_exception_logged_at_error_and_reraised(self, monkeypatch):
        records, raised = _run_one_tick(monkeypatch, SystemExit("died"))
        assert isinstance(raised, SystemExit), (
            "BaseException-subclass errors must be re-raised so the thread "
            "exits as Python intends"
        )
        fatal_logs = [
            r for r in records if "Cron ticker fatal error" in r.getMessage()
        ]
        assert fatal_logs, "BaseException path must log before re-raising"
        rec = fatal_logs[0]
        assert rec.levelno == logging.ERROR
        assert rec.exc_info is not None

    def test_keyboard_interrupt_also_logged_and_reraised(self, monkeypatch):
        records, raised = _run_one_tick(monkeypatch, KeyboardInterrupt())
        assert isinstance(raised, KeyboardInterrupt)
        fatal_logs = [
            r for r in records if "Cron ticker fatal error" in r.getMessage()
        ]
        assert fatal_logs
        assert fatal_logs[0].levelno == logging.ERROR

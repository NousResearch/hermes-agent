"""Regression tests for gateway cron ticker liveness."""

import logging
import threading

from gateway.run import _start_cron_ticker


def test_cron_ticker_logs_tick_exception_at_error(monkeypatch, caplog):
    import cron.scheduler

    stop = threading.Event()
    calls = {"count": 0}

    def fake_tick(*args, **kwargs):
        calls["count"] += 1
        stop.set()
        raise RuntimeError("boom")

    monkeypatch.setattr(cron.scheduler, "tick", fake_tick)

    with caplog.at_level(logging.ERROR):
        _start_cron_ticker(stop, interval=0)

    assert calls["count"] == 1
    assert any("Cron tick error" in record.message for record in caplog.records)
    assert any(record.levelno >= logging.ERROR for record in caplog.records)


def test_cron_ticker_catches_baseexception_logs_error_and_continues(monkeypatch, caplog):
    import cron.scheduler

    stop = threading.Event()
    calls = {"count": 0}

    def fake_tick(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise SystemExit("simulated fatal cron tick failure")
        stop.set()
        return 0

    monkeypatch.setattr(cron.scheduler, "tick", fake_tick)

    with caplog.at_level(logging.ERROR):
        _start_cron_ticker(stop, interval=0)

    assert calls["count"] == 2
    assert any("Cron tick error" in record.message for record in caplog.records)
    assert any(record.levelno >= logging.ERROR for record in caplog.records)

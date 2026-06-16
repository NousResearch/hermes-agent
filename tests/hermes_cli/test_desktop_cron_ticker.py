import logging
from typing import Any, cast

import cron.scheduler as cron_scheduler
import hermes_cli.web_server as web_server


class OneTickStopEvent:
    """Stop after one ticker iteration."""

    def __init__(self):
        self.waits = []

    def is_set(self):
        return bool(self.waits)

    def wait(self, interval):
        self.waits.append(interval)
        return True


def test_desktop_cron_ticker_stands_down_when_gateway_owns_cron(monkeypatch, caplog):
    tick_calls = []
    monkeypatch.setattr(web_server, "get_running_pid", lambda cleanup_stale=True: 4242)
    monkeypatch.setattr(cron_scheduler, "tick", lambda **kwargs: tick_calls.append(kwargs))

    caplog.set_level(logging.INFO, logger="hermes_cli.web_server")
    stop_event = OneTickStopEvent()

    web_server._start_desktop_cron_ticker(cast(Any, stop_event), interval=0)

    assert tick_calls == []
    assert stop_event.waits == [0]
    assert "Desktop cron ticker standing down; gateway PID 4242 owns cron" in caplog.text


def test_desktop_cron_ticker_runs_when_no_gateway_owns_cron(monkeypatch):
    tick_calls = []
    monkeypatch.setattr(web_server, "get_running_pid", lambda cleanup_stale=True: None)
    monkeypatch.setattr(
        cron_scheduler,
        "tick",
        lambda *, verbose, sync: tick_calls.append({"verbose": verbose, "sync": sync}),
    )
    stop_event = OneTickStopEvent()

    web_server._start_desktop_cron_ticker(cast(Any, stop_event), interval=0)

    assert tick_calls == [{"verbose": False, "sync": False}]
    assert stop_event.waits == [0]

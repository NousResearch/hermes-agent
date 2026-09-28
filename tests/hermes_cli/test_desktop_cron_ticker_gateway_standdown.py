"""Desktop cron ticker stands down while a live gateway owns cron on the same HERMES_HOME (#52202).

The ticker's per-tick ``profile_gate`` only arms in the multiplex path; the fail-open
paths (profile enumeration failure, empty served set, external provider) start an
ungated single-store ticker that races a live gateway on the same HERMES_HOME. The
fix stands down while a gateway is live on this home — but as a re-probing wait, not
a one-shot exit, so the backend still takes over the tick once that gateway dies (#126822).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest


@pytest.fixture()
def ticker_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME plus a seam recording whether the provider started."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    started = {}

    class _Provider:
        name = "builtin"

        def start(self, stop_event, **kwargs):
            started["kwargs"] = kwargs

    import cron.scheduler_provider as sp

    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: _Provider())
    return tmp_path, started


def _set_gateway_running(monkeypatch, running: bool) -> None:
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "_check_gateway_running", lambda home: running)


def test_ticker_stands_down_while_gateway_owns_cron(ticker_env, monkeypatch, caplog):
    from hermes_cli import web_server

    home, started = ticker_env
    _set_gateway_running(monkeypatch, True)

    stop = threading.Event()
    timer = threading.Timer(0.1, stop.set)
    timer.start()
    try:
        with caplog.at_level(logging.INFO, logger="hermes_cli.web_server"):
            # The gateway stays live for the whole window, so this call only returns
            # once the backend's stop_event fires: the wait must never self-deadlock.
            web_server._start_desktop_cron_ticker(stop, interval=0.05)
    finally:
        timer.cancel()

    assert started == {}  # provider.start never called
    assert "standing down" in caplog.text


def test_ticker_takes_over_after_the_gateway_dies(ticker_env, monkeypatch):
    from hermes_cli import web_server

    home, started = ticker_env

    # First probe (startup) sees the live gateway; the next one, after one wait
    # interval, sees it gone — the backend must then start the gated ticker.
    probes = []
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(
        profiles, "_check_gateway_running", lambda home: probes.append(1) or len(probes) < 2
    )

    web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    assert "kwargs" in started  # failover reached, no permanent zombie stand-down


def test_ticker_starts_when_no_gateway(ticker_env, monkeypatch):
    from hermes_cli import web_server

    home, started = ticker_env
    _set_gateway_running(monkeypatch, False)

    web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    assert "kwargs" in started  # provider started as before


def test_ticker_fails_open_when_ownership_probe_raises(ticker_env, monkeypatch, caplog):
    from hermes_cli import web_server

    home, started = ticker_env

    import hermes_cli.profiles as profiles

    def _boom(home):
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(profiles, "_check_gateway_running", _boom)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    assert "kwargs" in started  # per-tick gating fallback, not a silent stand-down
    assert "gateway-ownership probe failed" in caplog.text

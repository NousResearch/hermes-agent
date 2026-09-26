"""Desktop cron ticker stands down when a live gateway owns cron on the same HERMES_HOME (#52202).

The ticker's per-tick ``profile_gate`` only arms in the multiplex path; the fail-open
paths (profile enumeration failure, empty served set, external provider) start an
ungated single-store ticker that races a live gateway on the same HERMES_HOME. The
fix bails out before resolving the provider when a gateway is live on this home.
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


def test_ticker_stands_down_when_gateway_owns_cron(ticker_env, monkeypatch, caplog):
    from hermes_cli import web_server

    home, started = ticker_env
    _set_gateway_running(monkeypatch, True)

    with caplog.at_level(logging.INFO, logger="hermes_cli.web_server"):
        web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    assert started == {}  # provider.start never called
    assert "live gateway owns cron" in caplog.text


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

"""Desktop cron ticker yields to a live gateway that owns cron on the same HERMES_HOME (#52202).

The multiplex path gates each tick through ``profile_gate``. The built-in fail-open paths
(profile enumeration failure, empty served set) gate each tick through ``can_dispatch`` on
the same ownership probe, and an external provider defers its start until the gateway is
gone. Every path takes over once that gateway stops (#126822).
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
    """An external provider has no per-tick gate: its start waits out the live gateway, and
    happens once that gateway is gone instead of never (#126822)."""
    from hermes_cli import web_server

    home, started = ticker_env
    _set_gateway_running(monkeypatch, True)
    stopped = threading.Event()
    stopped.set()

    with caplog.at_level(logging.INFO, logger="hermes_cli.web_server"):
        web_server._start_desktop_cron_ticker(stopped, interval=0)

    assert started == {}  # backend shut down while the gateway still owned cron
    assert "live gateway owns cron" in caplog.text

    import hermes_cli.profiles as profiles

    probes = iter([True, True, False])
    monkeypatch.setattr(profiles, "_check_gateway_running", lambda _home: next(probes))

    web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    assert "kwargs" in started  # the gateway stopped: Desktop takes over


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


def test_gated_ticker_resumes_after_the_gateway_stops(ticker_env, monkeypatch):
    """A gateway live at backend start must not silence Desktop cron for good: the multiplex
    ticker still starts, and its per-tick gate stands down only while that gateway runs."""
    import cron.scheduler_provider as sp
    import hermes_cli.profiles as profiles
    import hermes_logging
    from hermes_cli import web_server

    home, started = ticker_env

    class _InProcess(sp.InProcessCronScheduler):
        def start(self, stop_event, **kwargs):
            started["kwargs"] = kwargs

    gateway = {"running": True}
    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: _InProcess())
    monkeypatch.setattr(profiles, "profiles_to_serve", lambda **_kw: [("default", home)])
    monkeypatch.setattr(profiles, "_check_gateway_running", lambda _home: gateway["running"])
    monkeypatch.setattr(hermes_logging, "enable_profile_log_routing", lambda _homes: None)

    web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    gate = started["kwargs"]["profile_gate"]
    assert gate("default", home) is False  # the live gateway ticks with its adapters
    gateway["running"] = False
    assert gate("default", home) is True  # it stopped: Desktop cron fires again


def test_fail_open_ticker_gates_every_tick_on_the_gateway(ticker_env, monkeypatch):
    """With no profile gate (enumeration failed), the single-store ticker still starts and stands
    down per tick only while a gateway owns this home, including one that comes back."""
    import cron.scheduler_provider as sp
    import hermes_cli.profiles as profiles
    from hermes_cli import web_server

    home, started = ticker_env

    class _InProcess(sp.InProcessCronScheduler):
        def start(self, stop_event, **kwargs):
            started["kwargs"] = kwargs

    def _enumeration_fails(**_kw):
        raise RuntimeError("enumeration failed")

    gateway = {"running": True}
    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: _InProcess())
    monkeypatch.setattr(profiles, "profiles_to_serve", _enumeration_fails)
    monkeypatch.setattr(profiles, "_check_gateway_running", lambda _home: gateway["running"])

    web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    can_dispatch = started["kwargs"]["can_dispatch"]
    assert can_dispatch() is False  # the live gateway ticks with its adapters
    gateway["running"] = False
    assert can_dispatch() is True  # it stopped: Desktop cron fires again
    gateway["running"] = True
    assert can_dispatch() is False  # a returning gateway is not raced

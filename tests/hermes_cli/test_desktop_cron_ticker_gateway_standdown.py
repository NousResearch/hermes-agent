"""Desktop cron ticker yields to a live gateway that owns cron on the same HERMES_HOME (#52202).

Every built-in path (multiplex, and the fail-open single-profile fallback) gates each tick
through ``profile_gate``; an external provider defers its start until the gateway is gone.
Every path takes over once that gateway stops (#126822).
"""

from __future__ import annotations

import logging
import threading

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


@pytest.fixture()
def gateway(monkeypatch):
    """Mutable gateway liveness; ``probes`` counts ownership checks."""
    from hermes_cli import profiles

    state = {"running": True, "probes": 0}

    def _probe(_home):
        state["probes"] += 1
        return state["running"]

    monkeypatch.setattr(profiles, "_check_gateway_running", _probe)
    return state


def _enumeration_fails(**_kw):
    raise RuntimeError("enumeration failed")


def test_external_provider_waits_while_gateway_owns_cron(ticker_env, gateway, caplog):
    """An external provider has no per-tick gate: it is not started while the gateway owns cron."""
    from hermes_cli import web_server

    _home, started = ticker_env
    stopped = threading.Event()
    stopped.set()

    with caplog.at_level(logging.INFO, logger="hermes_cli.web_server"):
        web_server._start_desktop_cron_ticker(stopped, interval=0)

    assert started == {}  # backend shut down while the gateway still owned cron
    assert gateway["probes"] == 1
    assert "live gateway owns cron" in caplog.text


def test_external_provider_starts_once_the_gateway_is_gone(ticker_env, gateway, monkeypatch):
    """The deferred start happens once that gateway stops instead of never (#126822), with one
    ownership probe per interval."""
    from hermes_cli import web_server

    _home, started = ticker_env
    stop = threading.Event()

    def _interval_elapses(_timeout):
        gateway["running"] = False  # the gateway stops during the wait
        return False

    monkeypatch.setattr(stop, "wait", _interval_elapses)

    web_server._start_desktop_cron_ticker(stop, interval=0)

    assert "kwargs" in started
    assert gateway["probes"] == 2  # startup probe + one re-probe after the interval


def test_ticker_starts_when_no_gateway(ticker_env, gateway):
    from hermes_cli import web_server

    _home, started = ticker_env
    gateway["running"] = False

    web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    assert "kwargs" in started  # provider started as before


def test_ticker_fails_open_when_ownership_probe_raises(ticker_env, monkeypatch, caplog):
    from hermes_cli import web_server

    _home, started = ticker_env

    from hermes_cli import profiles

    def _boom(home):
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(profiles, "_check_gateway_running", _boom)

    with caplog.at_level(logging.WARNING, logger="hermes_cli.web_server"):
        web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    assert "kwargs" in started  # not a silent stand-down
    assert "gateway-ownership probe failed" in caplog.text


def test_gated_ticker_resumes_after_the_gateway_stops(ticker_env, gateway, monkeypatch):
    """A gateway live at backend start must not silence Desktop cron for good: the multiplex
    ticker still starts, and its per-tick gate stands down only while that gateway runs."""
    import cron.scheduler_provider as sp
    from hermes_cli import profiles
    import hermes_logging
    from hermes_cli import web_server

    home, started = ticker_env

    class _InProcess(sp.InProcessCronScheduler):
        def start(self, stop_event, **kwargs):
            started["kwargs"] = kwargs

    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: _InProcess())
    monkeypatch.setattr("gateway.profile_serving.profiles_to_serve", lambda **_kw: [("default", home)])
    monkeypatch.setattr(hermes_logging, "enable_profile_log_routing", lambda _homes: None)

    web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    gate = started["kwargs"]["profile_gate"]
    assert gate("default", home) is False  # the live gateway ticks with its adapters
    gateway["running"] = False
    assert gate("default", home) is True  # it stopped: Desktop cron fires again


def test_fail_open_ticker_uses_the_same_profile_gate(ticker_env, gateway, monkeypatch):
    """With profile enumeration broken, the built-in ticker still ticks this backend's own store
    through ``profile_gate``, standing down per tick only while a gateway owns it, including one
    that comes back."""
    import cron.scheduler_provider as sp
    from hermes_cli import profiles
    from hermes_cli import web_server

    home, started = ticker_env

    class _InProcess(sp.InProcessCronScheduler):
        def start(self, stop_event, **kwargs):
            started["kwargs"] = kwargs

    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: _InProcess())
    monkeypatch.setattr("gateway.profile_serving.profiles_to_serve", _enumeration_fails)

    web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    kwargs = started["kwargs"]
    assert set(kwargs) == {"interval", "profile_homes", "profile_gate"}
    [(name, own_home)] = kwargs["profile_homes"]()
    assert own_home == home
    gate = kwargs["profile_gate"]
    assert gate(name, own_home) is False  # the live gateway ticks with its adapters
    gateway["running"] = False
    assert gate(name, own_home) is True  # it stopped: Desktop cron fires again
    gateway["running"] = True
    assert gate(name, own_home) is False  # a returning gateway is not raced


def test_fail_open_ticker_yields_to_the_multiplexer_serving_this_profile(tmp_path, monkeypatch):
    """A named profile served by the live default multiplexer has no gateway.pid of its own; the
    fail-open gate must still stand down for it, as the multiplex gate does."""
    import cron.scheduler_provider as sp
    from hermes_cli import profiles
    import hermes_constants
    from hermes_cli import web_server

    satellite = tmp_path / "profiles" / "worker"
    satellite.mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path)
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: satellite)
    started = {}

    class _InProcess(sp.InProcessCronScheduler):
        def start(self, stop_event, **kwargs):
            started["kwargs"] = kwargs

    multiplexer = {"serves": True}
    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: _InProcess())
    monkeypatch.setattr("gateway.profile_serving.profiles_to_serve", _enumeration_fails)
    monkeypatch.setattr(profiles, "_check_gateway_running", lambda _home: False)
    monkeypatch.setattr(
        profiles, "_served_by_running_multiplexer", lambda name: multiplexer["serves"] and name == "worker")

    web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    [(name, home)] = started["kwargs"]["profile_homes"]()
    assert (name, home) == ("worker", satellite)
    gate = started["kwargs"]["profile_gate"]
    assert gate(name, home) is False
    multiplexer["serves"] = False
    assert gate(name, home) is True


def test_gated_out_fail_open_tick_leaves_the_gateway_store_status_alone(tmp_path, monkeypatch):
    """Through the real built-in loop: while the gateway owns the store, the fail-open ticker
    neither ticks nor records a successful tick or clears the gateway's recorded tick error
    (#32612, #32895)."""
    from cron import jobs
    from hermes_cli import profiles
    import hermes_constants
    from cron.scheduler_provider import InProcessCronScheduler
    from hermes_cli import web_server

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("gateway.profile_serving.profiles_to_serve", _enumeration_fails)
    monkeypatch.setattr(profiles, "_check_gateway_running", lambda _home: True)
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", InProcessCronScheduler)
    ticked, beats, cleared = [], [], []
    monkeypatch.setattr("cron.scheduler.tick", lambda **_kw: ticked.append(True))
    monkeypatch.setattr(jobs, "record_ticker_heartbeat", lambda success=False: beats.append(success))
    monkeypatch.setattr(jobs, "clear_ticker_error", lambda: cleared.append(True))
    stop = threading.Event()
    # One real scheduler cycle, with no wall-clock wait.
    monkeypatch.setattr(stop, "wait", lambda _timeout: stop.set())

    web_server._start_desktop_cron_ticker(stop, interval=0)

    assert ticked == []
    assert True not in beats  # at most the startup liveness beat, never a success
    assert cleared == []

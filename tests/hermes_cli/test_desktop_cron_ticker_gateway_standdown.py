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


def test_legacy_external_provider_starts_without_gate(tmp_path, monkeypatch):
    """A third-party provider predating can_dispatch must start exactly as
    before — the gate is only passed when the signature accepts it (#126907)."""
    import hermes_constants
    from hermes_cli import web_server

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    _set_gateway_running(monkeypatch, False)

    started = {}

    class _Legacy:
        name = "legacy"

        def start(self, stop_event, *, adapters=None, loop=None, interval=60):
            started["kwargs"] = {"adapters": adapters, "loop": loop, "interval": interval}

    import cron.scheduler_provider as sp

    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: _Legacy())

    web_server._start_desktop_cron_ticker(threading.Event(), interval=0)

    assert started["kwargs"] == {"adapters": None, "loop": None, "interval": 0}


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


def test_chronos_fallback_stands_down_while_gateway_live(tmp_path, monkeypatch):
    """End-to-end for #126907 with the real ticker, real Chronos and the real
    built-in fallback (gateway liveness, the rejection and the tick body are
    stubbed; no network): gateway down -> identity rejected -> gateway up must
    produce no fallback ticks while the gateway is live, resuming when it stops.
    """
    import time

    import hermes_constants
    from hermes_cli import web_server

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    gateway = {"running": False}
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "_check_gateway_running", lambda home: gateway["running"])

    from plugins.cron_providers.chronos import ChronosCronScheduler
    from plugins.cron_providers.chronos._nas_client import NasCronClientError

    prov = ChronosCronScheduler()

    class _FakeClient:
        def provision(self, **kw):
            raise NasCronClientError(
                "POST /api/agent-cron/provision returned 403: invalid_client",
                status=403, error_code="invalid_client")

        def cancel(self, **kw):
            return {}

        def list_armed(self):
            return []

    prov._client = _FakeClient()
    monkeypatch.setattr(
        "plugins.cron_providers.chronos._cfg",
        lambda *k, default="": "https://agent.example/" if k[-1] == "callback_url" else "https://portal.test")
    jobs = [
        {"id": "a", "enabled": True, "next_run_at": "2026-06-18T12:00:00+00:00", "state": "scheduled"},
    ]
    monkeypatch.setattr("cron.jobs.load_jobs", lambda: jobs)
    monkeypatch.setattr("cron.jobs.get_job", lambda jid: jobs[0])
    monkeypatch.setattr("cron.executions.recover_interrupted_executions", lambda: 0)

    import cron.scheduler_provider as sp

    monkeypatch.setattr(sp, "resolve_cron_scheduler", lambda: prov)
    ticks = []
    monkeypatch.setattr("cron.scheduler.tick", lambda **kw: ticks.append(time.monotonic()))

    stop = threading.Event()
    try:
        web_server._start_desktop_cron_ticker(stop, interval=0.1)

        deadline = time.monotonic() + 5
        while len(ticks) < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(ticks) >= 3, "fallback ticker never took over after rejection"

        gateway["running"] = True  # a gateway comes up on the same home
        time.sleep(0.3)  # let an in-flight tick finish
        held = len(ticks)
        time.sleep(0.8)  # ~8 ticks would fire here ungated
        assert len(ticks) == held, (
            f"fallback raced the live gateway: {len(ticks) - held} ticks while owned")

        gateway["running"] = False
        deadline = time.monotonic() + 5
        while len(ticks) == held and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(ticks) > held, "fallback did not resume after the gateway stopped"
    finally:
        stop.set()
        for thread in threading.enumerate():
            if thread.name == "cron-scheduler-chronos-fallback":
                thread.join(timeout=5)

"""A Desktop-owned backend stops idle Bot Desktop screens for every profile it serves (#133660).

The serve lease watcher reaches only profiles a client addressed since the process started; this
ticker must also reach a profile no client ever opened, and leave a gateway-owned profile to the
gateway's housekeeping.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def served(tmp_path, monkeypatch):
    """Three served profiles; ``gateway`` names the ones a live gateway owns; ``stopped`` records
    the home each idle check ran under."""
    import hermes_cli.profiles as profiles
    import hermes_cli.web_server as web_server
    from hermes_constants import get_hermes_home
    from tools.bot_desktop import runtime

    homes = {name: tmp_path / name for name in ("default", "coder", "assistant")}
    for home in homes.values():
        home.mkdir()
    state = {"gateway": set(), "stopped": [], "broken": set()}

    def _stop_if_idle():
        home = Path(get_hermes_home())
        if home in state["broken"]:
            raise RuntimeError("unreadable profile")
        state["stopped"].append(home)

    monkeypatch.setattr(profiles, "profiles_to_serve", lambda multiplex: list(homes.items()))
    monkeypatch.setattr(web_server, "_gateway_owns_cron", lambda name, _home: name in state["gateway"])
    monkeypatch.setattr(runtime, "stop_if_idle", _stop_if_idle)
    return homes, state


def test_idle_stop_reaches_every_served_profile_a_gateway_does_not_own(served):
    """No client ever addressed ``assistant``; it is still checked, under its own home."""
    from hermes_cli.desktop_idle_screens import stop_idle_screens

    homes, state = served
    state["gateway"] = {"coder"}

    stop_idle_screens()

    assert state["stopped"] == [homes["default"], homes["assistant"]]


def test_one_unreadable_profile_does_not_skip_the_rest(served):
    from hermes_cli.desktop_idle_screens import stop_idle_screens

    homes, state = served
    state["broken"] = {homes["default"]}

    stop_idle_screens()

    assert state["stopped"] == [homes["coder"], homes["assistant"]]


def test_desktop_backend_runs_the_idle_ticker_from_boot_until_shutdown(monkeypatch):
    """It starts with the backend, not on a client's first ``display.*`` call, and stops with it."""
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    import threading

    import hermes_cli.desktop_idle_screens as idle
    import hermes_cli.web_server as ws

    started = {}
    ran = threading.Event()

    def _ticker(stop_event, interval=60):
        started["stop"] = stop_event
        ran.set()

    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setenv("HERMES_DASHBOARD_SESSION_TOKEN", "desktop-spawn-token")
    monkeypatch.setattr(ws, "_warm_gateway_module", lambda: None)
    monkeypatch.setattr(ws, "_start_desktop_cron_ticker", lambda *_args: None)
    monkeypatch.setattr(idle, "run_idle_screen_ticker", _ticker)

    with TestClient(ws.app):
        assert ran.wait(5)
        assert not started["stop"].is_set()

    assert started["stop"].is_set()

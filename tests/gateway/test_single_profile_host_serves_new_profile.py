"""A host gateway that booted with ONE profile comes up standalone (``SINGLE_PROFILE_REASON``), and
multiplexing is settled once per boot. A profile created afterwards (Desktop's New Agent, ``hermes
profile create``) was then never served: every ticket the Desktop minted for it through the host's
control socket was refused, so its sessions, chat and Delete all failed with "Invalid gateway ticket
response" (#141623). The creator's ``rescan-profiles`` signal (and the 30 s watcher) must restart such
a gateway into the multiplexer; any other standalone verdict must not restart (it would repeat).
"""
import asyncio
import os

import pytest

import hermes_constants
from gateway.config import GatewayConfig
from gateway.run import GatewayRunner
from gateway.status import flush_runtime_status, write_runtime_status
from hermes_cli import gateway_migrate as gm
from hermes_cli.gateway_multiplex_mode import SINGLE_PROFILE_REASON


@pytest.fixture
def standalone_host(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles").mkdir(parents=True)
    (home / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    # The migration preflight's service-manager probes: nothing else on this host runs a gateway.
    monkeypatch.setattr(gm, "_installed_services", lambda _home: [])
    monkeypatch.setattr(gm, "_live_gateway_pid", lambda _home: None)
    monkeypatch.setattr(gm, "_host_supports_migration", lambda: None)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=False)
    runner._running = True
    runner._primary_profile_name = "default"
    runner.session_runtime_descriptor = {"state": "ready", "served_profiles": [], "capabilities": []}
    runner.session_ticket_store = None
    restarts = []
    runner.request_restart = lambda **kw: restarts.append(kw) or True
    return runner, home, restarts


def _boot_verdict(reason):
    write_runtime_status(gateway_state="running", multiplex_standalone_reason=reason, served_profiles=[])
    flush_runtime_status()


def _mkprofile(home, name):
    d = home / "profiles" / name
    d.mkdir()
    (d / "config.yaml").write_text("model: {default: m}\n", encoding="utf-8")
    return d


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_profile_created_after_a_single_profile_boot_restarts_the_host_into_the_multiplexer(standalone_host):
    """End to end through the creator's real signal: the control socket's ``rescan-profiles`` verb."""
    from gateway import control_socket
    from gateway.run_bootstrap import _start_gateway_start_control_socket

    runner, home, restarts = standalone_host
    _boot_verdict(SINGLE_PROFILE_REASON)
    server = await _start_gateway_start_control_socket(runner)
    assert server is not None
    try:
        # Nothing new yet: a rescan of the one-profile host changes nothing.
        answer = await asyncio.to_thread(control_socket.rescan_gateway_profiles, home)
        assert answer["multiplex"] is False and restarts == []

        _mkprofile(home, "daily-news-bot")
        answer = await asyncio.to_thread(control_socket.rescan_gateway_profiles, home)
        assert answer["multiplex"] is False  # this process still serves only default until it restarts
        assert len(restarts) == 1
        # One restart per profile set: the watcher's next tick must not queue another.
        await runner.reconcile_served_profiles(reason="watcher")
        assert len(restarts) == 1
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_a_standalone_gateway_kept_by_a_real_blocker_never_restarts(standalone_host, monkeypatch):
    """A guard that a restart would repeat (a secondary still runs its own gateway) keeps the host as
    it is: restarting would only drop every live turn to land on the same verdict."""
    runner, home, restarts = standalone_host
    _mkprofile(home, "coder")
    monkeypatch.setattr(gm, "_live_gateway_pid",
                        lambda h: os.getpid() if hermes_constants.profile_name_for_home(h) == "coder" else None)
    _boot_verdict("profile(s) 'coder' (pid 1) still run their own gateway; fold them with `x`")
    await runner.reconcile_served_profiles(reason="watcher")
    assert restarts == []

    # Booted single-profile, but by the time a second profile appears that profile runs its own
    # gateway: the preflight blocks, so no restart either.
    _boot_verdict(SINGLE_PROFILE_REASON)
    await runner.reconcile_served_profiles(reason="watcher")
    assert restarts == []

"""Regression for #123490: the PM relaunch wrapper's embedded ``sys.argv`` never appears in the
OS command line, so the process-table scan's strict argv matcher cannot recognize a gateway
relaunched that way. The readiness poll must still accept the gateway's own control socket
answer as liveness (mirrors the fleet version check's existing socket-first behavior)."""

import asyncio
from pathlib import Path

import pytest

from gateway.control_socket import GatewayControlServer

import hermes_cli.gateway_windows as gateway_windows

pytestmark = pytest.mark.platforms("posix")  # Unix-socket transport; named pipes covered on wine2e


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    d = tmp_path / "home" / ".hermes"
    d.mkdir(parents=True)
    return d


def test_scoped_probe_falls_back_to_control_socket(home: Path):
    """No gateway.pid/lock at ``home`` (the scan's normal signal), but a live gateway answers the
    control socket there — the scoped readiness probe must return its PID, not []."""

    async def scenario():
        server = GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 4288, "protocol": 1}})
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, lambda: gateway_windows._live_gateway_pids(home=home))
        finally:
            await server.stop()

    assert _run(scenario()) == [4288]


def test_scoped_probe_stays_empty_with_no_socket_and_no_pid_file(home: Path):
    assert gateway_windows._live_gateway_pids(home=home) == []


def test_all_profiles_probe_falls_back_to_control_socket(monkeypatch, home: Path):
    """The all-profiles readiness poll (used by the post-update relaunch verification) must also
    consult each profile's control socket when the process-table scan sees nothing."""
    monkeypatch.setattr("hermes_cli.gateway.find_gateway_pids", lambda **kwargs: [])
    monkeypatch.setattr("hermes_cli.update_receipt._profile_homes", lambda: [("default", home)])

    async def scenario():
        server = GatewayControlServer(home, verb_handlers={"identify": lambda: {"pid": 4288, "protocol": 1}})
        assert await server.start()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, lambda: gateway_windows._live_gateway_pids(all_profiles=True)
            )
        finally:
            await server.stop()

    assert _run(scenario()) == [4288]

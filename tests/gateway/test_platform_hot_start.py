"""A running gateway starts, restarts and stops ONE platform adapter when the profile's platform config
changes, without a gateway restart, single-profile gateways included (no multiplexer involved).

Live: a real ``hermes gateway run`` on a disposable root; the webhook adapter binds a real loopback-
reachable port, so "started" is a socket that accepts and "stopped" is one that refuses.
"""
import asyncio
import os
import socket
import types
from pathlib import Path

import pytest

from gateway.config import Platform
from tests.gateway.test_multiplex_hot_serve_live import control, start_daemon, stop, tail
from tests.waiting import wait_until


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _accepts(port):
    with socket.socket() as peer:
        peer.settimeout(1)
        return peer.connect_ex(("127.0.0.1", port)) == 0


@pytest.mark.platforms("linux")
@pytest.mark.spawns_gateway_lookalike
def test_single_profile_gateway_starts_restarts_and_stops_a_platform_live(tmp_path):
    user, root = tmp_path / "user", tmp_path / "state"
    user.mkdir()
    root.mkdir(mode=0o700)
    (root / "config.yaml").write_text("gateway:\n  multiplex_profiles: false\n", encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    env = {k: os.environ[k] for k in ("PATH", "LANG", "TZ") if k in os.environ}
    env.update(HOME=str(user), HERMES_HOME=str(root), PYTHONPATH=str(repo), PYTHONUNBUFFERED="1")
    mux = {"root": root, "env": env, "log": tmp_path / "gateway.log"}
    proc, _desc = start_daemon(mux)
    try:
        first, second = _free_port(), _free_port()
        dotenv = root / ".env"
        dotenv.write_text(f"WEBHOOK_ENABLED=true\nWEBHOOK_PORT={first}\n", encoding="utf-8")
        answer = control(root, "reconcile-platforms", timeout=20)
        assert answer["applied"] is True and answer["started"] == ["webhook"], (answer, tail(mux))
        assert answer["platforms"]["webhook"] == "connected"
        wait_until(lambda: _accepts(first), timeout=10, interval=0.1, what="webhook listening")

        dotenv.write_text(f"WEBHOOK_ENABLED=true\nWEBHOOK_PORT={second}\n", encoding="utf-8")
        answer = control(root, "reconcile-platforms", timeout=20)
        assert answer["restarted"] == ["webhook"] and answer["platforms"]["webhook"] == "connected", answer
        wait_until(lambda: _accepts(second) and not _accepts(first), timeout=10, interval=0.1,
                   what="webhook moved to the new port")

        dotenv.write_text(f"WEBHOOK_ENABLED=false\nWEBHOOK_PORT={second}\n", encoding="utf-8")
        answer = control(root, "reconcile-platforms", timeout=20)
        assert answer["stopped"] == ["webhook"], answer
        wait_until(lambda: not _accepts(second), timeout=10, interval=0.1, what="webhook stopped")
        assert proc.poll() is None, tail(mux)
    finally:
        stop(proc, expect=None)


def _runner(adapter):
    """Just the state the retire path reads; the retire logic under test is the real mixin's."""
    from gateway.run_platform_reconcile import GatewayPlatformReconcileMixin

    class _Runner(_RunnerState, GatewayPlatformReconcileMixin):
        pass

    return _Runner(adapter)


class _RunnerState:
    def __init__(self, adapter):
        self._running = True
        self.adapters = {Platform.TELEGRAM: adapter}
        self._failed_platforms = {}
        self.turns = {"agent:main:telegram:dm:42": object(), "agent:main:discord:dm:7": object()}
        self.torn_down_with = None
        self.status = {}

    def _snapshot_running_agents(self):
        return dict(self.turns)

    async def _bounded_adapter_teardown(self, adapter, platform, *, profile=None):
        self.torn_down_with = dict(self.turns)

    def _update_platform_runtime_status(self, key, **fields):
        self.status[key] = fields


def test_disabled_platform_finishes_its_in_flight_turn_before_it_goes_down(monkeypatch):
    import hermes_cli.gateway as gateway_cli
    monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 30.0)
    runner = _runner(types.SimpleNamespace())

    async def scenario():
        retire = asyncio.ensure_future(runner._retire_primary_platform(Platform.TELEGRAM))
        await asyncio.sleep(1.2)  # health: allow HX013 -- still waiting after two poll intervals IS the behaviour
        assert not retire.done() and Platform.TELEGRAM in runner.adapters  # still delivering its turn
        runner.turns.pop("agent:main:telegram:dm:42")  # the turn finished; another platform's runs on
        await asyncio.wait_for(retire, timeout=5)

    asyncio.run(scenario())
    assert runner.torn_down_with == {"agent:main:discord:dm:7": runner.turns["agent:main:discord:dm:7"]}
    assert Platform.TELEGRAM not in runner.adapters
    assert runner.status["telegram"]["platform_state"] == "disabled"

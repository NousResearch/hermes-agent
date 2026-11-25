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

from gateway.config import Platform, PlatformConfig
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
        self.sources = {}

    def _snapshot_running_agents(self):
        return dict(self.turns)

    def _get_cached_session_source(self, key):
        return self.sources.get(key)

    def _delivery_adapter_for(self, source):
        return source.answered_by

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


def test_drain_waits_for_a_turn_this_bot_received_for_another_profile(monkeypatch):
    """A routed turn runs in another profile's namespace, yet this bot answers it: retiring waits."""
    import hermes_cli.gateway as gateway_cli
    monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 30.0)
    bot = types.SimpleNamespace()
    runner = _runner(bot)
    runner.turns = {"agent:other:discord:dm:9": object()}  # session key names neither this platform nor profile
    runner.sources = {"agent:other:discord:dm:9": types.SimpleNamespace(answered_by=bot)}

    async def scenario():
        retire = asyncio.ensure_future(runner._retire_primary_platform(Platform.TELEGRAM))
        await asyncio.sleep(1.2)  # health: allow HX013 -- still waiting after two poll intervals IS the behaviour
        assert not retire.done()
        runner.turns.clear()
        await asyncio.wait_for(retire, timeout=5)

    asyncio.run(scenario())
    assert runner.torn_down_with == {}


def test_build_print_tracks_env_only_credentials_and_typed_settings(tmp_path):
    """Slack reads SLACK_APP_TOKEN straight from the profile's .env and Telegram caches reply_to_mode:
    either change is a rebuild. Runner-read settings (home channel, allowlists) are not."""
    from gateway.config import PlatformConfig
    from gateway.run_platform_reconcile import platform_build_print
    env = tmp_path / ".env"

    def print_of(text, **settings):
        env.write_text(text, encoding="utf-8")
        from agent.secret_scope import invalidate_env_file_cache
        invalidate_env_file_cache(env)
        return platform_build_print(Platform.SLACK, PlatformConfig(enabled=True, token="xoxb-1", **settings), tmp_path)

    base = print_of("SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-1\n")
    assert print_of("SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-2\n").differs(base)
    assert print_of("SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-1\n", reply_to_mode="all").differs(base)
    assert not print_of("SLACK_BOT_TOKEN=xoxb-1\nSLACK_APP_TOKEN=xapp-1\nOTHER_KEY=x\n").differs(base)


def test_only_adapters_a_build_attempt_owns_are_stopped():
    """The LOCAL session adapter is installed at runtime, never built from config: a reconcile whose
    config does not list it must leave it alone (it carries the app's live sessions)."""
    from gateway.run_platform_reconcile import BuildPrint, plan_platform_changes
    webhook = BuildPrint("w", "e")
    plan = plan_platform_changes({}, {Platform.WEBHOOK: webhook}, {Platform.WEBHOOK, Platform.LOCAL})
    assert plan.stop == (Platform.WEBHOOK,) and Platform.LOCAL not in (*plan.stop, *plan.restart)
    unchanged = plan_platform_changes({Platform.WEBHOOK: webhook}, {Platform.WEBHOOK: webhook}, {Platform.LOCAL})
    assert not unchanged and not unchanged.start  # attempted at boot and failed: the boot verdict stands


def test_secondary_restart_is_not_refused_as_a_duplicate_of_itself(monkeypatch):
    """A settings-only restart keeps the same token: the retired adapter's claim must leave the rescan's
    ``claimed`` map, or the rebuild is parked as a duplicate of the adapter it replaces."""
    import hermes_cli.gateway as gateway_cli
    monkeypatch.setattr(gateway_cli, "_get_restart_drain_timeout", lambda: 0.0)
    adapter = types.SimpleNamespace()
    runner = _runner(adapter)
    runner._profile_adapters = {"b": {Platform.TELEGRAM: adapter}}
    runner._profile_failed_platforms = {}
    claim = ("telegram", "token-hash")
    runner._adapter_claims = lambda platform, a: (claim, None)
    runner._hold_secondary_claims = lambda *a: None
    runner._release_secondary_claims = lambda *a: None
    claimed = {claim: "b", ("discord", "other"): "c"}
    asyncio.run(runner._retire_secondary_platform("b", Path("."), Platform.TELEGRAM, {}, claimed))
    assert claimed == {("discord", "other"): "c"}
    assert Platform.TELEGRAM not in runner._profile_adapters["b"]


@pytest.mark.asyncio
async def test_a_reconnect_superseded_while_connecting_never_publishes(monkeypatch):
    """A reconcile retires a queued platform (and may rebuild it) while the watcher's older attempt is
    still connecting with the old config: that attempt's adapter is discarded, never published."""
    from unittest.mock import AsyncMock
    from tests.gateway.test_platform_reconnect import StubAdapter, _make_runner
    runner = _make_runner()
    runner._update_platform_runtime_status = lambda *a, **k: None
    runner._install_reconnected_adapter = AsyncMock()
    runner._wire_adapter_handlers = lambda adapter: None
    stale = StubAdapter()
    stale.disconnect = AsyncMock()
    replacement = {"config": PlatformConfig(enabled=True, token="new"), "attempts": 0, "next_retry": 0}

    async def connect_then_superseded(adapter, platform, *, is_reconnect):
        runner._failed_platforms[platform] = replacement  # the reconcile's new slot for this platform
        return True

    monkeypatch.setattr(runner, "_create_adapter", lambda p, c: stale)
    monkeypatch.setattr(runner, "_connect_adapter_with_timeout", connect_then_superseded)
    runner._failed_platforms[Platform.TELEGRAM] = {
        "config": PlatformConfig(enabled=True, token="old"), "attempts": 0, "next_retry": 0}
    await runner._reconnect_failed_platform(Platform.TELEGRAM, 0)
    runner._install_reconnected_adapter.assert_not_awaited()
    stale.disconnect.assert_awaited()
    assert runner._failed_platforms[Platform.TELEGRAM] is replacement and replacement["attempts"] == 0

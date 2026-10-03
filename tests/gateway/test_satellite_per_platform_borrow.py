"""A shared-bot satellite borrows the primary per platform, not all-or-nothing.

``ops`` owns its own Telegram bot but is route-only on Discord (a ``profile_routes`` entry through the
default bot). Restored sources and synthetic injections (async-delegation completions, heartbeats,
goal notices) carry no transport ref, so they resolve through ``_adapters_for_profile``; before the
fix a non-empty map skipped the satellite fallback and the Discord lane had no transport — durable
completion rows stayed ``pending`` forever with no log line.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.profile_routing import parse_profile_routes
from gateway.session import SessionSource


class _Stub(BasePlatformAdapter):
    pass


_Stub.__abstractmethods__ = frozenset()


def _stub(platform, runner, label, owner=None):
    adapter = _Stub.__new__(_Stub)
    adapter.platform, adapter.gateway_runner, adapter.label = platform, runner, label
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._pending_messages, adapter._active_sessions = {}, {}
    if owner:
        adapter.set_owner_profile(owner)
    return adapter


@pytest.fixture
def rig(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    home = tmp_path / "hh"
    for name in ("ops", "team_b"):
        (home / "profiles" / name).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.platforms = {p: PlatformConfig(enabled=True, extra={}) for p in (Platform.DISCORD, Platform.TELEGRAM)}
    runner.config.profile_routes = parse_profile_routes([
        {"name": "ops-thread", "platform": "discord", "profile": "ops", "chat_id": "123"},
    ])
    runner._primary_profile_name = "default"
    runner._profile_failed_platforms = {}
    primary_discord = _stub(Platform.DISCORD, runner, "PRIMARY_DISCORD")
    primary_tg = _stub(Platform.TELEGRAM, runner, "PRIMARY_TELEGRAM")
    ops_tg = _stub(Platform.TELEGRAM, runner, "OPS_TELEGRAM", owner="ops")
    team_b_tg = _stub(Platform.TELEGRAM, runner, "TEAM_B_TELEGRAM", owner="team_b")
    runner.adapters = {Platform.DISCORD: primary_discord, Platform.TELEGRAM: primary_tg}
    runner._profile_adapters = {"ops": {Platform.TELEGRAM: ops_tg}, "team_b": {Platform.TELEGRAM: team_b_tg}}
    served = [("default", home), ("ops", home / "profiles" / "ops"), ("team_b", home / "profiles" / "team_b")]
    with patch("hermes_cli.profiles.profiles_to_serve", return_value=served), \
            patch("hermes_cli.profiles.get_profile_dir",
                  side_effect=lambda n: home if n == "default" else home / "profiles" / n), \
            patch("hermes_cli.profiles.profile_exists", return_value=True):
        yield SimpleNamespace(runner=runner, home=home, primary_discord=primary_discord,
                              primary_tg=primary_tg, ops_tg=ops_tg)


def _restored(profile, platform=Platform.DISCORD, chat_id="123"):
    return SessionSource(platform=platform, chat_id=chat_id, chat_type="thread", user_id="7", profile=profile)


def test_mixed_credential_satellite_borrows_only_its_routed_platform(rig):
    r = rig.runner
    assert r._is_shared_bot_satellite("ops") is True
    # Routed Discord: the default bot is the only transport those chats can reach.
    assert r._resolve_injection_adapter("discord", _restored("ops")) is rig.primary_discord
    assert r._delivery_adapter_for(_restored("ops")) is rig.primary_discord
    assert r._authorization_adapter(Platform.DISCORD, "ops") is rig.primary_discord
    # Its own Telegram bot stays its boundary on Telegram — never the primary's.
    assert r._resolve_injection_adapter("telegram", _restored("ops", Platform.TELEGRAM)) is rig.ops_tg
    assert r._adapters_for_profile("ops") == {Platform.DISCORD: rig.primary_discord, Platform.TELEGRAM: rig.ops_tg}


def test_satellite_own_adapter_on_the_routed_platform_wins(rig):
    r = rig.runner
    ops_discord = _stub(Platform.DISCORD, r, "OPS_DISCORD", owner="ops")
    r._profile_adapters["ops"][Platform.DISCORD] = ops_discord
    assert r._resolve_injection_adapter("discord", _restored("ops")) is ops_discord


def test_fail_closed_rows_are_unchanged(rig):
    r = rig.runner
    # Not a satellite (no default-bot route targets it): its Telegram map never grows a Discord entry.
    assert r._is_shared_bot_satellite("team_b") is False
    assert r._resolve_injection_adapter("discord", _restored("team_b")) is None
    assert Platform.DISCORD not in r._adapters_for_profile("team_b")
    # A bot of its own queued for reconnect means it owns a credential: never borrow.
    r._profile_failed_platforms = {"ops": {Platform.DISCORD: object()}}
    assert r._resolve_injection_adapter("discord", _restored("ops")) is None
    r._profile_failed_platforms = {}
    # The primary has no transport for the routed platform: nothing to borrow.
    del r.adapters[Platform.DISCORD]
    assert r._resolve_injection_adapter("discord", _restored("ops")) is None
    r.adapters[Platform.DISCORD] = rig.primary_discord
    # Standalone gateway: no satellites at all.
    r.config.multiplex_profiles = False
    assert r._resolve_injection_adapter("discord", _restored("ops")) is None


def test_async_delegation_completion_is_deliverable_and_a_held_lane_logs_once(rig, caplog):
    r = rig.runner
    evt = {"type": "async_delegation", "delegation_id": "d1", "session_key": "agent:ops:discord:thread:123"}

    async def _no_parent(_sid):
        return "deliver"

    r._classify_completion_target = _no_parent
    r._build_process_event_source = lambda _evt: _restored("ops")
    assert asyncio.run(r._completion_delivery_ready(evt)) is True

    # A genuinely missing transport stays not-ready, but is no longer invisible — once per lane.
    del r.adapters[Platform.DISCORD]
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert asyncio.run(r._completion_delivery_ready(evt)) is False
    held = [rec for rec in caplog.records if "Async completion held" in rec.getMessage()]
    assert len(held) == 1 and "'ops'" in held[0].getMessage() and "discord" in held[0].getMessage()

"""The intake / delivery transport matrix (#88715 phase 4; gateway/AGENTS.md § Profile scope).

Real ``GatewayRunner`` resolvers against a temp ``HERMES_HOME`` with three served profiles:
``default`` (shared bot), ``ops`` (satellite routed through the shared bot) and ``team_b`` (its own
bot). Every row is asserted in one fixture so the two historical fixes (#69246 shared transport
preserved, #70625 route-override egress) cannot drift apart again.
"""

import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.platforms.base import BasePlatformAdapter
from gateway.profile_routing import parse_profile_routes
from gateway.session import SessionSource
from gateway.session_identity import resolve_identity


class _Stub(BasePlatformAdapter):
    pass


_Stub.__abstractmethods__ = frozenset()


def _stub(platform, runner, label):
    adapter = _Stub.__new__(_Stub)
    adapter.platform, adapter.gateway_runner, adapter.label = platform, runner, label
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._pending_messages, adapter._active_sessions = {}, {}
    return adapter


def _runner(*, multiplex, routes=()):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex)
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
    runner.config.profile_routes = parse_profile_routes(list(routes))
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {}
    runner._primary_profile_name = "default"
    primary = _stub(Platform.TELEGRAM, runner, "PRIMARY")
    team_b = _stub(Platform.TELEGRAM, runner, "TEAM_B")
    team_b.set_owner_profile("team_b")
    runner.adapters = {Platform.TELEGRAM: primary}
    runner._profile_adapters = {"team_b": {Platform.TELEGRAM: team_b}, "ops": {}}
    return SimpleNamespace(runner=runner, primary=primary, team_b=team_b)


@pytest.fixture
def mux(tmp_path, monkeypatch):
    home = tmp_path / "hh"
    for name in ("ops", "team_b"):
        (home / "profiles" / name).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    served = [("default", home), ("ops", home / "profiles" / "ops"), ("team_b", home / "profiles" / "team_b")]
    rig = _runner(multiplex=True, routes=[
        # shared credential → satellite (#69246)
        {"name": "ops-dm", "platform": "telegram", "profile": "ops", "chat_id": "72719239"},
        # per-credential source → a profile that owns its own bot (#70625, decision D1)
        {"name": "team-b-via-shared", "platform": "telegram", "profile": "team_b", "chat_id": "555"},
        # secondary-owned transport → default (the inverse row)
        {"name": "default-via-team-b", "platform": "telegram", "profile": "default", "chat_id": "999",
         "bot_profile": "team_b"},
    ])
    rig.home = home
    with patch("hermes_cli.profiles.profiles_to_serve", return_value=served), \
            patch("hermes_cli.profiles.get_profile_dir",
                  side_effect=lambda n: home if n == "default" else home / "profiles" / n), \
            patch("hermes_cli.profiles.profile_exists", return_value=True):
        yield rig


def _live(rig, adapter, chat_id, **resolve_kwargs):
    source = adapter.build_source(chat_id=chat_id, chat_type="dm", user_id=chat_id)
    identity = resolve_identity(source, runner=rig.runner, **resolve_kwargs)
    return source, identity


def test_live_rows_reply_through_the_receiving_bot_whatever_the_runtime(mux):
    """Rows 1–4 of the matrix: with live provenance, intake and delivery are BOTH the receiving
    adapter; only the runtime (profile, namespace, home) follows the route."""
    r = mux.runner
    # per-credential, no route: owner on every axis
    own, own_id = _live(mux, mux.team_b, "1", transport_profile="team_b")
    assert (own_id.transport_profile, own_id.runtime_profile) == ("team_b", "team_b")
    assert r._intake_adapter_for(own) is r._delivery_adapter_for(own) is mux.team_b

    # shared credential → satellite: runtime ops, transport stays the shared bot (#69246)
    sat, sat_id = _live(mux, mux.primary, "72719239")
    assert (sat_id.transport_profile, sat_id.runtime_profile) == ("default", "ops")
    assert r._intake_adapter_for(sat) is r._delivery_adapter_for(sat) is mux.primary
    assert r._session_key_for_source(sat) == "agent:ops:telegram:dm:72719239"

    # per-credential source routed to a profile that owns a bot: runtime team_b, reply via the
    # bot that received it — conversation continuity (D1), not the routed profile's bot (#70625).
    override, override_id = _live(mux, mux.primary, "555")
    assert (override_id.transport_profile, override_id.runtime_profile) == ("default", "team_b")
    assert override_id.runtime_home == mux.home / "profiles" / "team_b"
    assert r._intake_adapter_for(override) is r._delivery_adapter_for(override) is mux.primary

    # secondary-owned transport → default: agent:main key, default home, reply via the secondary bot
    inverse, inverse_id = _live(mux, mux.team_b, "999", transport_profile="team_b")
    assert (inverse_id.transport_profile, inverse_id.runtime_profile) == ("team_b", "default")
    assert inverse_id.runtime_home == mux.home and inverse_id.namespace == "agent:main"
    assert r._intake_adapter_for(inverse) is r._delivery_adapter_for(inverse) is mux.team_b
    assert mux.team_b._source_session_key(inverse) == "agent:main:telegram:dm:999"

    # Provenance survives a reconnect: the transport ref dies, the identity still names the bot.
    replacement = _stub(Platform.TELEGRAM, r, "PRIMARY2")
    r.adapters = {Platform.TELEGRAM: replacement}
    assert r._intake_adapter_for(sat) is r._delivery_adapter_for(sat) is replacement


def test_restored_rows_fail_closed_on_intake_and_deliver_only_via_a_unique_owner(mux, tmp_path, monkeypatch):
    """Row 5: no live provenance → intake is ``None`` (nothing may re-dispatch or apply intake
    policy); delivery goes to the unique owner of ``(platform, runtime)`` — the primary for a
    shared-bot satellite, a secondary's own bot, ``None`` when that bot is down (never the default
    bot). Outside multiplexing the one adapter is the receiver by construction."""
    r = mux.runner

    def restored(profile):
        return SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42", profile=profile)

    for profile, owner in ((None, mux.primary), ("ops", mux.primary), ("team_b", mux.team_b)):
        assert r._intake_adapter_for(restored(profile)) is None, profile
        assert r._delivery_adapter_for(restored(profile)) is owner, profile
    # The admitting allowlist for a restored source is its delivering bot's (auto-resume, injection).
    assert r._authorization_home_for_source(restored("team_b")) == mux.home / "profiles" / "team_b"
    assert r._authorization_home_for_source(restored("ops")) == mux.home

    # team_b's bot failed to connect and is queued for reconnect: it owns a credential, so the
    # route that also names it through the shared bot must not turn it into a satellite.
    r._profile_adapters["team_b"] = {}
    r._profile_failed_platforms = {"team_b": {Platform.TELEGRAM: {}}}
    assert r._delivery_adapter_for(restored("team_b")) is None
    assert r._authorization_home_for_source(restored("team_b")) is None

    # A dead transport ref with no identity is a restored row too, never a lookup by runtime.
    stale = restored("ops")
    stale._transport_adapter_ref = weakref.ref(_stub(Platform.TELEGRAM, r, "GONE"))
    assert r._intake_adapter_for(stale) is None and r._delivery_adapter_for(stale) is mux.primary

    solo_home = tmp_path / "solo"
    solo_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(solo_home))
    solo = _runner(multiplex=False)
    bare = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="42")
    assert solo.runner._intake_adapter_for(bare) is solo.runner._delivery_adapter_for(bare) is solo.primary
    assert solo.runner._intake_adapter_for(None) is None and solo.runner._delivery_adapter_for(None) is None

"""Invariants for ``gateway/session_identity.py`` (#88715 phase 1).

One frozen ``RoutingIdentity`` per inbound event, resolved by the real ``GatewayRunner`` resolvers
(no patched predicates) against a temp ``HERMES_HOME`` with two served profiles.
"""

import dataclasses
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.platforms.base import BasePlatformAdapter
from gateway.profile_routing import parse_profile_routes
from gateway.session import SessionSource, build_session_key
from gateway.session_identity import (
    IdentityUnresolved, RoutingIdentity, clear_identity, identity_of, replace_source,
    resolve_identity, restore_identity,
)


class _Stub(BasePlatformAdapter):
    pass


_Stub.__abstractmethods__ = frozenset()


def _stub(platform, runner, label):
    adapter = _Stub.__new__(_Stub)
    adapter.platform, adapter.gateway_runner, adapter.label = platform, runner, label
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._pending_messages, adapter._active_sessions = {}, {}
    return adapter


def _runner(home, *, multiplex, routes=()):
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
    return SimpleNamespace(runner=runner, home=home, primary=primary, team_b=team_b)


@pytest.fixture
def mux(tmp_path, monkeypatch):
    """Default home + satellite ``ops`` (routed through the shared bot for chat 72719239) + ``team_b``
    owning its own Telegram bot; a route to unserved ``ghost`` for chat 4040."""
    home = tmp_path / "hh"
    for name in ("ops", "team_b"):
        (home / "profiles" / name).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    served = [("default", home), ("ops", home / "profiles" / "ops"), ("team_b", home / "profiles" / "team_b")]
    rig = _runner(home, multiplex=True, routes=[
        {"name": "admin-dm", "platform": "telegram", "profile": "ops", "chat_id": "72719239",
         "response_policy": "silent"},
        {"name": "ghost", "platform": "telegram", "profile": "ghost", "chat_id": "4040"},
    ])
    with patch("hermes_cli.profiles.profiles_to_serve", return_value=served), \
            patch("hermes_cli.profiles.get_profile_dir", side_effect=lambda n: home if n == "default" else home / "profiles" / n), \
            patch("hermes_cli.profiles.profile_exists", return_value=True):
        yield rig


def test_identity_is_one_frozen_value_that_every_reader_agrees_on(mux):
    """Shared bot → routed satellite: transport stays default (authorization), runtime is ``ops``;
    the pinned identity is frozen, equal by value, and the legacy readers all report the same
    answer it does. A dedicated secondary resolves to itself on both axes."""
    routed = mux.primary.build_source(chat_id="72719239", chat_type="dm", user_id="72719239")
    identity = resolve_identity(routed, runner=mux.runner)

    assert identity == RoutingIdentity(
        transport_profile="default", runtime_profile="ops", authorization_home=mux.home,
        runtime_home=mux.home / "profiles" / "ops", response_policy="silent")
    assert identity.namespace == "agent:ops" and identity.store_path == mux.home / "profiles" / "ops" / "state.db"
    assert identity.adapter() is mux.primary
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.runtime_profile = "default"
    assert identity_of(routed) is identity
    # Thin readers: same object, same answers — no second derivation.
    assert mux.runner._authorization_home_for_source(routed) == mux.home
    assert mux.runner._resolve_profile_home_for_source(routed) == mux.home / "profiles" / "ops"
    assert mux.runner._transport_owner(routed) == (mux.primary, None)
    assert mux.primary._source_session_key(routed) == "agent:ops:telegram:dm:72719239"
    assert mux.runner._session_key_for_source(routed) == mux.primary._source_session_key(routed)

    own = mux.team_b.build_source(chat_id="72719239", chat_type="dm", user_id="72719239")
    own_identity = resolve_identity(own, runner=mux.runner, transport_profile="team_b")
    assert (own_identity.transport_profile, own_identity.runtime_profile) == ("team_b", "team_b")
    assert own_identity.authorization_home == own_identity.runtime_home == mux.home / "profiles" / "team_b"
    assert own_identity != identity
    # Provenance is not identity: a second event from the same bot has the same identity.
    again = mux.primary.build_source(chat_id="72719239", chat_type="dm", user_id="72719239")
    assert resolve_identity(again, runner=mux.runner) == identity
    assert hash(resolve_identity(again, runner=mux.runner)) == hash(identity)


def test_unresolved_under_multiplex_raises_and_never_means_default(mux, tmp_path, monkeypatch):
    """A route to an unserved profile raises ``IdentityUnresolved`` (the runner drops the event);
    outside multiplexing the same source resolves to an explicit default identity whose keys stay
    byte-identical to the legacy ``agent:main`` namespace."""
    rejected = mux.primary.build_source(chat_id="4040", chat_type="dm", user_id="4040")
    with pytest.raises(IdentityUnresolved):
        resolve_identity(rejected, runner=mux.runner)
    assert identity_of(rejected) is None
    assert mux.runner._admit_primary_source(rejected, mux.home) is None

    solo_home = tmp_path / "solo"
    solo_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(solo_home))
    solo = _runner(solo_home, multiplex=False)
    source = solo.primary.build_source(chat_id="4040", chat_type="dm", user_id="4040")
    identity = resolve_identity(source, runner=solo.runner)
    assert (identity.transport_profile, identity.runtime_profile, identity.multiplexed) == ("default", "default", False)
    assert identity.runtime_home == solo_home
    assert identity.namespace == "agent:main"
    assert solo.primary._source_session_key(source) == build_session_key(source) == "agent:main:telegram:dm:4040"
    assert source.profile is None  # wire format untouched
    assert solo.runner._authorization_home_for_source(source) is None  # ambient scope, as before


def test_replace_source_keeps_identity_and_transport_where_dataclasses_replace_drops_them(mux):
    routed = mux.primary.build_source(chat_id="72719239", chat_type="dm", user_id="72719239")
    identity = resolve_identity(routed, runner=mux.runner)

    bare = dataclasses.replace(routed)
    assert identity_of(bare) is None and mux.runner._transport_owner(bare) is None

    copied = replace_source(routed, thread_id="7")
    assert copied.thread_id == "7" and copied is not routed
    assert identity_of(copied) is identity
    assert mux.runner._transport_owner(copied) == (mux.primary, None)
    assert isinstance(copied._transport_adapter_ref, weakref.ref)
    assert Path(copied._authorization_profile_home) == mux.home
    assert mux.primary._source_session_key(copied) == "agent:ops:telegram:dm:72719239:7"


def test_replace_source_keeps_route_policy_before_identity_is_pinned(mux):
    routed = mux.primary.build_source(
        chat_id="72719239", chat_type="dm", user_id="72719239"
    )
    assert identity_of(routed) is None

    copied = replace_source(routed, thread_id="7")

    assert resolve_identity(copied, runner=mux.runner).response_policy == "silent"


def test_clear_identity_drops_silent_policy_before_source_is_reused(mux):
    source = mux.primary.build_source(
        chat_id="72719239", chat_type="dm", user_id="72719239"
    )
    assert resolve_identity(source, runner=mux.runner).response_policy == "silent"

    clear_identity(source)
    source.chat_id = "unrouted"

    assert resolve_identity(source, runner=mux.runner).response_policy == "normal"


def test_route_probe_resets_stale_response_policy_when_route_no_longer_matches(mux):
    source = mux.primary.build_source(
        chat_id="72719239", chat_type="dm", user_id="72719239"
    )
    assert source._profile_route_response_policy == "silent"

    source.chat_id = "unrouted"
    mux.runner._profile_name_for_source(source)

    assert source._profile_route_response_policy == "normal"


def test_restored_source_recomputes_current_route_response_policy(mux):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="72719239",
        user_id="72719239",
        chat_type="dm",
        profile="ops",
    )

    identity = restore_identity(
        source, runner=mux.runner, transport_profile="default"
    )

    assert identity is not None
    assert identity.response_policy == "silent"

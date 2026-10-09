"""Identity survives restore (#88715 phase 5).

A restarted multiplexed gateway rebuilds every lane from the routing index; the key namespace says
where the lane runs but not which bot received it. ``SessionEntry.transport_profile`` persists that
bot, ``_restored_source`` re-pins the identity, and delivery goes through that bot or fails closed.
Real ``GatewayRunner`` resolvers and a real ``SessionStore`` over a temp ``HERMES_HOME`` — no
patched predicates.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.platforms.base import BasePlatformAdapter
from gateway.profile_routing import parse_profile_routes
from gateway.session import SessionEntry, SessionStore
from gateway.session_identity import identity_of, resolve_identity


class _Stub(BasePlatformAdapter):
    pass


_Stub.__abstractmethods__ = frozenset()


def _stub(platform, runner, label):
    adapter = _Stub.__new__(_Stub)
    adapter.platform, adapter.gateway_runner, adapter.label = platform, runner, label
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._pending_messages, adapter._active_sessions = {}, {}
    return adapter


_ROUTES = [
    # Satellite ``ops`` drains through the default bot for this chat ...
    {"name": "admin-dm", "platform": "telegram", "profile": "ops", "chat_id": "72719239"},
    # ... and is ALSO the routed runtime for a chat that team_b's OWN bot receives.
    {"name": "b-to-ops", "platform": "telegram", "profile": "ops", "chat_id": "555", "bot_profile": "team_b"},
]


def _runner(home, *, multiplex=True):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=multiplex, sessions_dir=home / "sessions")
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={})}
    runner.config.profile_routes = parse_profile_routes(list(_ROUTES) if multiplex else [])
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
    home = tmp_path / "hh"
    for name in ("ops", "team_b"):
        (home / "profiles" / name).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    served = [("default", home), ("ops", home / "profiles" / "ops"), ("team_b", home / "profiles" / "team_b")]
    with patch("gateway.profile_serving.profiles_to_serve", return_value=served), \
            patch("profiles.paths.get_profile_dir", side_effect=lambda n: home if n == "default" else home / "profiles" / n), \
            patch("profiles.registry.profile_exists", return_value=True):
        yield _runner(home)


def _restart(rig, entry: SessionEntry):
    """A fresh process: the entry comes back through its wire dict, the runner is rebuilt."""
    fresh = _runner(rig.home)
    restored = SessionEntry.from_dict(entry.to_dict())
    return fresh, fresh.runner._restored_source(restored), restored


def test_restored_lane_delivers_through_the_bot_that_received_it_never_the_default_by_heuristic(mux, tmp_path):
    """Chat 555 arrives on team_b's bot and runs as ``ops``. Before the restart the transport ref
    answers; after it only the persisted ``transport_profile`` can — without it the shared-bot
    heuristic (ops IS a satellite of the default bot, for chat 72719239) hands the lane to the
    DEFAULT bot. The satellite lane itself keeps its default-bot egress, the row lands in state.db,
    and a pre-column entry (``transport_profile`` absent) still resolves as before."""
    store = SessionStore(sessions_dir=mux.home / "sessions", config=mux.runner.config)
    mux.runner.session_store = store

    routed = mux.team_b.build_source(chat_id="555", chat_type="dm", user_id="555")
    identity = resolve_identity(routed, runner=mux.runner, transport_profile="team_b")
    assert (identity.transport_profile, identity.runtime_profile) == ("team_b", "ops")
    entry = store.get_or_create_session(routed)
    assert entry.session_key.startswith("agent:ops:") and entry.transport_profile == "team_b"
    assert entry.to_dict()["transport_profile"] == "team_b"
    row = store._db_for_key(entry.session_key).get_session(entry.session_id)
    assert row["transport_profile"] == "team_b" and row["profile_name"] == "ops"

    fresh, source, restored = _restart(mux, entry)
    assert restored.transport_profile == "team_b"
    assert fresh.runner._transport_owner(source) is None  # no live provenance survives a restart
    restored_identity = identity_of(source)
    assert restored_identity is not None and restored_identity.transport is None
    assert (restored_identity.transport_profile, restored_identity.runtime_profile) == ("team_b", "ops")
    assert restored_identity.authorization_home == mux.home / "profiles" / "team_b"
    assert restored_identity.runtime_home == mux.home / "profiles" / "ops"
    assert fresh.runner._delivery_adapter_for(source) is fresh.team_b
    assert fresh.runner._adapter_profile_for_source(source) == "team_b"
    assert fresh.runner._authorization_home_for_source(source) == mux.home / "profiles" / "team_b"
    # Fail closed: team_b's bot did not reconnect → nothing delivers; the default bot never does.
    fresh.runner._profile_adapters["team_b"] = {}
    assert fresh.runner._delivery_adapter_for(source) is None

    # The satellite lane (shared default bot, runtime ops) keeps its default-bot egress.
    shared = mux.primary.build_source(chat_id="72719239", chat_type="dm", user_id="72719239")
    resolve_identity(shared, runner=mux.runner)
    shared_entry = store.get_or_create_session(shared)
    assert shared_entry.transport_profile == "default"
    fresh2, shared_source, _ = _restart(mux, shared_entry)
    assert identity_of(shared_source).transport_profile == "default"
    assert fresh2.runner._delivery_adapter_for(shared_source) is fresh2.primary

    # A routing entry written before the column existed: nothing is pinned, old chain unchanged.
    legacy = entry.to_dict()
    legacy.pop("transport_profile")
    fresh3 = _runner(mux.home)
    legacy_source = fresh3.runner._restored_source(SessionEntry.from_dict(legacy))
    assert identity_of(legacy_source) is None
    assert fresh3.runner._delivery_adapter_for(legacy_source) is fresh3.primary  # the heuristic, as before


def test_standalone_gateway_persists_nothing_and_keys_stay_agent_main(tmp_path, monkeypatch):
    """Control: outside multiplexing there is one bot and one home — no transport is recorded, the
    wire dict is byte-identical to before, and a restored source resolves as it always did."""
    home = tmp_path / "solo"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    solo = _runner(home, multiplex=False)
    store = SessionStore(sessions_dir=home / "sessions", config=solo.runner.config)
    source = solo.primary.build_source(chat_id="4040", chat_type="dm", user_id="4040")
    resolve_identity(source, runner=solo.runner)
    entry = store.get_or_create_session(source)
    assert entry.session_key == "agent:main:telegram:dm:4040"
    assert entry.transport_profile is None and "transport_profile" not in entry.to_dict()
    assert store._db_for_key(entry.session_key).get_session(entry.session_id)["transport_profile"] is None
    fresh = _runner(home, multiplex=False)
    restored = fresh.runner._restored_source(SessionEntry.from_dict(entry.to_dict()))
    assert identity_of(restored) is None
    assert fresh.runner._delivery_adapter_for(restored) is fresh.primary
    assert fresh.runner._session_key_for_source(restored) == "agent:main:telegram:dm:4040"

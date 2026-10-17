"""A detached completion for a session a ``profile_routes`` entry re-homed must resolve the
RECEIVING bot's adapter, not the runtime profile's adapter map (#139726).

A route may name ``bot_profile`` (the bot that received the message) alongside ``profile`` (the
runtime that executes the turn). The routing index persists the receiving bot on
``SessionEntry.transport_profile`` (``transport_profile_of``) and ``_restored_source`` re-pins it on
the source rebuilt from that entry. ``_delivery_adapter_for`` already answers through it — live
replies go out of the bot the user is talking to — but the synthetic-injection resolver looked the
platform up in the RUNTIME profile's map, found no adapter, and ``_completion_delivery_ready`` then
refused the delivery: re-queued every 2s with nothing logged until a restart dropped the row.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.platforms.base import BasePlatformAdapter
from gateway.profile_routing import parse_profile_routes
from gateway.session import SessionSource
from gateway.session_identity import resolve_identity, restore_identity, transport_profile_of


class _Stub(BasePlatformAdapter):
    pass


_Stub.__abstractmethods__ = frozenset()


def _stub(platform, runner, label):
    adapter = _Stub.__new__(_Stub)
    adapter.platform, adapter.gateway_runner, adapter.label = platform, runner, label
    adapter._pending_messages, adapter._active_sessions = {}, {}
    return adapter


@pytest.fixture
def routed(tmp_path, monkeypatch):
    """Profile ``signalbox`` owns the Signal bot; the route runs those chats in ``planner``."""
    from gateway.run import GatewayRunner

    home = tmp_path / "hh"
    for name in ("planner", "signalbox"):
        (home / "profiles" / name).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True, extra={}),
                               Platform.SIGNAL: PlatformConfig(enabled=True, extra={})}
    runner.config.profile_routes = parse_profile_routes([
        {"name": "signal-group", "platform": "signal", "profile": "planner",
         "chat_id": "G1", "bot_profile": "signalbox"},
    ])
    runner.pairing_store = PairingStore(profile="default")
    runner.pairing_stores = {}
    runner._primary_profile_name = "default"
    primary = _stub(Platform.TELEGRAM, runner, "PRIMARY")
    signalbox = _stub(Platform.SIGNAL, runner, "SIGNALBOX")
    signalbox.set_owner_profile("signalbox")
    runner.adapters = {Platform.TELEGRAM: primary}
    runner._profile_adapters = {"planner": {}, "signalbox": {Platform.SIGNAL: signalbox}}
    served = [("default", home), ("planner", home / "profiles" / "planner"),
              ("signalbox", home / "profiles" / "signalbox")]
    with patch("hermes_cli.profiles.profiles_to_serve", return_value=served), \
            patch("hermes_cli.profiles.get_profile_dir", side_effect=lambda n: home / "profiles" / n), \
            patch("hermes_cli.profiles.profile_exists", return_value=True):
        yield SimpleNamespace(runner=runner, home=home, primary=primary, signalbox=signalbox)


def _restored(platform, chat_id, profile):
    return SessionSource(platform=platform, chat_id=chat_id, chat_type="group",
                         user_id="7", profile=profile)


def test_route_names_the_receiving_bot_and_the_runtime_separately(routed):
    """The premise the fix rests on: the routed event's identity (and what the routing index
    persists) carries the RECEIVING bot, while ``source.profile`` carries the runtime."""
    source = routed.signalbox.build_source(chat_id="G1", chat_type="group", user_id="7")
    identity = resolve_identity(source, runner=routed.runner)
    assert (identity.transport_profile, identity.runtime_profile) == ("signalbox", "planner")
    assert transport_profile_of(source) == "signalbox"


def test_restored_routed_source_resolves_the_receiving_bots_transport(routed):
    """The regression: live replies answer through the receiving bot, and so must a detached
    completion injected for the same session."""
    restored = _restored(Platform.SIGNAL, "G1", "planner")
    restore_identity(restored, runner=routed.runner, transport_profile="signalbox")
    assert routed.runner._delivery_adapter_for(restored) is routed.signalbox
    assert routed.runner._resolve_injection_adapter("signal", restored) is routed.signalbox


def test_completion_delivery_ready_accepts_the_routed_restored_transport(routed):
    """The gate that stranded the ledger row: an unresolvable adapter reads as "not ready", so the
    watcher re-queued the completion forever without spending an attempt."""
    runner = routed.runner
    restored = _restored(Platform.SIGNAL, "G1", "planner")
    restore_identity(restored, runner=runner, transport_profile="signalbox")
    runner._build_process_event_source = lambda _evt: restored

    async def _deliver(_session_id, _session_key=None):
        return "deliver"

    runner._classify_completion_target = _deliver
    evt = {"type": "async_delegation", "parent_session_id": "sess-1",
           "session_key": "agent:planner:signal:group:G1"}
    assert asyncio.run(runner._completion_delivery_ready(evt)) is True


def test_identity_less_restored_source_still_resolves_by_runtime_profile(routed):
    """No pinned identity (a row written before the column existed): the runtime profile's own
    adapter is the answer, and a runtime profile without that bot stays fail-closed."""
    owner = _restored(Platform.SIGNAL, "G1", "signalbox")
    assert routed.runner._resolve_injection_adapter("signal", owner) is routed.signalbox
    stranger = _restored(Platform.SIGNAL, "G1", "planner")
    assert routed.runner._resolve_injection_adapter("signal", stranger) is None

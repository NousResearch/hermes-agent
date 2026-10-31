"""Receiver config, not launch/runtime config, decides unauthorized Signal DMs."""
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent import secret_scope
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner, _profile_runtime_scope
from plugins.platforms.signal.adapter import SignalAdapter


def config_for(rung, behavior):
    config = GatewayConfig(multiplex_profiles=True)
    extra = {}
    if rung == "global":
        config.unauthorized_dm_behavior = behavior
    elif rung == "platform":
        extra["unauthorized_dm_behavior"] = behavior
    else:
        extra["dm_policy"] = behavior
    config.platforms[Platform.SIGNAL] = PlatformConfig(enabled=True, extra={
        "http_url": "http://localhost:8080", "account": "+15500000000", **extra,
    })
    return config


@pytest.fixture
def ingress(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    homes = {"default": home, "groupbot": home / "profiles" / "groupbot",
             "target": home / "profiles" / "target"}
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    for directory in homes.values():
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".env").write_text("")
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda **kwargs: list(homes.items()))
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: homes[name])
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in homes)

    def make(launch, receiving, routed, receiver="groupbot"):
        host = object.__new__(GatewayRunner)
        host._primary_profile_name = "default"
        host.config = launch if receiver != "default" else receiving
        host.config.profile_routes = parse_profile_routes([
            {"name": "signal-route", "platform": "signal", "profile": "target", "bot_profile": receiver},
        ] if routed else [])
        host._profile_configs = {"groupbot": receiving, "target": config_for("global", "pair")}
        host.pairing_stores = {}
        for name, directory in homes.items():
            with _profile_runtime_scope(directory):
                host.pairing_stores[name] = PairingStore()
        host.pairing_store = host.pairing_stores["default"]
        with _profile_runtime_scope(homes[receiver]):
            adapter = SignalAdapter(receiving.platforms[Platform.SIGNAL])
        adapter.gateway_runner = host
        adapter.send = AsyncMock()
        host.adapters = {Platform.SIGNAL: adapter} if receiver == "default" else {}
        host._profile_adapters = {"target": {}, "groupbot": {}}
        if receiver != "default":
            adapter.set_owner_profile(receiver)
            host._profile_adapters[receiver][Platform.SIGNAL] = adapter
        verdicts = []

        async def admit(event):
            verdicts.append(await host._hm_admit_event(event))

        host._handle_message = admit  # Real ingress; no agent turn or live network.
        adapter.handle_message = (host._make_default_profile_message_handler() if receiver == "default"
                                  else host._make_profile_message_handler(receiver))

        async def deliver():
            with _profile_runtime_scope(homes["target"]):
                await adapter._handle_envelope({"sourceNumber": "alice", "dataMessage": {"message": "hello"}})
            assert verdicts[-1] is None  # Pairing is not admission to the agent.

        return host, adapter, deliver

    return make


@pytest.mark.asyncio
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("receiver", ["default", "groupbot"])
@pytest.mark.parametrize("rung,behavior", [
    ("global", "ignore"), ("global", "decline"),
    ("platform", "ignore"), ("platform", "decline"), ("dm_policy", "disabled"),
])
async def test_receiver_restrictions_prevent_pairing(ingress, rung, behavior, routed, receiver):
    launch = config_for("global", "pair")
    launch.unauthorized_dm_decline_message = "Wrong launch-profile refusal"
    receiving = config_for(rung, behavior)
    receiving.unauthorized_dm_decline_message = "Receiver refuses this DM"
    host, adapter, deliver = ingress(launch, receiving, routed, receiver)
    await deliver()
    await deliver()  # Decline is deduplicated; ignore stays silent.
    assert all(store.list_pending("signal") == [] for store in host.pairing_stores.values())
    if behavior == "decline":
        adapter.send.assert_awaited_once_with("alice", "Receiver refuses this DM")
    else:
        adapter.send.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("rung,behavior", [("global", "pair"), ("platform", "pair"), ("dm_policy", "pairing")])
async def test_receiver_pairing_is_not_silenced_by_launch_config(ingress, rung, behavior, routed):
    host, adapter, deliver = ingress(config_for("global", "ignore"), config_for(rung, behavior), routed)
    await deliver()
    adapter.send.assert_awaited_once()
    pending = [request for store in host.pairing_stores.values() for request in store.list_pending("signal")]
    assert len(pending) == 1
    assert pending[0]["user_id"] == "alice"
    assert "pairing approve signal" in adapter.send.await_args.args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("routed", [False, True])
async def test_missing_receiver_config_never_borrows_launch_pairing(ingress, routed):
    host, adapter, deliver = ingress(config_for("global", "pair"), config_for("global", "ignore"), routed)
    del host._profile_configs["groupbot"]
    await deliver()
    adapter.send.assert_not_awaited()
    assert all(store.list_pending("signal") == [] for store in host.pairing_stores.values())

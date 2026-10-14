"""Unauthorized Signal replies follow the receiving bot, not the routed runtime."""
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent import secret_scope
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.platforms.signal import SignalAdapter
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner, _profile_runtime_scope


@pytest.mark.asyncio
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("receiver", ["default", "groupbot"])
async def test_group_only_ingress_never_offers_pairing(tmp_path, monkeypatch, routed, receiver):
    # Real profile files and runtime scopes; no live gateway, credentials or network.
    home = tmp_path / ".hermes"
    homes = {"default": home, "groupbot": home / "profiles" / "groupbot",
             "target": home / "profiles" / "target"}
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    for name, directory in homes.items():
        directory.mkdir(parents=True, exist_ok=True)
        settings = 'SIGNAL_ALLOWED_USERS=""\nSIGNAL_GROUP_ALLOWED_USERS=abc\n' if name == receiver else ''
        (directory / ".env").write_text(settings)
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda **kwargs: list(homes.items()))
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: homes[name])
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in homes)

    host = object.__new__(GatewayRunner)
    host._primary_profile_name = "default"
    host.config = GatewayConfig(multiplex_profiles=True)
    host.config.profile_routes = parse_profile_routes([
        {"name": "routed-signal", "platform": "signal", "profile": "target", "bot_profile": receiver},
    ] if routed else [])
    host.pairing_stores = {}
    for name, directory in homes.items():
        with _profile_runtime_scope(directory):
            host.pairing_stores[name] = PairingStore()
    host.pairing_store = host.pairing_stores["default"]
    with _profile_runtime_scope(homes[receiver]):
        adapter = SignalAdapter(PlatformConfig(enabled=True, extra={
            "http_url": "http://localhost:8080", "account": "+15550000000",
        }))
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

    host._handle_message = admit  # Stop after real ingress, before an LLM turn.
    handler = (host._make_default_profile_message_handler() if receiver == "default"
               else host._make_profile_message_handler(receiver))
    adapter.handle_message = handler

    # A -> B -> A: a permissive ambient runtime must not alter the receiving bot's rules.
    for ambient in (receiver, "target", receiver):
        with _profile_runtime_scope(homes[ambient]):
            await adapter._handle_envelope({"sourceNumber": "alice", "dataMessage": {"message": "hello"}})
    assert verdicts == [None, None, None]
    adapter.send.assert_not_awaited()
    assert all(store.list_pending("signal") == [] for store in host.pairing_stores.values())
    assert not list(home.rglob("signal-pending.json"))

    with _profile_runtime_scope(homes[receiver]):
        for group in ("abc", "other"):
            await adapter._handle_envelope({"sourceNumber": "alice", "dataMessage": {
                "message": "hello", "groupInfo": {"groupId": group},
            }})
    assert len(verdicts) == 4  # The unlisted group was rejected at adapter intake.
    assert verdicts[-1] is not None
    assert verdicts[-1][1].chat_id == "group:abc"
    adapter.send.assert_not_awaited()

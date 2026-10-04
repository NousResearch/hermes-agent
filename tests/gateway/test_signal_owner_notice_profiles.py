"""Owner notices follow the receiving Signal transport, never a routed runtime."""
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from agent import secret_scope
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.signal import SignalAdapter
from gateway.profile_routing import parse_profile_routes
from gateway.run import GatewayRunner, _profile_runtime_scope


@pytest.mark.asyncio
@pytest.mark.parametrize("secondary_home", [True, False])
async def test_routed_stranger_notifies_only_receiving_owner(tmp_path, monkeypatch, secondary_home):
    home = tmp_path / ".hermes"
    homes = {"default": home, "groupbot": home / "profiles" / "groupbot",
             "target": home / "profiles" / "target"}
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", True)
    for name, directory in homes.items():
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".env").write_text(
            'SIGNAL_ALLOWED_USERS=""\nSIGNAL_GROUP_ALLOWED_USERS=abc\n' if name != "target" else "")
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda **kwargs: list(homes.items()))
    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", lambda name: homes[name])
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name in homes)

    host = object.__new__(GatewayRunner)
    host._primary_profile_name = "default"
    host.config = GatewayConfig(multiplex_profiles=True)
    host.config.profile_routes = parse_profile_routes([
        {"name": "route-" + receiver, "platform": "signal", "profile": "target", "bot_profile": receiver}
        for receiver in ("default", "groupbot")
    ])
    host.adapters = {}
    host._profile_adapters = {"target": {}, "groupbot": {}}
    host._profile_configs = {"groupbot": GatewayConfig(), "target": GatewayConfig()}
    adapters = {}
    verdicts = []

    async def admit(event):
        assert event.source.profile == "target"
        verdicts.append(await host._hm_admit_event(event))

    host._handle_message = admit
    for receiver in ("default", "groupbot"):
        config = PlatformConfig(enabled=True, extra={"http_url": "http://localhost:8080", "account": "test-account"})
        if receiver == "default" or secondary_home:
            config.home_channel = HomeChannel(platform=Platform.SIGNAL, chat_id="owner-" + receiver, name="Test owner")
        with _profile_runtime_scope(homes[receiver]):
            adapter = SignalAdapter(config)
        adapter.gateway_runner = host
        adapter.send = AsyncMock()
        adapters[receiver] = adapter
        if receiver == "default":
            host.config.platforms[Platform.SIGNAL] = config
            host.adapters[Platform.SIGNAL] = adapter
            adapter.handle_message = host._make_default_profile_message_handler()
        else:
            host._profile_configs[receiver].platforms[Platform.SIGNAL] = config
            adapter.set_owner_profile(receiver)
            host._profile_adapters[receiver][Platform.SIGNAL] = adapter
            adapter.handle_message = host._make_profile_message_handler(receiver)

    # Same stranger, two bots, and A -> B -> A ambient scopes: dedupe belongs to the receiver.
    for receiver in ("groupbot", "default", "groupbot", "default"):
        for ambient in (receiver, "target", receiver):
            with _profile_runtime_scope(homes[ambient]):
                await adapters[receiver]._handle_envelope({
                    "sourceNumber": "stranger", "sourceName": "Example", "dataMessage": {"message": "hello"},
                })
    assert verdicts == [None] * 12
    for receiver, adapter in adapters.items():
        if receiver == "groupbot" and not secondary_home:
            adapter.send.assert_not_awaited()
            continue
        adapter.send.assert_awaited_once()
        call = adapter.send.await_args
        assert call.args[0] == "owner-" + receiver
        assert "stranger" in call.args[1]
        expected_home = "~/.hermes" if receiver == "default" else "~/.hermes/profiles/groupbot"
        assert expected_home + "/.env" in call.args[1]
        assert "/profiles/target" not in call.args[1]
    assert not list(home.rglob("signal-pending.json"))

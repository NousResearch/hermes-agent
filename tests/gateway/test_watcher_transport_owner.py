"""Restored watcher notices use the bot that received the original Discord turn."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.session_identity import identity_of, resolve_identity, restore_identity


def _adapter():
    async def admit(event):
        event._gateway_accepted = True

    return SimpleNamespace(
        send=AsyncMock(), handle_message=AsyncMock(side_effect=admit), _active_sessions={},
    )


def _profile_home(monkeypatch, tmp_path):
    import hermes_cli.profiles as profiles

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(GatewayRunner, "_VOICE_MODE_PATH", tmp_path / "gateway_voice_mode.json")
    research_home = tmp_path / "profiles" / "research"
    research_home.mkdir(parents=True)
    (research_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: tmp_path if name == "default" else research_home)
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name in {"default", "research"})


@pytest.mark.asyncio
@pytest.mark.parametrize("default_online", [True, False])
@pytest.mark.parametrize("default_transport", ["discord", "relay"])
async def test_restored_watcher_uses_persisted_default_bot_not_research_bot(
    monkeypatch, tmp_path, default_online, default_transport,
):
    import gateway.run as gateway_run

    _profile_home(monkeypatch, tmp_path)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    config = GatewayConfig(multiplex_profiles=True, sessions_dir=tmp_path / "sessions")
    before = GatewayRunner(config)
    origin = SessionSource(
        platform=Platform.DISCORD, chat_id="channel", chat_type="thread",
        thread_id="thread", user_id="person", profile="research",
    )
    resolve_identity(origin, runner=before, transport_profile="default")
    spawned_in = before.session_store.get_or_create_session(origin)
    assert spawned_in.transport_profile == "default"
    assert before.session_store._db_for_key(spawned_in.session_key).get_session(spawned_in.session_id)

    runner = GatewayRunner(config)
    primary, research = _adapter(), _adapter()
    if default_transport == "relay":
        primary.fronts_platform = lambda platform: platform is Platform.DISCORD
        primary.send_for_platform = AsyncMock()
    adapter_platform = Platform.RELAY if default_transport == "relay" else Platform.DISCORD
    runner.adapters = {adapter_platform: primary} if default_online else {}
    runner._profile_adapters = {"research": {Platform.DISCORD: research}}
    restored = runner.session_store.lookup_by_session_key(spawned_in.session_key)
    assert restored is not None and restored.transport_profile == "default"
    from hermes_state import AsyncSessionDB
    runner._session_db = AsyncSessionDB(runner.session_store._db_for_key(spawned_in.session_key))
    restored_source = runner._restored_source(restored)
    assert runner._transport_owner(restored_source) is None
    assert identity_of(restored_source).transport_profile == "default"
    assert restored_source.profile == "research"

    watcher = {
        "session_id": "proc_restored", "session_key": spawned_in.session_key,
        "parent_session_id": spawned_in.session_id, "platform": "discord",
        "chat_type": "thread", "chat_id": "channel", "thread_id": "thread",
    }
    process = SimpleNamespace(
        parent_session_id=spawned_in.session_id, session_key=spawned_in.session_key,
        owner_task_id="owner", task_id="owner",
    )
    primary._active_sessions[spawned_in.session_key] = object()

    assert await runner._launching_turn_active("discord", watcher) is default_online
    assert await runner._watcher_message_route_owned(watcher, process)
    await runner._send_watcher_message("discord", "channel", "thread", "status", watcher, process)
    assert primary.send.await_count == int(default_online and default_transport == "discord")
    if default_transport == "relay":
        assert primary.send_for_platform.await_count == int(default_online)
        if default_online:
            assert primary.send_for_platform.await_args.args[:3] == ("discord", "channel", "status")
    research.send.assert_not_awaited()

    delivered = await runner._inject_watch_notification("finished", {**watcher, "type": "completion"})
    assert delivered is default_online
    assert primary.handle_message.await_count == int(default_online)
    research.handle_message.assert_not_awaited()


@pytest.mark.parametrize("relay_online", [True, False])
def test_pinned_transport_resolves_relay_alias_only_on_receiving_bot(
    monkeypatch, tmp_path, relay_online,
):
    _profile_home(monkeypatch, tmp_path)
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=True, sessions_dir=tmp_path / "sessions"))
    relay = SimpleNamespace(fronts_platform=lambda platform: platform is Platform.DISCORD)
    research = _adapter()
    runner.adapters = {Platform.RELAY: relay} if relay_online else {}
    runner._profile_adapters = {"research": {Platform.DISCORD: research}}
    source = SessionSource(platform=Platform.DISCORD, chat_id="channel", profile="research")
    restore_identity(source, runner=runner, transport_profile="default")

    assert runner._resolve_injection_adapter("discord", source) is (relay if relay_online else None)

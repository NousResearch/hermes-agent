"""Discord pending replay proves current native authority before cached media."""

from copy import deepcopy
import sys
import base64
from pathlib import Path
from unittest.mock import AsyncMock
from typing import Any

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.shutdown_pending import PendingQueueSnapshot
from gateway.shutdown_pending_codec import decode_pending_event


@pytest.fixture
def native_discord(monkeypatch):
    for module in list(sys.modules):
        if (
            module == "discord"
            or module.startswith("discord.")
            or module.startswith("plugins.platforms.discord")
        ):
            monkeypatch.delitem(sys.modules, module, raising=False)
    discord = pytest.importorskip("discord")
    return discord


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "media,state",
    [
        (False, "current"),
        (True, "current"),
        (True, "changed-cache"),
        (False, "deleted"),
        (False, "wrong-author"),
        (False, "edited"),
        (True, "ignored-channel"),
        (True, "missing-mention"),
        (True, "revoked-user"),
    ],
)
async def test_restored_discord_input_requires_current_native_source_and_cache(
    monkeypatch, native_discord, media, state
):
    discord = native_discord
    from discord.ext import commands
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv("DISCORD_ALLOW_ALL_USERS", "true")
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="test",
            extra={
                "require_mention": False,
                "auto_thread": False,
                "history_backfill": False,
            },
        )
    )
    runner.adapters[Platform.DISCORD] = adapter
    adapter.gateway_runner = runner
    adapter._client = commands.Bot(
        command_prefix="!", intents=discord.Intents.default()
    )
    await adapter._client._async_setup_hook()
    connection = adapter._client._connection
    connection.user = discord.ClientUser(
        state=connection,
        data={"id": "900", "username": "hermes", "discriminator": "0", "avatar": None},
    )
    guild = discord.Guild(
        data={"id": "999", "name": "test", "owner_id": "333"}, state=connection
    )
    connection._guilds[guild.id] = guild
    channel_data = {
        "id": "555",
        "guild_id": "999",
        "type": 0,
        "name": "allowed",
        "position": 0,
        "permission_overwrites": [],
    }
    channel = discord.TextChannel(state=connection, guild=guild, data=channel_data)
    guild._channels[channel.id] = channel
    data: dict[str, Any] = {
        "id": "101",
        "type": 0,
        "content": "authored input",
        "attachments": [],
        "embeds": [],
        "mentions": [],
        "mention_roles": [],
        "author": {
            "id": "333",
            "username": "sender",
            "discriminator": "0",
            "avatar": None,
        },
    }
    if media:
        data["attachments"] = [
            {
                "id": "777",
                "filename": "image.png",
                "size": 8,
                "url": "https://cdn.discordapp.com/original.png",
                "proxy_url": "https://media.discordapp.net/original.png",
                "content_type": "image/png",
                "width": 1,
                "height": 1,
            }
        ]
        monkeypatch.setattr(
            connection.http,
            "get_from_cdn",
            AsyncMock(
                return_value=base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
                )
            ),
        )
    message = discord.Message(state=connection, channel=channel, data=deepcopy(data))
    try:
        prepared = await adapter._prepare_inbound_event(message)
        assert prepared is not None
        event = prepared.event
        record = PendingQueueSnapshot.capture(
            runner._session_key_for_source(event.source), [event]
        ).events[0]
        restored = decode_pending_event(record, adapter=adapter)
        current = deepcopy(data)
        if state == "wrong-author":
            current["author"]["id"] = "444"
        if state == "edited":
            current["content"] = "changed authored input"
        if state == "changed-cache":
            assert event.media_urls
            Path(event.media_urls[0]).write_bytes(b"changed")
        if state == "ignored-channel":
            adapter.config.extra["ignored_channels"] = ["555"]
        if state == "missing-mention":
            adapter.config.extra["require_mention"] = True
        if state == "revoked-user":
            adapter._allowed_user_ids = {"444"}
        from gateway.pending_native import PendingNativeInput

        checks = []
        original_check = PendingNativeInput.attachments_available

        def check_available(native, paths):
            checks.append(tuple(paths))
            return original_check(native, paths)

        monkeypatch.setattr(
            PendingNativeInput, "attachments_available", check_available
        )
        requests = []

        async def get_channel(channel_id):
            requests.append(("channel", str(channel_id)))
            return deepcopy(channel_data)

        async def get_message(channel_id, message_id):
            requests.append(("message", str(channel_id), str(message_id)))
            if state == "deleted":
                raise discord.NotFound(
                    type("Response", (), {"status": 404, "reason": "Not Found"})(),
                    {"code": 10008, "message": "Unknown Message"},
                )
            return deepcopy(current)

        async def get_member(guild_id, member_id):
            return {
                "user": deepcopy(current["author"]),
                "roles": [],
                "joined_at": "2026-10-01T00:00:00+00:00",
                "flags": 0,
            }

        monkeypatch.setattr(connection.http, "get_member", get_member)
        monkeypatch.setattr(connection.http, "get_channel", get_channel)
        monkeypatch.setattr(connection.http, "get_message", get_message)
        verified = await adapter.revalidate_pending_event(restored)
        actual = (
            None
            if verified is None
            else (
                verified.text,
                verified.media_urls,
                verified.message_type,
                verified.source.chat_id,
                verified.source.user_id,
                verified.source.scope_id,
            )
        )
        expected = (
            None
            if state != "current"
            else (event.text, event.media_urls, event.message_type, "555", "333", "999")
        )
        expected_checks = (
            [tuple(event.media_urls)]
            if media and state in {"current", "changed-cache"}
            else []
        )
        assert (actual, checks) == (expected, expected_checks)
    finally:
        await adapter._client.close()
        runner.session_store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role_state", ["current", "revoked", "revoked-with-stale-member"]
)
async def test_role_restoration_uses_current_member_and_actual_profile_receipt(
    tmp_path, monkeypatch, native_discord, role_state
):
    import asyncio
    import hermes_state
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from gateway.input_owner import gateway_input_owner
    from gateway.pending_execution import consume_pending_execution
    from gateway.profile_routing import ProfileRoute
    from gateway.run import _profile_runtime_scope
    from hermes_constants import get_hermes_home
    from utils import atomic_json_write

    discord = native_discord
    from discord.ext import commands
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    launch = tmp_path / ".hermes"
    launch.mkdir()
    (launch / ".env").write_text("DISCORD_ALLOW_ALL_USERS=false\n")
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.delenv("DISCORD_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)
    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    homes = {profile: launch / "profiles" / profile for profile in ("a", "b")}
    for home in homes.values():
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("{}\n")
        (home / ".env").write_text("DISCORD_ALLOW_ALL_USERS=false\n")
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="test",
            extra={
                "require_mention": False,
                "auto_thread": False,
                "history_backfill": False,
            },
        )
    )
    adapter._allowed_role_ids = {666}
    runner = GatewayRunner(
        GatewayConfig(
            multiplex_profiles=True,
            profile_routes=[
                ProfileRoute(
                    name=profile,
                    platform="discord",
                    profile=profile,
                    chat_id=channel_id,
                )
                for profile, channel_id in (("a", "555"), ("b", "556"))
            ],
        )
    )
    runner.adapters[Platform.DISCORD] = adapter
    runner._profile_adapters = {profile: {} for profile in homes}
    adapter.gateway_runner = runner
    adapter._mark_connected()
    adapter._client = commands.Bot(
        command_prefix="!", intents=discord.Intents.default()
    )
    await adapter._client._async_setup_hook()
    connection = adapter._client._connection
    connection.user = discord.ClientUser(
        state=connection,
        data={"id": "900", "username": "hermes", "discriminator": "0", "avatar": None},
    )
    guild = discord.Guild(
        data={
            "id": "999",
            "name": "test",
            "owner_id": "333",
            "roles": [
                {
                    "id": "999",
                    "name": "@everyone",
                    "permissions": "0",
                    "position": 0,
                    "color": 0,
                },
                {
                    "id": "666",
                    "name": "allowed",
                    "permissions": "0",
                    "position": 1,
                    "color": 0,
                },
            ],
        },
        state=connection,
    )
    connection._guilds[guild.id] = guild
    channels = {}
    for channel_id in ("555", "556"):
        data = {
            "id": channel_id,
            "guild_id": "999",
            "type": 0,
            "name": "allowed",
            "position": 0,
            "permission_overwrites": [],
        }
        channel = discord.TextChannel(state=connection, guild=guild, data=data)
        guild._channels[channel.id] = channel
        channels[channel_id] = (channel, data)
    native_messages = {}
    reads = []
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    monkeypatch.setattr(connection.http, "get_from_cdn", AsyncMock(return_value=png))
    from gateway.pending_native import PendingNativeInput

    cache_checks = []
    original_check = PendingNativeInput.attachments_available

    def check_available(native, paths):
        cache_checks.append((get_hermes_home(), tuple(paths)))
        return original_check(native, paths)

    monkeypatch.setattr(PendingNativeInput, "attachments_available", check_available)

    async def get_channel(channel_id):
        return deepcopy(channels[str(channel_id)][1])

    async def get_message(channel_id, message_id):
        reads.append((str(message_id), get_hermes_home()))
        return deepcopy(native_messages[str(message_id)])

    async def get_member(guild_id, member_id):
        return {
            "user": {
                "id": "333",
                "username": "sender",
                "discriminator": "0",
                "avatar": None,
            },
            "roles": ["666"] if role_state == "current" else [],
            "joined_at": "2026-10-01T00:00:00+00:00",
            "flags": 0,
        }

    monkeypatch.setattr(connection.http, "get_channel", get_channel)
    monkeypatch.setattr(connection.http, "get_message", get_message)
    monkeypatch.setattr(connection.http, "get_member", get_member)
    seen = []

    async def receive(event):
        key = runner._session_key_for_source(event.source)
        entry = runner.session_store.lookup_by_session_key(key)
        assert entry is not None
        seen.append((
            event.message_id,
            get_hermes_home(),
            event.source.role_authorized,
            Path(event.media_urls[0]).read_bytes(),
        ))
        db = runner.session_store._db_for_session_id(entry.session_id)
        db.append_message(
            entry.session_id,
            "user",
            event.text,
            display_metadata={
                "gateway_input_owner": gateway_input_owner(event, event.source)
            },
        )
        consume_pending_execution(runner, event)
        return None

    runner._wire_adapter_handlers(adapter, message_handler=receive)
    monkeypatch.setattr(runner, "_await_startup_warmup", AsyncMock())
    active = is_multiplex_active()
    set_multiplex_active(True)
    expected = []
    expected_reads = []
    expected_checks = []
    retained = []
    try:
        for index, profile in enumerate(("a", "b", "a")):
            message_id = str(101 + index)
            channel = channels["555" if profile == "a" else "556"][0]
            data = {
                "id": message_id,
                "type": 0,
                "content": f"authored-{index}",
                "attachments": [
                    {
                        "id": str(777 + index),
                        "filename": "image.png",
                        "size": len(png),
                        "url": f"https://cdn.discordapp.com/{index}.png",
                        "proxy_url": f"https://media.discordapp.net/{index}.png",
                        "content_type": "image/png",
                        "width": 1,
                        "height": 1,
                    }
                ],
                "embeds": [],
                "mentions": [],
                "mention_roles": [],
                "author": {
                    "id": "333",
                    "username": "sender",
                    "discriminator": "0",
                    "avatar": None,
                },
                "member": {
                    "roles": ["666"],
                    "joined_at": "2026-10-01T00:00:00+00:00",
                    "flags": 0,
                },
            }
            native_messages[message_id] = data
            with _profile_runtime_scope(homes[profile]):
                message = discord.Message(
                    state=connection, channel=channel, data=deepcopy(data)
                )
                if role_state == "revoked-with-stale-member":
                    guild._add_member(message.author)
                admitted, role_authorized = adapter._discord_source_admission(message)
                assert (admitted, role_authorized) == (True, True)
                prepared = await adapter._prepare_inbound_event(
                    message, role_authorized
                )
                assert prepared is not None
                event = prepared.event
                assert adapter._canonicalize(event.source) is not None
                entry = runner.session_store.get_or_create_session(event.source)
                snapshot = PendingQueueSnapshot.capture(entry.session_key, [event])
                path = homes[profile] / "pending_messages" / f"pending-{index}.json"
                path.parent.mkdir(exist_ok=True)
                atomic_json_write(path, snapshot.to_payload(), mode=0o600)
            runner._startup_restore_in_progress = True
            await runner._finish_startup_restore()
            if tasks := list(adapter._background_tasks):
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
            retained.append((message_id, homes[profile]))
            expected_reads.extend(
                sorted(retained, key=lambda item: (str(item[1]), item[0]))
                if role_state != "current"
                else [(message_id, homes[profile])]
            )
            if role_state == "current":
                expected.append((message_id, homes[profile], True, png))
                expected_checks.append((homes[profile], tuple(event.media_urls)))
            assert (seen, reads, cache_checks, path.exists(), get_hermes_home()) == (
                expected,
                expected_reads,
                expected_checks,
                role_state != "current",
                launch,
            )
    finally:
        set_multiplex_active(active)
        await adapter._client.close()
        runner.session_store.close_all_db_handles()

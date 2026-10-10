"""Native deletion notifications withdraw queued input without trusting a deleting actor."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from types import ModuleType
from datetime import datetime
from typing import TYPE_CHECKING, Literal
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.platforms.base import BasePlatformAdapter

if TYPE_CHECKING:
    from discord.types.guild import Guild as GuildPayload
    from discord.types.threads import Thread as ThreadPayload
    from discord.types.gateway import MessageDeleteEvent, MessageDeleteBulkEvent


@pytest.fixture(scope="module", autouse=True)
def _use_installed_discord_sdk():
    with pytest.MonkeyPatch.context() as patch:
        for name in ("discord", "discord.ext", "discord.ext.commands"):
            if name in sys.modules and not isinstance(sys.modules[name], ModuleType):
                patch.delitem(sys.modules, name)
        yield


def test_native_deletion_domain_imports_before_platform_modules():
    result = subprocess.run(
        [sys.executable, "-c", (
            "from gateway.native_message_deletion import NativeMessageDeletion; "
            "from gateway.platforms.event import MessageEvent"
        )],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")


def _guild_payload() -> GuildPayload:
    return {
        "id": "999", "name": "test", "owner_id": "333", "icon": None,
        "splash": None, "discovery_splash": None, "emojis": [], "stickers": [],
        "features": [], "description": None, "incidents_data": None,
        "region": "", "afk_channel_id": None, "afk_timeout": 60,
        "verification_level": 0, "default_message_notifications": 0,
        "explicit_content_filter": 0, "roles": [], "mfa_level": 0,
        "nsfw_level": 0, "application_id": None, "system_channel_id": None,
        "system_channel_flags": 0, "rules_channel_id": None,
        "vanity_url_code": None, "banner": None, "premium_tier": 0,
        "preferred_locale": "en-US", "public_updates_channel_id": None,
        "stage_instances": [], "guild_scheduled_events": [],
    }


def _thread_payload(thread_id: str, *, name: str, archive_duration: Literal[60, 1440]) -> ThreadPayload:
    return {
        "id": thread_id, "guild_id": "999", "type": 11, "name": name,
        "parent_id": "444", "owner_id": "333", "message_count": 0,
        "member_count": 1, "total_message_sent": 0, "rate_limit_per_user": 0,
        "thread_metadata": {"archived": False, "auto_archive_duration": archive_duration,
                            "archive_timestamp": "2026-10-02T00:00:00+00:00", "locked": False},
    }


def _pending(adapter: BasePlatformAdapter, platform: Platform, message_id: str, *,
             scope: str | None = "999", chat: str = "555", thread: str | None = None) -> MessageEvent:
    return MessageEvent(
        text=f"input {message_id}", message_id=message_id,
        source=adapter.build_source(
            chat_id=chat, chat_type="thread" if thread else ("group" if scope else "dm"), user_id="original-author",
            scope_id=scope, thread_id=thread), timestamp=datetime(2026, 10, 2))


@pytest.mark.asyncio
@pytest.mark.parametrize("platform,bulk,thread,scope,notification", [
    (Platform.DISCORD, False, None, "999", "valid"),
    (Platform.DISCORD, True, "555", "999", "valid"),
    (Platform.DISCORD, False, None, None, "valid"),
    (Platform.DISCORD, False, None, "999", "wrong"),
    (Platform.SLACK, False, None, "999", "valid"),
    (Platform.SLACK, False, "root-ts", "999", "valid"),
    (Platform.SLACK, False, None, "999", "wrong"),
    (Platform.SLACK, False, None, "999", "missing"),
    (Platform.SLACK, False, None, "999", "conflicting"),
])
async def test_registered_native_deletion_preserves_other_scopes(platform, bulk, thread, scope, notification, monkeypatch):
    if platform == Platform.DISCORD:
        pytest.importorskip("discord")
        import plugins.platforms.discord.adapter as module
        from discord.ext import commands
        adapter = module.DiscordAdapter(PlatformConfig(enabled=True, token="test"))
        other = module.DiscordAdapter(PlatformConfig(enabled=True, token="test-other"))
        adapter._slash_commands = False
        monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)
        monkeypatch.setattr(module, "_wait_for_ready_or_bot_exit", AsyncMock())
        async def wait_for_close(*args):
            await asyncio.Event().wait()
        monkeypatch.setattr(commands.Bot, "start", wait_for_close)
        monkeypatch.setattr(adapter, "_start_liveness_probe", lambda: None)
        monkeypatch.setattr(adapter, "_wire_plugin_handlers", lambda client: None)
        assert await adapter.connect()
        client = adapter._client
        assert client is not None
        await client._async_setup_hook()
        if thread:
            import discord
            state = client._connection
            guild = discord.Guild(data=_guild_payload(), state=state)
            state._guilds[guild.id] = guild
            guild._threads[555] = discord.Thread(guild=guild, state=state, data=_thread_payload("555", name="test-thread", archive_duration=60))
    else:
        pytest.importorskip("slack_bolt")
        from slack_bolt.async_app import AsyncApp
        from slack_bolt.authorization import AuthorizeResult
        from plugins.platforms.slack.adapter import SlackAdapter
        adapter = SlackAdapter(PlatformConfig(enabled=True, token="test"))
        other = SlackAdapter(PlatformConfig(enabled=True, token="test-other"))
        async def authorize(team_id, enterprise_id, logger):
            return AuthorizeResult(enterprise_id=enterprise_id, team_id=team_id, bot_token="test", bot_user_id="bot")
        app = AsyncApp(authorize=authorize, process_before_response=True)
        adapter._app = app
        adapter._team_clients = {"999": app.client}
        adapter._register_bolt_handlers()
    ids = ("101", "102") if bulk else ("101",)
    events = {
        "deleted": _pending(adapter, platform, "101", scope=scope, thread=thread),
        "next": _pending(adapter, platform, "102", scope=scope, thread=thread),
        "other-workspace": _pending(adapter, platform, "101", scope="elsewhere", thread=thread),
        "other-channel": _pending(adapter, platform, "101", scope=scope, chat="elsewhere", thread=thread),
        "other-adapter": _pending(other, platform, "101", scope=scope, thread=thread),
    }
    if thread:
        events["other-thread"] = _pending(adapter, platform, "101", scope=scope, thread="different-root")
    adapter._pending_messages.update(events)
    try:
        if platform == Platform.DISCORD:
            assert isinstance(adapter, module.DiscordAdapter)
            dispatched = []
            monkeypatch.setattr(client._connection, "dispatch", lambda *args: dispatched.append(args))
            if bulk:
                bulk_data: MessageDeleteBulkEvent = {"channel_id": "555", "ids": list(ids)}
                if scope is not None:
                    bulk_data["guild_id"] = scope if notification == "valid" else "888"
                client._connection.parse_message_delete_bulk(bulk_data)
            else:
                single_data: MessageDeleteEvent = {"channel_id": "555", "id": "101"}
                if scope is not None:
                    single_data["guild_id"] = scope if notification == "valid" else "888"
                client._connection.parse_message_delete(single_data)
            for event_name, payload in dispatched:
                callback = getattr(client, f"on_{event_name}", None)
                if callback:
                    await callback(payload)
        else:
            from slack_bolt.request.async_request import AsyncBoltRequest
            event = {"type": "message", "subtype": "message_deleted", "channel": "555", "deleted_ts": "101", "ts": "delete-ts"}
            if thread:
                event["previous_message"] = {"thread_ts": thread, "user": "someone-else"}
            body = {"type": "event_callback", "team_id": "999", "event_id": "delete-event", "event": event}
            if notification == "missing":
                body.pop("team_id")
            elif notification == "wrong":
                body["team_id"] = "unconfigured-workspace"
            elif notification == "conflicting":
                event["team"] = "conflicting-workspace"
            response = await app.async_dispatch(AsyncBoltRequest(body=body, mode="socket_mode"))
            assert response.status == 200
        expected = events if notification != "valid" else {
            key: event for key, event in events.items()
            if key != "deleted" and not (bulk and key == "next")}
        assert adapter._pending_messages == expected
    finally:
        if platform == Platform.DISCORD:
            assert isinstance(adapter, module.DiscordAdapter)
            await adapter._cancel_bot_task()
            await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("notification", ["single", "bulk", "wrong-guild", "wrong-parent", "thread-message"])
async def test_raw_deletion_uses_the_origin_channel_of_auto_thread_input(notification, monkeypatch):
    discord = pytest.importorskip("discord")
    import plugins.platforms.discord.adapter as module
    from discord.ext import commands

    adapter = module.DiscordAdapter(PlatformConfig(enabled=True, token="test", extra={
        "require_mention": False, "history_backfill": False, "auto_thread": True}))
    adapter._slash_commands = False
    adapter._text_batch_delay_seconds = 60
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)
    monkeypatch.setattr(module, "_wait_for_ready_or_bot_exit", AsyncMock())

    async def wait_for_close(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(commands.Bot, "start", wait_for_close)
    monkeypatch.setattr(adapter, "_start_liveness_probe", lambda: None)
    monkeypatch.setattr(adapter, "_wire_plugin_handlers", lambda client: None)
    assert await adapter.connect()
    client = adapter._client
    assert client is not None
    await client._async_setup_hook()
    state = client._connection
    guild = discord.Guild(data=_guild_payload(), state=state)
    state._guilds[guild.id] = guild
    parent = discord.TextChannel(state=state, guild=guild, data={
        "id": "444", "type": 0, "name": "parent", "position": 0, "permission_overwrites": []})
    guild._channels[parent.id] = parent
    thread_data = _thread_payload("101", name="new-thread", archive_duration=1440)
    create_thread = AsyncMock(return_value=thread_data)
    monkeypatch.setattr(state.http, "start_thread_with_message", create_thread)
    message = discord.Message(state=state, channel=parent, data={
        "id": "101", "type": 0, "content": "start a task", "attachments": [], "embeds": [],
        "mentions": [], "mention_roles": [], "author": {"id": "333", "username": "sender", "discriminator": "0", "avatar": None}})
    ordinary = MessageEvent(text="thread reply", message_id="202", source=adapter.build_source(
        chat_id="101", chat_type="thread", thread_id="101", parent_chat_id="444", guild_id="999", user_id="333"))
    adapter._pending_messages["ordinary"] = ordinary
    try:
        assert await adapter._handle_message(message)
        create_thread.assert_awaited_once()
        queued = next(iter(adapter._pending_text_batches.values()))
        origin = {
            "chat_id": queued.source.chat_id,
            "parent_chat_id": queued.source.parent_chat_id,
            "thread_id": queued.source.thread_id,
            "auto_thread_created": queued.source.auto_thread_created,
            "scope_id": queued.source.scope_id,
            "message_id": queued.message_id,
        }
        dispatched = []
        monkeypatch.setattr(state, "dispatch", lambda *args: dispatched.append(args))
        channel_id = "777" if notification == "wrong-parent" else "444"
        guild_id = "888" if notification == "wrong-guild" else "999"
        if notification == "bulk":
            bulk_data: MessageDeleteBulkEvent = {"channel_id": channel_id, "guild_id": guild_id, "ids": ["101"]}
            state.parse_message_delete_bulk(bulk_data)
        else:
            single_data: MessageDeleteEvent = {"channel_id": channel_id, "guild_id": guild_id,
                                              "id": "202" if notification == "thread-message" else "101"}
            state.parse_message_delete(single_data)
        for event_name, payload in dispatched:
            callback = getattr(client, f"on_{event_name}", None)
            if callback:
                await callback(payload)
        pending = [*adapter._pending_text_batches.values(), *adapter._pending_messages.values()]
        pending_ids = []
        for event in pending:
            assert event.message_id is not None
            pending_ids.append(event.message_id)
        assert {"origin": origin, "pending_ids": sorted(pending_ids)} == {
            "origin": {"chat_id": "101", "parent_chat_id": "444", "thread_id": "101",
                       "auto_thread_created": True, "scope_id": "999", "message_id": "101"},
            "pending_ids": ["202"] if notification in {"single", "bulk"} else ["101", "202"],
        }
    finally:
        for task in adapter._pending_text_batch_tasks.values():
            task.cancel()
        await asyncio.gather(*adapter._pending_text_batch_tasks.values(), return_exceptions=True)
        await adapter._cancel_bot_task()
        await client.close()

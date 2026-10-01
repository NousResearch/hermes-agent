"""Native deletion notifications withdraw queued input without trusting a deleting actor."""

import asyncio
import subprocess
import sys
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent


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


def _pending(adapter, platform, message_id, *, scope="999", chat="555", thread=None):
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
        adapter = module.DiscordAdapter(PlatformConfig(enabled=True, token="test"))
        other = module.DiscordAdapter(PlatformConfig(enabled=True, token="test-other"))
        adapter._slash_commands = False
        monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *args: True)
        monkeypatch.setattr(module, "_wait_for_ready_or_bot_exit", AsyncMock())
        async def wait_for_close(*args):
            await asyncio.Event().wait()
        monkeypatch.setattr(module.commands.Bot, "start", wait_for_close)
        monkeypatch.setattr(adapter, "_start_liveness_probe", lambda: None)
        monkeypatch.setattr(adapter, "_wire_plugin_handlers", lambda client: None)
        assert await adapter.connect()
        await adapter._client._async_setup_hook()
        if thread:
            import discord
            state = adapter._client._connection
            guild = discord.Guild(data={"id": "999", "name": "test", "owner_id": "333"}, state=state)
            state._guilds[guild.id] = guild
            guild._threads[555] = discord.Thread(guild=guild, state=state, data={
                "id": "555", "type": 11, "name": "test-thread", "parent_id": "444",
                "owner_id": "333", "message_count": 0, "member_count": 1,
                "thread_metadata": {"archived": False, "auto_archive_duration": 60,
                    "archive_timestamp": "2026-10-02T00:00:00+00:00", "locked": False}})
    else:
        pytest.importorskip("slack_bolt")
        from slack_bolt.async_app import AsyncApp
        from slack_bolt.authorization import AuthorizeResult
        from plugins.platforms.slack.adapter import SlackAdapter
        adapter = SlackAdapter(PlatformConfig(enabled=True, token="test"))
        other = SlackAdapter(PlatformConfig(enabled=True, token="test-other"))
        async def authorize(team_id, enterprise_id, logger):
            return AuthorizeResult(enterprise_id=enterprise_id, team_id=team_id, bot_token="test", bot_user_id="bot")
        adapter._app = AsyncApp(authorize=authorize, process_before_response=True)
        adapter._team_clients = {"999": adapter._app.client}
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
            dispatched = []
            monkeypatch.setattr(adapter._client._connection, "dispatch", lambda *args: dispatched.append(args))
            data = {"channel_id": "555"}
            if scope is not None:
                data["guild_id"] = scope if notification == "valid" else "888"
            if bulk:
                data["ids"] = list(ids)
                adapter._client._connection.parse_message_delete_bulk(data)
            else:
                data["id"] = "101"
                adapter._client._connection.parse_message_delete(data)
            for event_name, payload in dispatched:
                callback = getattr(adapter._client, f"on_{event_name}", None)
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
            response = await adapter._app.async_dispatch(AsyncBoltRequest(body=body, mode="socket_mode"))
            assert response.status == 200
        expected = events if notification != "valid" else {
            key: event for key, event in events.items()
            if key != "deleted" and not (bulk and key == "next")}
        assert adapter._pending_messages == expected
    finally:
        if platform == Platform.DISCORD:
            await adapter._cancel_bot_task()
            await adapter._client.close()

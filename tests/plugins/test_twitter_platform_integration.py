import json
import time
from unittest.mock import AsyncMock

import httpx
import pytest

from gateway.config import PlatformConfig


@pytest.mark.asyncio
async def test_twitter_plugin_end_to_end_with_profile_isolation(monkeypatch, tmp_path):
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import PluginManager
    from plugins.platforms.twitter.oauth import SCOPES, load_tokens, save_tokens

    home = tmp_path / "profile-a"
    monkeypatch.setenv("HERMES_HOME", str(home))
    save_tokens(
        {
            "access_token": "fake-token",
            "refresh_token": "fake-refresh",
            "expires_at": time.time() + 3600,
            "scopes": list(SCOPES),
            "client_id": "fake-client",
            "user_id": "7",
            "username": "bot",
        }
    )

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/2/users/me":
            return httpx.Response(200, json={"data": {"id": "7", "username": "bot"}})
        if request.url.path == "/2/users/7/mentions":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "101",
                            "author_id": "42",
                            "conversation_id": "100",
                            "text": "@bot hello",
                            "entities": {"mentions": [{"id": "7"}]},
                        }
                    ],
                    "includes": {"users": [{"id": "42", "username": "alice"}]},
                    "meta": {},
                },
            )
        if request.url.path == "/2/dm_events":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "501",
                            "event_type": "MessageCreate",
                            "sender_id": "42",
                            "dm_conversation_id": "42-7",
                            "text": "hello privately",
                        }
                    ],
                    "includes": {"users": [{"id": "42", "username": "alice"}]},
                    "meta": {},
                },
            )
        if request.url.path == "/2/tweets/search/recent":
            return httpx.Response(200, json={"data": [], "includes": {}})
        if request.url.path == "/2/tweets" and request.method == "POST":
            assert isinstance(json.loads(request.content)["text"], str)
            return httpx.Response(201, json={"data": {"id": "701"}})
        if request.url.path == "/2/dm_conversations/42-7/messages":
            return httpx.Response(201, json={"data": {"dm_event_id": "702"}})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    transport = httpx.MockTransport(handler)
    manager = PluginManager()
    manager.discover_and_load(force=True)
    entry = platform_registry.get("twitter")
    assert entry is not None

    config = PlatformConfig(
        extra={
            "client_id": "fake-client",
            "allow_all_users": True,
            "initial_backfill": 1,
            "poll_interval_seconds": 3600,
            "_http_transport": transport,
        }
    )
    adapter = entry.adapter_factory(config)
    adapter.handle_message = AsyncMock()

    assert await adapter.connect()
    assert [call.args[0].message_id for call in adapter.handle_message.await_args_list] == [
        "101",
        "501",
    ]
    public = await adapter.send("tweet:100:101", "public reply", reply_to="101")
    direct = await adapter.send("dm:42-7", "private reply")
    assert public.success and public.message_id == "701"
    assert direct.success and direct.message_id == "702"
    await adapter.disconnect()

    restarted = entry.adapter_factory(config)
    assert restarted._state.seen("101")
    assert restarted._state.seen("501")

    standalone = await entry.standalone_sender_fn(config, "timeline", "cron post")
    assert standalone == {"success": True, "message_id": "701"}
    assert calls.count(("POST", "/2/tweets")) == 2

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-b"))
    assert load_tokens() is None
    isolated = entry.adapter_factory(config)
    assert not isolated._state.seen("101")

import asyncio
import base64
import json
import stat
import time
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from gateway.config import PlatformConfig


READY_POLICY = {
    "ai_reply_approval_confirmed": True,
    "automated_label_confirmed": True,
    "human_operator_account_confirmed": True,
    "opt_out_keywords": ["stop"],
}
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.mark.asyncio
async def test_twitter_plugin_end_to_end_with_profile_isolation(monkeypatch, tmp_path):
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import PluginManager
    from plugins.platforms.twitter import oauth as oauth_module
    from plugins.platforms.twitter.client import XClient
    from plugins.platforms.twitter.oauth import (
        OAuthTokens,
        SCOPES,
        authorize,
        load_tokens,
        refresh_if_needed,
        save_tokens,
    )
    from plugins.platforms.twitter.state import TwitterState, state_path
    from tools.registry import registry
    from tools import url_safety

    profiles = {"public": tmp_path / "profile-public", "confidential": tmp_path / "profile-confidential"}
    writes = []
    refreshes = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal refreshes
        authorization = request.headers.get("Authorization", "")

        if request.url.path == "/2/oauth2/token":
            form = parse_qs(request.content.decode())
            if form["grant_type"] == ["refresh_token"]:
                refreshes += 1
                return httpx.Response(
                    200,
                    json={
                        "access_token": "public-refreshed",
                        "refresh_token": "public-refresh-rotated",
                        "expires_in": 3600,
                        "scope": " ".join(SCOPES),
                    },
                )
            if authorization.startswith("Basic "):
                assert authorization == "Basic Y2xpZW50LWNvbmZpZGVudGlhbDpzZWNyZXQ="
                assert "client_id" not in form and "client_secret" not in form
                access_token = "confidential-access"
            else:
                assert form["client_id"] == ["client-public"]
                access_token = "public-access"
            assert form["code_verifier"] == ["verifier"]
            return httpx.Response(
                200,
                json={
                    "access_token": access_token,
                    "refresh_token": f"{access_token}-refresh",
                    "expires_in": 3600,
                    "scope": " ".join(SCOPES),
                },
            )

        if request.url.path == "/2/users/me":
            if authorization == "Bearer confidential-access":
                return httpx.Response(200, json={"data": {"id": "8", "username": "otherbot"}})
            assert authorization in {"Bearer public-access", "Bearer public-refreshed"}
            return httpx.Response(200, json={"data": {"id": "7", "username": "bot"}})

        if request.url.path == "/2/users/7/mentions":
            since_id = request.url.params.get("since_id", "")
            if not since_id:
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "101",
                                "author_id": "42",
                                "conversation_id": "100",
                                "text": "@bot image",
                                "entities": {"mentions": [{"id": "7"}]},
                                "attachments": {"media_keys": ["3_1"]},
                            },
                            {
                                "id": "102",
                                "author_id": "42",
                                "conversation_id": "100",
                                "text": "@bot use this quote",
                                "entities": {"mentions": [{"id": "7"}]},
                                "referenced_tweets": [{"type": "quoted", "id": "900"}],
                            },
                            {
                                "id": "103",
                                "author_id": "42",
                                "conversation_id": "100",
                                "text": "quote only",
                                "referenced_tweets": [{"type": "quoted", "id": "900"}],
                            },
                        ],
                        "includes": {
                            "users": [{"id": "42", "username": "alice"}],
                            "media": [
                                {
                                    "media_key": "3_1",
                                    "type": "photo",
                                    "url": "https://media.test/image.png",
                                    "alt_text": "one pixel",
                                    "width": 1,
                                    "height": 1,
                                }
                            ],
                        },
                        "meta": {},
                    },
                )
            if since_id == "103":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "104",
                                "author_id": "44",
                                "conversation_id": "100",
                                "text": "branch follow-up",
                                "in_reply_to_user_id": "7",
                                "referenced_tweets": [{"type": "replied_to", "id": "701"}],
                            }
                        ],
                        "includes": {
                            "users": [{"id": "44", "username": "bob"}],
                            "tweets": [{"id": "701", "author_id": "7", "text": "first reply"}],
                        },
                        "meta": {},
                    },
                )
            assert since_id == "104"
            return httpx.Response(200, json={"data": [], "includes": {}, "meta": {}})

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
                        },
                        {
                            "id": "502",
                            "event_type": "MessageCreate",
                            "sender_id": "43",
                            "dm_conversation_id": "43-7",
                            "text": "stop",
                        },
                    ],
                    "includes": {
                        "users": [
                            {"id": "42", "username": "alice"},
                            {"id": "43", "username": "carol"},
                        ]
                    },
                    "meta": {},
                },
            )

        if request.url.path == "/2/tweets/search/recent":
            return httpx.Response(200, json={"data": [], "includes": {}})
        if request.url.path == "/2/tweets/701/quote_tweets":
            return httpx.Response(200, json={"data": [], "includes": {}})
        if request.url.path == "/2/tweets" and request.method == "GET":
            if "non_public_metrics" in request.url.params.get("tweet.fields", ""):
                return httpx.Response(
                    200,
                    json={"data": [{"id": "701", "public_metrics": {"like_count": 2}}]},
                )
            assert request.url.params["ids"] == "900"
            return httpx.Response(
                200,
                json={"data": [{"id": "900", "author_id": "55", "text": "quoted background"}]},
            )
        if request.url.path == "/2/users/7/bookmarks":
            return httpx.Response(200, json={"data": [{"id": "900"}]})
        if request.url.path == "/image.png":
            return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})
        if request.url.path == "/2/media/upload":
            body = json.loads(request.content)
            assert body["media_type"] == "image/png"
            return httpx.Response(201, json={"data": {"id": "801"}})
        if request.url.path == "/2/tweets" and request.method == "POST":
            body = json.loads(request.content)
            writes.append((request.url.path, body, authorization))
            post_id = str({1: 701, 2: 703, 3: 704}[sum(path == "/2/tweets" for path, _, _ in writes)])
            return httpx.Response(201, json={"data": {"id": post_id}})
        if request.url.path == "/2/dm_conversations/42-7/messages":
            writes.append((request.url.path, json.loads(request.content), authorization))
            return httpx.Response(201, json={"data": {"dm_event_id": "702"}})
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    async def callback(_redirect_uri, state, *, timeout, on_ready):
        assert state and timeout == 1
        on_ready()
        return "authorization-code"

    transport = httpx.MockTransport(handler)
    opened = []
    monkeypatch.setattr(oauth_module, "wait_for_callback", callback)
    monkeypatch.setattr(oauth_module, "create_pkce_pair", lambda: ("verifier", "challenge"))

    monkeypatch.setenv("HERMES_HOME", str(profiles["public"]))
    public_tokens = await authorize(
        "client-public",
        "http://127.0.0.1:8765/callback",
        client_type="public",
        timeout=1,
        open_url=opened.append,
        transport=transport,
    )
    assert public_tokens.user_id == "7" and public_tokens.client_type == "public"
    assert stat.S_IMODE((profiles["public"] / "twitter/oauth2.json").stat().st_mode) == 0o600

    monkeypatch.setenv("HERMES_HOME", str(profiles["confidential"]))
    confidential_tokens = await authorize(
        "client-confidential",
        "http://127.0.0.1:8765/callback",
        client_type="confidential",
        client_secret="secret",
        timeout=1,
        open_url=opened.append,
        transport=transport,
    )
    assert confidential_tokens.user_id == "8" and confidential_tokens.client_type == "confidential"
    assert stat.S_IMODE(
        (profiles["confidential"] / "twitter/oauth2.json").stat().st_mode
    ) == 0o600
    for url in opened:
        query = parse_qs(urlparse(url).query)
        assert query["code_challenge_method"] == ["S256"]
        assert query["scope"] == [" ".join(SCOPES)]

    monkeypatch.setenv("HERMES_HOME", str(profiles["public"]))
    assert load_tokens() == public_tokens
    monkeypatch.setattr(url_safety, "is_safe_url", lambda _url: True)

    manager = PluginManager()
    manager.discover_and_load(force=True)
    entry = platform_registry.get("twitter")
    assert entry is not None and entry.source == "plugin"
    assert registry.get_entry("twitter_bookmarks").is_async
    assert registry.get_entry("twitter_post_metrics").is_async

    config = PlatformConfig(
        extra={
            "client_id": "client-public",
            "allow_all_users": True,
            "policy": READY_POLICY,
            "initial_backfill": 10,
            "poll_interval_seconds": 3600,
            "_http_transport": transport,
        }
    )
    adapter = entry.adapter_factory(config)
    adapter.handle_message = AsyncMock()

    assert await adapter.connect()
    events = [call.args[0] for call in adapter.handle_message.await_args_list]
    assert [event.message_id for event in events] == ["101", "102", "501"]
    assert events[0].media_types == ["image/png"]
    assert "one pixel" in events[0].channel_context
    assert "quoted background" in events[1].channel_context
    assert events[2].source.chat_id == "dm:42-7"
    assert adapter._state.seen("103") is False
    assert adapter._state.seen("502")
    assert "43-7" in adapter._state.opted_out_dm_conversations

    public = await adapter.send("tweet:100:101", "**public** reply", reply_to="101")
    direct = await adapter.send("dm:42-7", "private reply", reply_to="501")
    assert public.success and public.message_id == "701"
    assert direct.success and direct.message_id == "702"
    assert writes[0][1]["text"] == "public reply"

    adapter.handle_message.reset_mock()
    await adapter._poll_mentions_once()
    branch_event = adapter.handle_message.await_args.args[0]
    assert branch_event.message_id == "104"
    assert branch_event.source.chat_id == "tweet:100:101"
    image_reply = await adapter.send(
        "tweet:100:101",
        "image reply",
        reply_to="104",
        metadata={"media_files": [events[0].media_urls[0]]},
    )
    duplicate = await adapter.send("tweet:100:101", "duplicate", reply_to="104")
    assert image_reply.success and image_reply.message_id == "703"
    assert not duplicate.success and "not eligible" in duplicate.error
    assert writes[2][1]["media"] == {"media_ids": ["801"]}
    await adapter.disconnect()

    restarted = entry.adapter_factory(config)
    assert restarted._state.resolve_anchor("999", ["701"]) == "101"
    assert restarted._state.mention_since_id == "104"
    assert restarted._state.dm_last_seen_event_id == "502"
    restarted.handle_message = AsyncMock()
    assert await restarted.connect(is_reconnect=True)
    restarted.handle_message.assert_not_awaited()
    replay = await restarted.send("tweet:100:101", "replay", reply_to="101")
    assert not replay.success and "not eligible" in replay.error
    await restarted.disconnect()

    bookmark_tool = registry.get_entry("twitter_bookmarks")
    metrics_tool = registry.get_entry("twitter_post_metrics")
    discovered_tools = bookmark_tool.handler.__globals__
    monkeypatch.setitem(
        discovered_tools,
        "XClient",
        lambda *, token: XClient(token=token, transport=transport),
    )
    bookmarks = json.loads(await bookmark_tool.handler({"operation": "list"}))
    metrics = json.loads(
        await metrics_tool.handler({"post_ids": ["701", "703"]})
    )
    assert bookmarks == {"data": [{"id": "900"}]}
    assert metrics["data"][0]["public_metrics"]["like_count"] == 2

    save_tokens(
        OAuthTokens(
            access_token="public-expired",
            refresh_token="public-access-refresh",
            expires_at=time.time() - 1,
            scopes=SCOPES,
            client_id="client-public",
            client_type="public",
            user_id="7",
            username="bot",
        )
    )
    standalone, refreshed = await asyncio.gather(
        entry.standalone_sender_fn(config, "timeline", "cron post"),
        refresh_if_needed(
            "client-public",
            "http://127.0.0.1:8765/callback",
            transport=transport,
        ),
    )
    assert standalone == {"success": True, "message_id": "704"}
    assert refreshed.access_token == "public-refreshed"
    assert refreshes == 1
    assert writes[-1][2] == "Bearer public-refreshed"

    public_state_path = state_path()
    public_state_before = TwitterState.load().to_dict()
    monkeypatch.setenv("HERMES_HOME", str(profiles["confidential"]))
    monkeypatch.setenv("TWITTER_CLIENT_SECRET", "secret")
    assert load_tokens() == confidential_tokens
    assert state_path() != public_state_path
    assert not state_path().exists()
    assert TwitterState.load().to_dict()["seen_ids"] == []
    confidential_config = PlatformConfig(
        extra={
            "client_id": "client-confidential",
            "oauth_client_type": "confidential",
            "allow_all_users": True,
            "policy": READY_POLICY,
            "_http_transport": transport,
        }
    )
    assert entry.is_connected(confidential_config)
    isolated = entry.adapter_factory(confidential_config)
    assert not isolated._state.seen("101")

    def mutate_confidential(state):
        state.advance_mentions("9001")
        state.advance_dms("9501")
        state.mark_seen("9001")
        state.map_bot_post("9701", "9001")
        state.record_dm_inbound("88-8", "9501")
        state.opt_out_dm("99-8")

    await isolated._mutate_state(mutate_confidential)
    confidential_state = TwitterState.load().to_dict()
    assert confidential_state["mention_since_id"] == "9001"
    assert confidential_state["dm_last_seen_event_id"] == "9501"
    assert confidential_state["bot_post_anchors"] == [["9701", "9001"]]
    assert confidential_state["known_dm_conversations"] == ["88-8", "99-8"]
    assert confidential_state["opted_out_dm_conversations"] == ["99-8"]

    monkeypatch.setenv("HERMES_HOME", str(profiles["public"]))
    assert load_tokens().access_token == "public-refreshed"
    assert TwitterState.load().to_dict() == public_state_before

    public_adapter = entry.adapter_factory(config)
    await public_adapter._mutate_state(lambda state: state.advance_mentions("105"))
    public_state_after = TwitterState.load().to_dict()
    assert public_state_after["mention_since_id"] == "105"

    monkeypatch.setenv("HERMES_HOME", str(profiles["confidential"]))
    assert TwitterState.load().to_dict() == confidential_state

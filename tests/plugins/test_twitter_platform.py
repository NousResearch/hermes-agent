import inspect
import asyncio
import json
import os
import socket
import stat
from unittest.mock import AsyncMock, Mock

import pytest
import httpx

from gateway.config import PlatformConfig


def test_registers_twitter_platform():
    from plugins.platforms.twitter import register

    ctx = Mock()
    register(ctx)

    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "twitter"
    assert kwargs["label"] == "Twitter / X"
    assert kwargs["allowed_users_env"] == "TWITTER_ALLOWED_USERS"
    assert kwargs["allow_all_env"] == "TWITTER_ALLOW_ALL_USERS"
    assert kwargs["cron_deliver_env_var"] == "TWITTER_HOME_CHANNEL"
    assert kwargs["max_message_length"] == 280
    assert callable(kwargs["standalone_sender_fn"])
    assert {call.kwargs["name"] for call in ctx.register_tool.call_args_list} == {
        "twitter_bookmarks",
        "twitter_post_metrics",
    }


def test_settings_reject_unsafe_limits():
    from plugins.platforms.twitter.adapter import TwitterSettings

    with pytest.raises(ValueError, match="poll_interval_seconds"):
        TwitterSettings.from_config(
            PlatformConfig(
                extra={"client_id": "client", "poll_interval_seconds": 0}
            )
        )


def test_apply_yaml_config_uses_nested_platform_block(monkeypatch):
    from plugins.platforms.twitter.adapter import apply_yaml_config

    monkeypatch.delenv("TWITTER_ALLOWED_USERS", raising=False)
    nested = {
        "client_id": "nested-client",
        "allowed_users": ["42"],
        "home_channel": "timeline",
    }
    assert apply_yaml_config({}, nested)["client_id"] == "nested-client"
    assert os.environ["TWITTER_ALLOWED_USERS"] == "42"


def test_adapter_send_signature_matches_base():
    from gateway.platforms.base import BasePlatformAdapter
    from plugins.platforms.twitter.adapter import TwitterAdapter

    assert inspect.signature(TwitterAdapter.send) == inspect.signature(
        BasePlatformAdapter.send
    )


def test_s256_challenge_matches_rfc_vector():
    from plugins.platforms.twitter.oauth import create_s256_challenge

    verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    assert (
        create_s256_challenge(verifier)
        == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    )


def test_tokens_follow_active_hermes_home(monkeypatch, tmp_path):
    from plugins.platforms.twitter.oauth import load_tokens, save_tokens, token_path

    first = tmp_path / "first"
    second = tmp_path / "second"
    monkeypatch.setenv("HERMES_HOME", str(first))
    save_tokens({"access_token": "one", "refresh_token": "r1"})
    assert token_path() == first / "twitter" / "oauth2.json"

    monkeypatch.setenv("HERMES_HOME", str(second))
    assert load_tokens() is None
    save_tokens({"access_token": "two", "refresh_token": "r2"})
    stored = second / "twitter" / "oauth2.json"
    assert json.loads(stored.read_text())["access_token"] == "two"
    assert stat.S_IMODE(stored.stat().st_mode) == 0o600


@pytest.mark.asyncio
async def test_loopback_callback_rejects_state_mismatch():
    from plugins.platforms.twitter.oauth import wait_for_callback

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    waiter = asyncio.create_task(
        wait_for_callback(
            f"http://127.0.0.1:{port}/callback", "expected", timeout=1
        )
    )
    await asyncio.sleep(0.01)
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        b"GET /callback?code=secret&state=wrong HTTP/1.1\r\nHost: localhost\r\n\r\n"
    )
    await writer.drain()
    await reader.read()
    writer.close()
    await writer.wait_closed()

    with pytest.raises(ValueError, match="state"):
        await waiter


@pytest.mark.asyncio
async def test_loopback_callback_has_bounded_timeout():
    from plugins.platforms.twitter.oauth import wait_for_callback

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(TimeoutError):
        await wait_for_callback(
            f"http://127.0.0.1:{port}/callback", "expected", timeout=0.01
        )


def test_branch_anchor_follows_mapped_bot_ancestor(monkeypatch, tmp_path):
    from plugins.platforms.twitter.state import TwitterState

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state = TwitterState.load(max_seen=100, max_branches=100)
    state.map_bot_post("9007199254740993", "123")
    assert state.resolve_anchor("456", ["42", "9007199254740993"]) == "123"
    assert state.resolve_anchor("789", ["42"]) == "789"


def test_state_survives_restart_and_profile_switch(monkeypatch, tmp_path):
    from plugins.platforms.twitter.state import TwitterState

    first = tmp_path / "a"
    second = tmp_path / "b"
    monkeypatch.setenv("HERMES_HOME", str(first))
    state = TwitterState.load()
    state.mark_seen("999")
    state.advance_mentions("999")
    state.save()
    assert TwitterState.load().seen("999")

    monkeypatch.setenv("HERMES_HOME", str(second))
    assert not TwitterState.load().seen("999")


@pytest.mark.asyncio
async def test_write_timeout_is_not_retried():
    from plugins.platforms.twitter.client import AmbiguousWriteError, XClient

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("uncertain", request=request)

    client = XClient(
        token="token",
        transport=httpx.MockTransport(handler),
        max_pending=2,
        max_wait_seconds=1,
    )
    with pytest.raises(AmbiguousWriteError):
        await client.create_post("hello", reply_to="123")
    assert calls == 1
    await client.close()


@pytest.mark.asyncio
async def test_queue_overflow_fails_before_network():
    from plugins.platforms.twitter.queue import RateQueue

    queue = RateQueue(max_pending=1, max_wait_seconds=1)
    blocker = asyncio.Event()
    first = asyncio.create_task(queue.run("write", blocker.wait))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="queue is full"):
        await queue.run("write", lambda: asyncio.sleep(0))
    blocker.set()
    await first


@pytest.mark.asyncio
async def test_queue_wait_timeout_does_not_cancel_started_operation():
    from plugins.platforms.twitter.queue import RateQueue

    queue = RateQueue(max_pending=1, max_wait_seconds=0.01)

    async def slow_operation():
        await asyncio.sleep(0.03)
        return "completed"

    assert await queue.run("write", slow_operation) == "completed"


@pytest.mark.asyncio
async def test_client_keeps_large_ids_as_strings():
    from plugins.platforms.twitter.client import XClient

    def handler(request):
        assert request.url.path == "/2/tweets"
        assert json.loads(request.content)["reply"]["in_reply_to_tweet_id"] == (
            "9007199254740993"
        )
        return httpx.Response(201, json={"data": {"id": "9007199254740994"}})

    client = XClient(token="token", transport=httpx.MockTransport(handler))
    result = await client.create_post("hello", reply_to="9007199254740993")
    assert result == "9007199254740994"
    await client.close()


@pytest.mark.asyncio
async def test_client_retries_explicit_rate_limit_response():
    from plugins.platforms.twitter.client import XClient

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                json={"detail": "rate limited"},
            )
        return httpx.Response(200, json={"data": {"id": "7"}})

    client = XClient(token="token", transport=httpx.MockTransport(handler))
    assert (await client.identity())["data"]["id"] == "7"
    assert calls == 2
    await client.close()


@pytest.mark.asyncio
async def test_client_uses_fresh_token_provider():
    from plugins.platforms.twitter.client import XClient

    def handler(request):
        assert request.headers["Authorization"] == "Bearer fresh"
        return httpx.Response(200, json={"data": {"id": "7"}})

    provider = AsyncMock(return_value="fresh")
    client = XClient(
        token="stale",
        token_provider=provider,
        transport=httpx.MockTransport(handler),
    )
    await client.identity()
    provider.assert_awaited_once()
    await client.close()


@pytest.mark.asyncio
async def test_refresh_is_serialized_across_oauth_clients(monkeypatch, tmp_path):
    from plugins.platforms.twitter.oauth import OAuthClient, SCOPES, save_tokens

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    expired = save_tokens(
        {
            "access_token": "expired",
            "refresh_token": "rotate-once",
            "expires_at": 1,
            "scopes": list(SCOPES),
            "client_id": "client",
        }
    )
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "access_token": "fresh",
                "refresh_token": "rotated",
                "expires_in": 3600,
                "scope": " ".join(SCOPES),
            },
        )

    transport = httpx.MockTransport(handler)
    first_http = httpx.AsyncClient(transport=transport)
    second_http = httpx.AsyncClient(transport=transport)
    first = OAuthClient("client", "http://127.0.0.1:8765/callback", client=first_http)
    second = OAuthClient("client", "http://127.0.0.1:8765/callback", client=second_http)
    one, two = await asyncio.gather(first.refresh(expired), second.refresh(expired))

    assert one.access_token == two.access_token == "fresh"
    assert calls == 1
    await first_http.aclose()
    await second_http.aclose()


@pytest.mark.asyncio
async def test_mention_requires_structured_trigger_and_authorization(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TWITTER_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("TWITTER_ALLOW_ALL_USERS", raising=False)
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allowed_users": ["42"]})
    )
    adapter._account_id = "7"
    adapter.handle_message = AsyncMock()
    post = {
        "id": "101",
        "author_id": "42",
        "conversation_id": "100",
        "text": "@bot hello",
        "entities": {"mentions": [{"id": "7", "username": "bot"}]},
        "referenced_tweets": [],
    }
    await adapter._process_mention(post, {"users": [{"id": "42", "username": "alice"}]})
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "tweet:100:101"
    assert event.source.user_id == "42"
    assert event.message_id == "101"

    adapter.handle_message.reset_mock()
    await adapter._process_mention({**post, "id": "102", "author_id": "99"}, {})
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_common_gateway_authorization_precedes_local_fallback(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TWITTER_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("TWITTER_ALLOW_ALL_USERS", raising=False)
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allow_all_users": "false"})
    )
    adapter._account_id = "7"
    assert not adapter._authorized("42", chat_type="group", chat_id="tweet:1:2")

    adapter.set_authorization_check(lambda user_id, chat_type, chat_id: user_id == "42")
    assert adapter._authorized("42", chat_type="group", chat_id="tweet:1:2")
    assert not adapter._authorized("99", chat_type="group", chat_id="tweet:1:2")


@pytest.mark.asyncio
async def test_dm_routes_by_real_conversation_id(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allowed_users": ["42"]})
    )
    adapter._account_id = "7"
    adapter.handle_message = AsyncMock()
    await adapter._process_dm(
        {
            "id": "501",
            "event_type": "MessageCreate",
            "sender_id": "42",
            "dm_conversation_id": "42-7",
            "text": "hello",
        },
        {"users": [{"id": "42", "username": "alice"}]},
    )
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "dm:42-7"
    assert event.message_id == "501"


@pytest.mark.asyncio
async def test_adapter_send_uses_typed_routes(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(return_value="700")
    adapter._client.send_dm = AsyncMock(return_value="701")

    public = await adapter.send(
        "tweet:100:101", "public", reply_to="9007199254740993"
    )
    direct = await adapter.send("dm:42-7", "private", reply_to="ignored")
    invalid = await adapter.send("123", "bad")

    assert public.success and public.message_id == "700"
    assert direct.success and direct.message_id == "701"
    assert not invalid.success
    adapter._client.send_dm.assert_awaited_once_with("42-7", "private")


@pytest.mark.asyncio
async def test_send_uploads_images_before_creating_post(monkeypatch, tmp_path):
    from PIL import Image

    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    image = tmp_path / "one.png"
    Image.new("RGB", (1, 1)).save(image)
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.upload_image = AsyncMock(return_value="800")
    adapter._client.create_post = AsyncMock(return_value="801")

    result = await adapter.send(
        "timeline", "with image", metadata={"media_files": [(str(image), False)]}
    )

    assert result.success
    adapter._client.create_post.assert_awaited_once_with(
        "with image", media_ids=["800"]
    )


@pytest.mark.asyncio
async def test_partial_image_upload_never_creates_text_only_post(
    monkeypatch, tmp_path
):
    from PIL import Image

    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    images = []
    for name in ("one.png", "two.png"):
        path = tmp_path / name
        Image.new("RGB", (1, 1)).save(path)
        images.append((str(path), False))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.upload_image = AsyncMock(
        side_effect=["800", RuntimeError("upload failed")]
    )
    adapter._client.create_post = AsyncMock(return_value="801")

    result = await adapter.send(
        "timeline", "with images", metadata={"media_files": images}
    )

    assert not result.success
    adapter._client.create_post.assert_not_awaited()


def test_twitter_tools_are_gated_by_profile_oauth(monkeypatch, tmp_path):
    from plugins.platforms.twitter.oauth import save_tokens
    from plugins.platforms.twitter.tools import twitter_available

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert not twitter_available()
    save_tokens({"access_token": "test"})
    assert twitter_available()

    save_tokens({"access_token": "expired", "expires_at": 1})
    assert not twitter_available()


@pytest.mark.asyncio
async def test_standalone_sender_uses_fresh_client(monkeypatch, tmp_path):
    from plugins.platforms.twitter import adapter as adapter_module
    from plugins.platforms.twitter.oauth import save_tokens

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    save_tokens({"access_token": "test"})
    client = Mock()
    client.create_post = AsyncMock(return_value="901")
    client.close = AsyncMock()
    monkeypatch.setattr(adapter_module, "XClient", Mock(return_value=client))

    result = await adapter_module.standalone_send(
        PlatformConfig(extra={"client_id": "client"}), "timeline", "cron post"
    )

    assert result == {"success": True, "message_id": "901"}
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_rejects_invalid_routes_and_oversized_text(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._client = Mock()
    adapter._client.create_post = AsyncMock(return_value="1")
    adapter._client.send_dm = AsyncMock(return_value="2")

    invalid_post = await adapter.send("tweet:not-an-id:anchor", "bad")
    invalid_dm = await adapter.send("dm:../../tokens", "bad")
    oversized = await adapter.send("timeline", "x" * 281)

    assert not invalid_post.success
    assert not invalid_dm.success
    assert not oversized.success
    adapter._client.create_post.assert_not_awaited()
    adapter._client.send_dm.assert_not_awaited()


def test_conversation_context_is_bounded_and_chronological():
    from plugins.platforms.twitter.adapter import build_conversation_context

    posts = [
        {"id": "5", "author_id": "50", "text": "late sibling", "created_at": "2026-01-05", "referenced_tweets": [{"type": "replied_to", "id": "2"}]},
        {"id": "3", "author_id": "7", "text": "bot", "created_at": "2026-01-03", "referenced_tweets": [{"type": "replied_to", "id": "2"}]},
        {"id": "1", "author_id": "10", "text": "root", "created_at": "2026-01-01"},
        {"id": "4", "author_id": "40", "text": "trigger", "created_at": "2026-01-04", "referenced_tweets": [{"type": "replied_to", "id": "3"}]},
        {"id": "2", "author_id": "20", "text": "summon", "created_at": "2026-01-02", "referenced_tweets": [{"type": "replied_to", "id": "1"}]},
        {"id": "6", "author_id": "60", "text": "older sibling", "created_at": "2026-01-02T12:00:00Z", "referenced_tweets": [{"type": "replied_to", "id": "2"}]},
    ]
    rendered = build_conversation_context(
        posts,
        trigger_id="4",
        bot_post_ids={"3"},
        max_depth=3,
        max_posts=5,
        siblings_per_parent=1,
    )

    assert rendered.index("root") < rendered.index("summon") < rendered.index("bot")
    assert rendered.index("bot") < rendered.index("trigger")
    assert "late sibling" in rendered
    assert "older sibling" not in rendered


@pytest.mark.asyncio
async def test_denied_mention_does_not_enrich(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allowed_users": ["42"]})
    )
    adapter._account_id = "7"
    adapter._client = Mock()
    adapter._client.conversation_posts = AsyncMock(side_effect=AssertionError)
    adapter.handle_message = AsyncMock()
    await adapter._process_mention(
        {
            "id": "201",
            "author_id": "99",
            "conversation_id": "200",
            "text": "@bot denied",
            "entities": {"mentions": [{"id": "7"}]},
        },
        {},
    )

    adapter._client.conversation_posts.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mention_polling_consumes_all_pages_oldest_first(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(
        PlatformConfig(extra={"client_id": "client", "allow_all_users": True})
    )
    adapter._account_id = "7"
    adapter._state.mention_since_id = "100"
    adapter._client = Mock()
    adapter._client.mentions = AsyncMock(
        side_effect=[
            {
                "data": [{"id": "102", "author_id": "42", "conversation_id": "90", "text": "second", "entities": {"mentions": [{"id": "7"}]}}],
                "meta": {"next_token": "next"},
            },
            {
                "data": [{"id": "101", "author_id": "42", "conversation_id": "90", "text": "first", "entities": {"mentions": [{"id": "7"}]}}],
                "meta": {},
            },
        ]
    )
    adapter._client.conversation_posts = AsyncMock(return_value={})
    adapter.handle_message = AsyncMock()

    await adapter._poll_mentions_once()

    assert [call.args[0].message_id for call in adapter.handle_message.await_args_list] == [
        "101",
        "102",
    ]
    assert adapter._state.mention_since_id == "102"


@pytest.mark.asyncio
async def test_dm_pagination_stops_at_saved_boundary(monkeypatch, tmp_path):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = TwitterAdapter(PlatformConfig(extra={"client_id": "client"}))
    adapter._state.dm_since_id = "100"
    adapter._client = Mock()
    adapter._client.dm_events = AsyncMock(
        side_effect=[
            {
                "data": [{"id": "103"}, {"id": "102"}],
                "meta": {"next_token": "next"},
            },
            {
                "data": [{"id": "101"}, {"id": "100"}],
                "meta": {},
            },
        ]
    )

    page = await adapter._dm_pages()

    assert [item["id"] for item in page["data"]] == ["103", "102", "101"]
    assert page["meta"]["complete"] is True
    assert adapter._client.dm_events.await_count == 2

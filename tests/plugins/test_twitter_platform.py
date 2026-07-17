import inspect
import asyncio
import json
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


def test_settings_reject_unsafe_limits():
    from plugins.platforms.twitter.adapter import TwitterSettings

    with pytest.raises(ValueError, match="poll_interval_seconds"):
        TwitterSettings.from_config(
            PlatformConfig(
                extra={"client_id": "client", "poll_interval_seconds": 0}
            )
        )


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
async def test_mention_requires_structured_trigger_and_authorization(
    monkeypatch, tmp_path
):
    from plugins.platforms.twitter.adapter import TwitterAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
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

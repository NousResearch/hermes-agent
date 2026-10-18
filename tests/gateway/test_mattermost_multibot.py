"""Mattermost multi-bot intake contracts; regression for #86925."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.mattermost.adapter import MattermostAdapter


def make_adapter(**extra):
    adapter = MattermostAdapter(PlatformConfig(enabled=True, token="test-token", extra=extra))
    adapter._bot_user_id, adapter._bot_username = "own-bot", "hermes-bot"
    adapter.handle_message = AsyncMock()
    adapter.set_authorization_check(lambda *args: True)
    return adapter


def runner_for(adapter, primary=None):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.MATTERMOST: (primary or adapter).config})
    runner.adapters, runner.pairing_store = {Platform.MATTERMOST: primary or adapter}, None
    runner._profile_adapters = {"secondary": {Platform.MATTERMOST: adapter}}
    return runner


def posted(text, *, channel="listed", user="allowed", kind="O", post="post"):
    return {"event": "posted", "data": {
        "post": json.dumps({"id": post, "user_id": user, "channel_id": channel, "message": text}),
        "channel_type": kind, "sender_name": "@synthetic"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("mention", ["@peer-bot", "@PEER-BOT", "@peer-bot."])
async def test_peer_mentions_strip_bots_preserve_humans_and_cache(mention):
    adapter = make_adapter()
    async def lookup(path):
        username = path.rsplit("/", 1)[-1]
        return {"username": username, **({"is_bot": True} if username == "peer-bot" else {})}
    adapter._api_get = AsyncMock(side_effect=lookup)
    for index in range(2):
        await adapter._handle_ws_event(posted(
            f"@hermes-bot ask {mention} and @human-user via mail@peer-bot or @here", post=str(index)))
        assert adapter.handle_message.call_args[0][0].text == (
            "ask " + (". " if mention.endswith(".") else "") + "and @human-user via mail@peer-bot or @here")
    assert adapter._api_get.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [{}, TimeoutError("lookup timed out")])
async def test_failed_peer_lookup_preserves_mention_until_ttl(monkeypatch, failure):
    adapter = make_adapter()
    clock = [100.0]
    monkeypatch.setattr("plugins.platforms.mattermost.adapter.time.monotonic", lambda: clock[0])
    adapter._api_get = AsyncMock(side_effect=[failure, {"username": "peer-bot", "is_bot": True}])
    for index, expected in enumerate(["ask @peer-bot", "ask @peer-bot", "ask"]):
        if index == 2:
            clock[0] += 61
        await adapter._handle_ws_event(posted("@hermes-bot ask @peer-bot", post=str(index)))
        assert adapter.handle_message.call_args[0][0].text == expected
    assert adapter._api_get.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["O", "P", "G"])
async def test_channel_allowlist_gates_intake_and_gateway(monkeypatch, kind):
    adapter = make_adapter(groups={"listed": {"allow_from": ["allowed"]}},
                           dm_policy="allowlist", group_policy="allowlist")
    runner, verdicts = runner_for(adapter), []
    async def receive(event):
        verdicts.append(runner._is_user_authorized(event.source))
    adapter.handle_message.side_effect = receive
    await adapter._handle_ws_event(posted("@hermes-bot hi", user="denied", kind=kind))
    adapter.handle_message.assert_not_awaited()
    await adapter._handle_ws_event(posted("@hermes-bot hi", kind=kind, post="allowed"))
    await adapter._handle_ws_event(posted("@hermes-bot hi", channel="unlisted", kind=kind, post="unlisted"))
    await adapter._handle_ws_event(posted("hi", kind="D", post="dm"))
    assert verdicts == [True, False, False]
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    await adapter._handle_ws_event(posted("@hermes-bot hi", user="denied", kind=kind, post="denied2"))
    assert verdicts == [True, False, False]


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["listed", "*"])
async def test_multiplex_channel_grants_use_receiving_adapter(key):
    secondary = make_adapter()
    runner = runner_for(secondary, make_adapter(groups={key: {"allow_from": ["alice"]}}))
    secondary.set_authorization_check(lambda user, kind, channel: runner._is_user_authorized(
        SessionSource(platform=Platform.MATTERMOST, user_id=user, chat_type=kind,
                      chat_id=channel, profile="secondary")))
    secondary._api_get = AsyncMock()
    await secondary._handle_ws_event(posted("@hermes-bot ask @peer-bot", user="bob"))
    event = secondary.handle_message.call_args[0][0]
    event.source.profile = "secondary"
    assert runner._is_user_authorized(event.source) is False
    secondary._api_get.assert_not_awaited()
    listed = make_adapter(groups={"listed": {"allow_from": ["bob"]}})
    runner = runner_for(listed, make_adapter())
    assert runner._is_user_authorized(event.source) is True


@pytest.mark.parametrize("policy,behavior", [("disabled", "ignore"), ("allowlist", "ignore"), ("pairing", "pair")])
def test_dm_policy_preserves_base_behavior(policy, behavior):
    adapter = make_adapter(dm_policy=policy)
    runner = runner_for(adapter)
    assert runner._get_unauthorized_dm_behavior(Platform.MATTERMOST) == behavior
    assert not runner._is_user_authorized(SessionSource(
        platform=Platform.MATTERMOST, user_id="unknown", chat_id="dm", chat_type="dm"))


@pytest.mark.asyncio
async def test_mention_lookups_bounded_concurrent_and_authorized():
    adapter = make_adapter()
    gate = asyncio.Event()
    async def lookup(path):
        if adapter._api_get.await_count == 4:
            gate.set()
        await asyncio.wait_for(gate.wait(), timeout=2)
        return {}
    adapter._api_get = AsyncMock(side_effect=lookup)
    text = "@hermes-bot " + " ".join(f"@u{i}" + "." * (64 - len(f"u{i}")) for i in range(100))
    adapter.set_authorization_check(lambda *args: None)
    await adapter._handle_ws_event(posted(text))
    adapter._api_get.assert_not_awaited()
    adapter.set_authorization_check(lambda *args: True)
    await adapter._handle_ws_event(posted(text, post="authorized"))
    assert adapter._api_get.await_count == 4

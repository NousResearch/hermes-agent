"""Surfaced Discord thread cooldowns must not trigger more creation attempts.

Regression for #88653. Real SDK exceptions, synthetic transport, no Discord API calls.
"""

import asyncio
import json
import math
import time
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

discord = pytest.importorskip("discord")

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.fixture
def adapter():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="test-token", extra={
        "require_mention": False, "auto_thread": True,
    }))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    adapter._text_batch_delay_seconds = 0
    adapter.handle_message = AsyncMock()
    return adapter


def message_with_error(error):
    channel = SimpleNamespace(
        id=800, name="general", guild=SimpleNamespace(name="Test server"),
        topic=None, send=AsyncMock(return_value=SimpleNamespace(
            create_thread=AsyncMock(side_effect=error), delete=AsyncMock(),
        )),
    )
    return SimpleNamespace(
        id=123, content="hello", mentions=[], attachments=[], reference=None,
        created_at=datetime.now(UTC), channel=channel,
        author=SimpleNamespace(id=42, display_name="TestUser", name="TestUser"),
        create_thread=AsyncMock(side_effect=error),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress", [False, True])
@pytest.mark.parametrize("error, hint", [
    (discord.RateLimited(286.02), 286.02),
    (discord.RateLimited(0.1), 1),
    (discord.RateLimited(60), 60),
    (discord.HTTPException(SimpleNamespace(status=429, reason="Too many requests", headers={
        "Retry-After": "120",
    }), "rate limited"), 120),
    (discord.HTTPException(SimpleNamespace(status=429, reason="Too many requests", headers={}),
                           "rate limited"), None),
])
async def test_direct_cooldown_stops_creation_and_respects_warning_policy(
    adapter, monkeypatch, tmp_path, suppress, error, hint,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"display:\n  suppress_warning_notifications: {str(suppress).lower()}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(time, "time", lambda: 1700000000.25)
    message = message_with_error(error)

    assert await adapter._handle_message(message) is False

    adapter.handle_message.assert_not_awaited()
    message.create_thread.assert_awaited_once()
    message.channel.send.assert_awaited_once()
    notice = message.channel.send.await_args.args[0]
    if suppress:
        assert notice == "Hermes could not confirm a new thread. Check for an existing thread before retrying."
    else:
        assert "rate-limiting" in notice
        if hint is None:
            assert "Try again later" in notice
            assert "<t:" not in notice
        else:
            deadline = math.ceil(1700000000.25 + hint)
            assert f"<t:{deadline}:R>" in notice
            assert f"<t:{deadline}:F>" in notice
        assert "not processed" not in notice
        assert "existing thread" in notice


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["seed_send", "seed_thread", "seed_delete_failure"])
async def test_fallback_cooldown_stops_retries_and_cleans_seed(adapter, stage):
    message = message_with_error(RuntimeError("connection failed"))
    seed = SimpleNamespace(create_thread=AsyncMock(side_effect=discord.RateLimited(60)), delete=AsyncMock())
    if stage == "seed_send":
        message.channel.send.side_effect = [discord.RateLimited(60), None]
    else:
        message.channel.send.side_effect = [seed, None]
        if stage == "seed_delete_failure":
            seed.delete.side_effect = RuntimeError("delete failed")

    assert await adapter._handle_message(message) is False

    adapter.handle_message.assert_not_awaited()
    message.create_thread.assert_awaited_once()
    if stage == "seed_send":
        seed.create_thread.assert_not_awaited()
        seed.delete.assert_not_awaited()
    else:
        seed.create_thread.assert_awaited_once()
        seed.delete.assert_awaited_once()
    assert message.channel.send.await_count == 2
    notice = message.channel.send.await_args.args[0]
    assert "<t:" in notice
    assert "rate-limiting" in notice
    assert "not processed" not in notice


class _Response:
    def __init__(self, status, data):
        self.status = status
        self.data = data
        self.headers = {"content-type": "application/json", "Via": "test"}

    async def text(self, **kwargs):
        return json.dumps(self.data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize("global_limit", [False, True])
async def test_sdk_backoff_keeps_request_owned_and_unrelated_sends_work(adapter, monkeypatch, global_limit):
    """Characterize SDK waits, including global gates, without waiting a minute."""
    from discord.http import HTTPClient, Route

    http = HTTPClient(asyncio.get_running_loop())
    http._global_over = asyncio.Event()
    http._global_over.set()
    entered_backoff = asyncio.Event()
    release_backoff = asyncio.Event()
    other_started = asyncio.Event()
    sleep = asyncio.sleep
    thread_posts = []
    other_posts = []

    async def controlled_sleep(delay):
        if delay == 0:
            return await sleep(0)
        assert delay == 60
        entered_backoff.set()
        await release_backoff.wait()

    def request(method, url, **kwargs):
        if url.endswith("/threads"):
            thread_posts.append(url)
            if len(thread_posts) == 1:
                return _Response(429, {"retry_after": 60, "global": global_limit})
            return _Response(200, {"id": 123, "name": "hello"})
        other_posts.append(url)
        return _Response(200, {"id": 456})

    http._HTTPClient__session = SimpleNamespace(request=request)
    monkeypatch.setattr(discord.http.asyncio, "sleep", controlled_sleep)
    message = message_with_error(None)

    async def create_thread(**kwargs):
        data = await http.request(Route("POST", "/channels/{channel_id}/messages/{message_id}/threads",
                                        channel_id=800, message_id=123))
        return SimpleNamespace(**data)

    async def send_elsewhere():
        other_started.set()
        return await http.request(Route("POST", "/channels/{channel_id}/messages", channel_id=900))

    message.create_thread.side_effect = create_thread
    creation = asyncio.create_task(adapter._auto_create_thread(message))
    other = None
    try:
        await asyncio.wait_for(entered_backoff.wait(), timeout=5)
        assert not creation.done()
        other = asyncio.create_task(send_elsewhere())
        await asyncio.wait_for(other_started.wait(), timeout=5)
        if global_limit:
            assert not http._global_over.is_set()
            assert not other_posts
        else:
            assert http._global_over.is_set()
            assert other_posts
        release_backoff.set()
        _, pending = await asyncio.wait({creation, other}, timeout=5)
        assert not pending
        assert creation.result().id == 123
        assert other.result() == {"id": 456}
        assert http._global_over.is_set()
        assert len(thread_posts) == 2
        message.create_thread.assert_awaited_once()
        message.channel.send.assert_not_awaited()
    finally:
        release_backoff.set()
        tasks = [task for task in (creation, other) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

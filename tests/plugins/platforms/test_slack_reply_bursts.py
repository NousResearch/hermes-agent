"""Final DM delivery uses the real adapter, Slack SDK, HTTP and delivery ledger."""
import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio
from aiohttp import web
from slack_sdk.web.async_client import AsyncWebClient

from gateway.config import GatewayConfig, Platform, PlatformConfig, StreamingConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from gateway.run import GatewayRunner
from gateway import delivery_ledger as ledger
from plugins.platforms.slack.adapter import SlackAdapter


@pytest_asyncio.fixture
async def slack(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "_db_path", lambda: tmp_path / "state.db")
    state = SimpleNamespace(posts=[], attempts=[], refuse=None)

    async def endpoint(request):
        payload = await request.json()
        if request.match_info["method"] == "chat.postMessage":
            state.attempts.append(payload)
            if state.refuse and state.refuse in payload["text"]:
                return web.json_response({"ok": False, "error": "invalid_auth"})
            state.posts.append(payload)
        return web.json_response({"ok": True, "ts": str(len(state.posts)), "channel": "DCHAT"})

    app = web.Application()
    app.router.add_post("/{method}", endpoint)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    state.client = AsyncWebClient(token="test-token", base_url=f"http://127.0.0.1:{port}/", proxy="")
    yield state
    await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,channel,internal,body,count", [
    (True, "DCHAT", False, "First thought\n\nSecond thought\n\nThird thought", 3),
    (False, "DCHAT", False, "First thought\n\nSecond thought", 1),
    (True, "CCHANNEL", False, "First thought\n\nSecond thought", 1),
    (True, "DCHAT", True, "First thought\n\nSecond thought", 1),
    (True, "DCHAT", False, "Here is the code\n\n```py\na = 1\n\nb = 2\n```\n\nDone", 3),
    (True, "DCHAT", False, "- First item\n\n- Second item", 1),
    (True, "DCHAT", False, "See [guide][ref]\n\n[ref]: https://example.com", 1),
])
async def test_final_reply_scope_and_structure(slack, enabled, channel, internal, body, count):
    config = PlatformConfig.from_dict({"enabled": True, "extra": {
        "dm_reply_bursts": enabled, "reply_in_thread": False,
    }})
    adapter = SlackAdapter(config)
    adapter._app = SimpleNamespace(client=slack.client)
    source = SessionSource(platform=Platform.SLACK, chat_id=channel, thread_id="123.456")
    event = MessageEvent(text="Explain", source=source, message_id="123.789", internal=internal)
    result, _ = await adapter.send_final_ledgered(event, "session", body,
                                                 {"thread_id": source.thread_id}, reply_to=event.message_id)
    assert result.success
    assert len(slack.posts) == count
    assert all(post["channel"] == channel and post["thread_ts"] == "123.456" for post in slack.posts)
    if "```" in body:
        assert "a = 1\n\nb = 2" in slack.posts[1]["text"]
    else:
        assert "First" in slack.posts[0]["text"] or "guide" in slack.posts[0]["text"]
    # Generic sends, including cron and cards, never enter the final-reply splitter.
    slack.posts.clear()
    await adapter.send(channel, body, metadata={"job_id": "routine", "notify": True})
    assert len(slack.posts) == 1
    assert "thread_ts" not in slack.posts[0]
    assert adapter.prefers_buffered_reply(channel) is (enabled and channel.startswith("D"))
    gateway = GatewayRunner.__new__(GatewayRunner)
    gateway.config = GatewayConfig(streaming=StreamingConfig(enabled=True))
    gateway._delivery_adapter_for = lambda _: adapter
    consumer = gateway._proxy_stream_consumer(source, event.message_id, {}, lambda: True)
    assert (consumer is None) is (enabled and channel.startswith("D"))


@pytest.mark.asyncio
async def test_partial_failure_recovery_and_cancellation_keep_only_unsent_tail(slack):
    adapter = SlackAdapter(PlatformConfig(extra={"dm_reply_bursts": True, "reply_in_thread": False}))
    adapter._app = SimpleNamespace(client=slack.client)
    source = SessionSource(platform=Platform.SLACK, chat_id="DCHAT", thread_id="123.456")
    event = MessageEvent(text="Explain", source=source, message_id="123.789")
    slack.refuse = "Second"
    result, _ = await adapter.send_final_ledgered(event, "session", "First\n\nSecond\n\nThird",
                                                 {"thread_id": source.thread_id}, reply_to=event.message_id)
    assert not result.success
    assert [post["text"] for post in slack.posts] == ["First"]
    with ledger._connect() as conn:
        row = conn.execute("SELECT obligation_id, content, state FROM delivery_obligations").fetchone()
    assert row[1:] == ("Second\n\nThird", "failed")
    slack.refuse = None
    gateway = GatewayRunner.__new__(GatewayRunner)
    gateway.adapters = {Platform.SLACK: adapter}
    recovered = await gateway._redeliver_claimed_obligations([{
        "obligation_id": row[0], "content": row[1], "platform": "slack", "chat_id": "DCHAT",
        "thread_id": source.thread_id, "attempts": 1,
    }])
    assert recovered == 1
    assert [post["text"] for post in slack.posts] == ["First", "Second\n\nThird"]
    assert all(post["thread_ts"] == source.thread_id for post in slack.posts)

    task = asyncio.create_task(adapter.send_final_ledgered(
        event, "cancel-session", "Fourth\n\nFifth", {}, reply_to=None))
    async def wait_for_first():
        while not any(post["text"] == "Fourth" for post in slack.posts):
            await asyncio.sleep(0.01)
        while True:
            with ledger._connect() as conn:
                saved = conn.execute("SELECT content FROM delivery_obligations WHERE session_key='cancel-session'").fetchone()
            if saved and saved[0] == "Fifth":
                return
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait_for_first(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not any(post["text"] == "Fifth" for post in slack.posts)

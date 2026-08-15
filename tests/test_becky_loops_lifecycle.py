from types import SimpleNamespace

import pytest

from gateway import becky_loops
from gateway.becky_loops import (
    BeckyLoopsConfig,
    BeckyLoopsBridgeServer,
    SessionDBBeckyLoopsStore,
    start_becky_loops_bridge,
    stop_becky_loops_bridge,
)
from gateway.config import Platform
from gateway.run import GatewayRunner


class EmptyDB:
    def list_sessions_rich(self, **kwargs):
        return []

    def get_messages(self, session_id, include_inactive=False):
        return []


class FakeTopicSender:
    async def send_topic(self, **kwargs):
        del kwargs
        return becky_loops.TopicSendReceipt(message_id="1")


class FakeReplyGenerator:
    async def generate(self, **kwargs):
        del kwargs
        return "answer"


@pytest.mark.asyncio
async def test_lifecycle_starts_loopback_bridge_and_stops_it() -> None:
    config = BeckyLoopsConfig(
        enabled=True,
        chat_id="123456789",
        token="t" * 64,
        port=0,
    )
    server = await start_becky_loops_bridge(config=config, db=EmptyDB())
    assert isinstance(server, BeckyLoopsBridgeServer)
    assert server.bound_port > 0
    await stop_becky_loops_bridge(server)


@pytest.mark.asyncio
async def test_lifecycle_is_noop_when_disabled() -> None:
    config = BeckyLoopsConfig(
        enabled=False,
        chat_id="123456789",
        token="t" * 64,
        port=0,
    )
    assert await start_becky_loops_bridge(config=config, db=EmptyDB()) is None
    await stop_becky_loops_bridge(None)


@pytest.mark.asyncio
async def test_lifecycle_injects_proven_reply_dependencies() -> None:
    config = BeckyLoopsConfig(
        enabled=True,
        chat_id="123456789",
        token="t" * 64,
        port=0,
        topic_reply="bot_api_private_topic",
    )
    server = await start_becky_loops_bridge(
        config=config,
        db=EmptyDB(),
        topic_sender=FakeTopicSender(),
        reply_generator=FakeReplyGenerator(),
    )
    assert isinstance(server, BeckyLoopsBridgeServer)
    assert (await server._method("becky.loops.capabilities", {}))[
        "topic_reply"
    ] == "bot_api_private_topic"
    await stop_becky_loops_bridge(server)


@pytest.mark.asyncio
async def test_lifecycle_stays_read_only_without_sender() -> None:
    config = BeckyLoopsConfig(
        enabled=True,
        chat_id="123456789",
        token="t" * 64,
        port=0,
        topic_reply="bot_api_private_topic",
    )
    server = await start_becky_loops_bridge(
        config=config,
        db=EmptyDB(),
        reply_generator=FakeReplyGenerator(),
    )
    assert isinstance(server, BeckyLoopsBridgeServer)
    assert (await server._method("becky.loops.capabilities", {}))[
        "topic_reply"
    ] == "unavailable"
    await stop_becky_loops_bridge(server)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("topic_reply", "adapter_present", "expect_sender"),
    [
        ("bot_api_private_topic", True, True),
        ("unavailable", True, False),
        ("bot_api_private_topic", False, False),
    ],
)
async def test_runner_passes_only_a_connected_explicitly_proven_telegram_sender(
    monkeypatch: pytest.MonkeyPatch,
    topic_reply: str,
    adapter_present: bool,
    expect_sender: bool,
) -> None:
    config = BeckyLoopsConfig(
        enabled=True,
        chat_id="123456789",
        token="t" * 64,
        port=0,
        topic_reply=topic_reply,
    )
    adapter = object()
    captured: list[dict] = []

    async def fake_start(**kwargs):
        captured.append(kwargs)
        return "server"

    monkeypatch.setattr(becky_loops, "load_becky_loops_config", lambda: config)
    monkeypatch.setattr(becky_loops, "start_becky_loops_bridge", fake_start)
    runner = object.__new__(GatewayRunner)
    runner._session_db = SimpleNamespace(_db=EmptyDB())
    runner._becky_loops_bridge = None
    runner.adapters = {Platform.TELEGRAM: adapter} if adapter_present else {}

    await runner._start_becky_loops_bridge()

    assert runner._becky_loops_bridge == "server"
    sender = captured[0]["topic_sender"]
    if expect_sender:
        assert isinstance(sender, becky_loops.TelegramTopicSender)
        assert sender._adapter is adapter
    else:
        assert sender is None

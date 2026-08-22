import asyncio
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
async def test_runner_dispatches_dashboard_reply_into_telegram_agent_pipeline() -> None:
    class FakeTelegramAdapter:
        def __init__(self) -> None:
            self.events = []

        async def handle_message(self, event) -> None:
            self.events.append(event)

    adapter = FakeTelegramAdapter()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}

    await runner._dispatch_becky_agent_reply(
        chat_id="-1004476874933",
        thread_id="3964",
        session_id="session-1",
        text="Please turn on the porch light.",
        reply_to_message_id="101",
    )

    event = adapter.events[0]
    assert event.text == "Please turn on the porch light."
    assert event.source.platform is Platform.TELEGRAM
    assert event.source.chat_id == "-1004476874933"
    assert event.source.chat_type == "group"
    assert event.source.thread_id == "3964"
    assert event.message_id.startswith("becky-dashboard-")
    assert event.internal is True


@pytest.mark.asyncio
async def test_runner_binds_dashboard_reply_to_existing_session_before_dispatch() -> None:
    class FakeTelegramAdapter:
        async def handle_message(self, event) -> None:
            self.event = event

    class FakeSessionStore:
        def lookup_by_session_id(self, session_id):
            assert session_id == "session-1"
            return SimpleNamespace(
                session_key="agent:main:telegram:group:-1004476874933:3964",
                session_id=session_id,
            )

    class FakeSessionDB:
        def __init__(self) -> None:
            self.bindings = []

        def bind_telegram_topic(self, **kwargs) -> None:
            self.bindings.append(kwargs)

    adapter = FakeTelegramAdapter()
    session_db = FakeSessionDB()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = FakeSessionStore()
    runner._session_db = SimpleNamespace(_db=session_db)

    await runner._dispatch_becky_agent_reply(
        chat_id="-1004476874933",
        thread_id="3964",
        session_id="session-1",
        text="Please turn on the porch light.",
        reply_to_message_id="101",
    )

    assert session_db.bindings == [
        {
            "chat_id": "-1004476874933",
            "thread_id": "3964",
            "user_id": "",
            "session_key": "agent:main:telegram:group:-1004476874933:3964",
            "session_id": "session-1",
        }
    ]
    assert adapter.event.text == "Please turn on the porch light."


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
        assert callable(captured[0]["agent_dispatcher"])
    else:
        assert sender is None
        assert captured[0]["agent_dispatcher"] is None


@pytest.mark.asyncio
async def test_runner_starts_and_stops_proven_mtproto_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = BeckyLoopsConfig(
        enabled=True,
        chat_id="8837347581",
        token="t" * 64,
        port=0,
        topic_control="mtproto_private_topic",
    )

    class FakeMtprotoController:
        method = "mtproto_private_topic"
        is_connected = True
        supports_close = True

        def __init__(self) -> None:
            self.started = False
            self.stopped = False

        @classmethod
        def from_environment(cls, *, chat_id: str):
            assert chat_id == "8837347581"
            return instance

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

    instance = FakeMtprotoController()
    captured: list[dict] = []

    async def fake_start(**kwargs):
        captured.append(kwargs)
        return "server"

    async def fake_stop(server) -> None:
        assert server == "server"

    monkeypatch.setattr(becky_loops, "load_becky_loops_config", lambda: config)
    monkeypatch.setattr(becky_loops, "start_becky_loops_bridge", fake_start)
    monkeypatch.setattr(becky_loops, "stop_becky_loops_bridge", fake_stop)
    monkeypatch.setattr(
        "gateway.telegram_mtproto.MTProtoPrivateTopicController",
        FakeMtprotoController,
    )
    runner = object.__new__(GatewayRunner)
    runner._session_db = SimpleNamespace(_db=EmptyDB())
    runner._becky_loops_bridge = None
    runner._becky_loops_topic_controller = None
    runner.adapters = {}

    await runner._start_becky_loops_bridge()

    assert instance.started is True
    assert captured[0]["topic_controller"] is instance
    assert runner._becky_loops_topic_controller is instance

    await runner._stop_becky_loops_bridge()

    assert instance.stopped is True
    assert runner._becky_loops_topic_controller is None


@pytest.mark.asyncio
async def test_runner_stops_mtproto_controller_when_bridge_stop_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeController:
        stopped = False

        async def stop(self) -> None:
            self.stopped = True

    controller = FakeController()

    async def cancelled_stop(server) -> None:
        assert server == "server"
        raise asyncio.CancelledError

    monkeypatch.setattr(becky_loops, "stop_becky_loops_bridge", cancelled_stop)
    runner = object.__new__(GatewayRunner)
    runner._becky_loops_bridge = "server"
    runner._becky_loops_topic_controller = controller

    with pytest.raises(asyncio.CancelledError):
        await runner._stop_becky_loops_bridge()

    assert controller.stopped is True
    assert runner._becky_loops_topic_controller is None

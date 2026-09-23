import asyncio
from datetime import UTC, datetime
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
from gateway.run import GatewayRunner, _validated_becky_dashboard_url
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


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


class CloseController:
    method = "mtproto_private_topic"
    is_connected = True
    supports_close = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def close_topic(self, *, chat_id: str, thread_id: str):
        self.calls.append((chat_id, thread_id))
        from datetime import UTC, datetime

        return datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


class FailingCloseController(CloseController):
    async def close_topic(self, *, chat_id: str, thread_id: str):
        self.calls.append((chat_id, thread_id))
        raise RuntimeError("Telegram rejected close")


class PostDeliveryAdapter:
    def __init__(self) -> None:
        self.callbacks: list[tuple[str, object, int | None]] = []

    def register_post_delivery_callback(
        self, session_key: str, callback, *, generation: int | None = None
    ) -> None:
        self.callbacks.append((session_key, callback, generation))


class CloseSessionDB:
    def __init__(self) -> None:
        self.ended: list[tuple[str, str]] = []

    async def get_telegram_topic_binding(self, *, chat_id: str, thread_id: str):
        assert (chat_id, thread_id) == ("-1004476874933", "3")
        return {"session_id": "session-3"}

    async def end_session(self, session_id: str, reason: str) -> None:
        self.ended.append((session_id, reason))


class UnboundCloseSessionDB(CloseSessionDB):
    async def get_telegram_topic_binding(self, *, chat_id: str, thread_id: str):
        assert (chat_id, thread_id) == ("-1004476874933", "3")
        return None

    async def list_sessions_rich(self, **kwargs: object):
        assert kwargs["source"] == "telegram"
        return [
            {
                "id": "session-3",
                "chat_id": "-1004476874933",
                "thread_id": "3",
                "ended_at": None,
                "last_active": 2.0,
            },
            {
                "id": "session-3-older",
                "chat_id": "-1004476874933",
                "thread_id": "3",
                "ended_at": None,
                "last_active": 1.0,
            },
        ]


class PaginatedUnboundCloseSessionDB(UnboundCloseSessionDB):
    def __init__(self) -> None:
        super().__init__()
        self.offsets: list[int] = []

    async def list_sessions_rich(self, **kwargs: object):
        offset = int(kwargs["offset"])
        self.offsets.append(offset)
        if offset == 0:
            return [
                {
                    "id": f"unrelated-{index}",
                    "chat_id": "-1004476874933",
                    "thread_id": str(index + 10),
                    "ended_at": None,
                }
                for index in range(200)
            ]
        if offset == 200:
            return [{
                "id": "session-3-late",
                "chat_id": "-1004476874933",
                "thread_id": "3",
                "ended_at": None,
            }]
        return []


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com:8787",
        "http://localhost:8787",
        "http://127.0.0.1.evil.example:8787",
        "http://user:password@127.0.0.1:8787",
        "http://127.0.0.1:8787/dashboard",
        "http://127.0.0.1:8787/?redirect=example.com",
        "http://169.254.169.254:8787",
    ],
)
def test_becky_dashboard_callback_url_must_be_loopback_without_url_tricks(url: str) -> None:
    assert _validated_becky_dashboard_url(url) is None


def test_becky_dashboard_callback_url_accepts_loopback_ip_only() -> None:
    assert _validated_becky_dashboard_url("http://127.0.0.1:8787/") == (
        "http://127.0.0.1:8787"
    )


@pytest.mark.asyncio
async def test_becky_archive_never_posts_token_to_external_url(monkeypatch) -> None:
    import httpx

    runner = object.__new__(GatewayRunner)
    runner._becky_loops_config = BeckyLoopsConfig(
        enabled=True,
        chat_id="-1004476874933",
        token="t" * 64,
    )
    monkeypatch.setenv("HERMES_BECKY_DASHBOARD_URL", "https://example.com")

    def unexpected_client(*_args, **_kwargs):
        raise AssertionError("external archive callback must not be opened")

    monkeypatch.setattr(httpx, "AsyncClient", unexpected_client)

    result = await runner._notify_becky_topic_closed(
        source_ref="loop_" + "A" * 43,
        closed_at=datetime(2026, 8, 22, 12, 0, tzinfo=UTC),
        control_method="mtproto_private_topic",
    )

    assert result == "pending"


@pytest.mark.asyncio
async def test_telegram_close_command_closes_topic_ends_session_and_notifies_becky() -> None:
    runner = object.__new__(GatewayRunner)
    runner._becky_loops_config = BeckyLoopsConfig(
        enabled=True,
        chat_id="-1004476874933",
        token="t" * 64,
        managed_topic_ids=frozenset({"2"}),
    )
    controller = CloseController()
    session_db = CloseSessionDB()
    notifications: list[dict[str, object]] = []
    runner._becky_loops_topic_controller = controller
    runner._session_db = session_db

    async def notify(**kwargs: object) -> str:
        notifications.append(kwargs)
        return "archived"

    runner._notify_becky_topic_closed = notify

    result = await runner._handle_becky_close_command(
        chat_id="-1004476874933",
        thread_id="3",
        user_id="8837347581",
        message_id="42",
    )

    assert result == "Topic closed and archived."
    assert controller.calls == [("-1004476874933", "3")]
    assert session_db.ended == [("session-3", "telegram_topic_closed")]
    assert notifications[0]["source_ref"].startswith("loop_")


@pytest.mark.asyncio
async def test_telegram_close_command_rejects_managed_topic() -> None:
    runner = object.__new__(GatewayRunner)
    runner._becky_loops_config = BeckyLoopsConfig(
        enabled=True,
        chat_id="-1004476874933",
        token="t" * 64,
        managed_topic_ids=frozenset({"2"}),
    )
    controller = CloseController()
    runner._becky_loops_topic_controller = controller

    result = await runner._handle_becky_close_command(
        chat_id="-1004476874933",
        thread_id="2",
        user_id="8837347581",
        message_id="42",
    )

    assert "system topic" in result.casefold()
    assert controller.calls == []


@pytest.mark.asyncio
async def test_telegram_close_command_ends_unbound_topic_session() -> None:
    runner = object.__new__(GatewayRunner)
    runner._becky_loops_config = BeckyLoopsConfig(
        enabled=True,
        chat_id="-1004476874933",
        token="t" * 64,
        managed_topic_ids=frozenset({"2"}),
    )
    runner._becky_loops_topic_controller = CloseController()
    session_db = UnboundCloseSessionDB()
    runner._session_db = session_db

    async def notify(**kwargs: object) -> str:
        del kwargs
        return "pending"

    runner._notify_becky_topic_closed = notify

    await runner._handle_becky_close_command(
        chat_id="-1004476874933",
        thread_id="3",
        user_id="8837347581",
        message_id="42",
    )

    assert session_db.ended == [
        ("session-3", "telegram_topic_closed"),
        ("session-3-older", "telegram_topic_closed"),
    ]


@pytest.mark.asyncio
async def test_telegram_close_command_scans_all_session_pages() -> None:
    runner = object.__new__(GatewayRunner)
    runner._becky_loops_config = BeckyLoopsConfig(
        enabled=True,
        chat_id="-1004476874933",
        token="t" * 64,
        managed_topic_ids=frozenset({"2"}),
    )
    runner._becky_loops_topic_controller = CloseController()
    session_db = PaginatedUnboundCloseSessionDB()
    runner._session_db = session_db

    async def notify(**kwargs: object) -> str:
        del kwargs
        return "pending"

    runner._notify_becky_topic_closed = notify

    await runner._handle_becky_close_command(
        chat_id="-1004476874933",
        thread_id="3",
        user_id="8837347581",
        message_id="42",
    )

    assert session_db.offsets == [0, 200]
    assert ("session-3-late", "telegram_topic_closed") in session_db.ended


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
async def test_runner_private_dashboard_reply_uses_dm_identity_and_real_anchor() -> None:
    class FakeTelegramAdapter:
        async def handle_message(self, event) -> None:
            self.event = event

    adapter = FakeTelegramAdapter()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}

    await runner._dispatch_becky_agent_reply(
        chat_id="8837347581",
        thread_id="3964",
        session_id="session-1",
        text="Add the appointment.",
        reply_to_message_id="101",
    )

    event = adapter.event
    assert event.source.chat_type == "dm"
    assert GatewayRunner._reply_anchor_for_event(event) == "101"


def test_shortcut_source_uses_private_dm_identity_for_positive_chat_ids() -> None:
    store = SessionDBBeckyLoopsStore(EmptyDB())
    store._chat_id = "8837347581"

    assert store._shortcut_source("3964").chat_type == "dm"


@pytest.mark.asyncio
async def test_runner_propagates_the_narrow_auto_close_policy_in_event_metadata() -> None:
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
        text="Add the appointment.",
        reply_to_message_id="101",
        auto_close_policy="simple_calendar_todoist_success",
        new_topic=True,
    )

    assert adapter.events[0].metadata == {
        "becky_dashboard_reply": True,
        "becky_dashboard_new_topic": True,
        "becky_auto_close_policy": "simple_calendar_todoist_success",
    }


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
                origin=SimpleNamespace(
                    user_id="8837347581",
                    user_name="Cory",
                    user_id_alt=None,
                ),
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
            "user_id": "8837347581",
            "session_key": "agent:main:telegram:group:-1004476874933:3964",
            "session_id": "session-1",
        }
    ]
    assert adapter.event.text == "Please turn on the porch light."
    assert adapter.event.reply_to_message_id == "101"
    assert adapter.event.source.user_id == "8837347581"
    assert adapter.event.source.user_name == "Cory"


@pytest.mark.asyncio
async def test_runner_does_not_publish_bridge_after_shutdown_begins(monkeypatch) -> None:
    config = becky_loops.BeckyLoopsConfig(
        enabled=True,
        chat_id="8837347581",
        token="t" * 64,
        port=0,
    )
    bridge = object()
    stopped: list[object] = []
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._session_db = SimpleNamespace(_db=object())
    runner._startup_should_abort = lambda: True

    async def stop_bridge(value) -> None:
        stopped.append(value)

    async def start_bridge(**kwargs):
        del kwargs
        return bridge

    monkeypatch.setattr(becky_loops, "load_becky_loops_config", lambda: config)
    monkeypatch.setattr(becky_loops, "start_becky_loops_bridge", start_bridge)
    monkeypatch.setattr(becky_loops, "stop_becky_loops_bridge", stop_bridge)

    await runner._start_becky_loops_bridge()

    assert stopped == [bridge]
    assert not hasattr(runner, "_becky_loops_bridge")


@pytest.mark.asyncio
async def test_runner_registers_auto_close_after_a_successful_simple_action() -> None:
    adapter = PostDeliveryAdapter()
    controller = CloseController()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._session_run_generation = {
        "agent:main:telegram:group:-1004476874933:3964": 7,
    }
    runner._is_session_run_current = lambda _session_key, generation: generation == 7
    runner._becky_loops_config = BeckyLoopsConfig(
        enabled=True,
        chat_id="-1004476874933",
        token="t" * 64,
    )
    runner._becky_loops_topic_controller = controller
    runner._session_db = None

    notifications: list[dict[str, object]] = []

    async def notify(**kwargs: object) -> str:
        notifications.append(kwargs)
        return "archived"

    runner._notify_becky_topic_closed = notify
    event = MessageEvent(
        text="Done.",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1004476874933",
            chat_type="group",
            thread_id="3964",
        ),
        metadata={
            "becky_dashboard_new_topic": True,
            "becky_auto_close_policy": "simple_calendar_todoist_success",
        },
        internal=True,
    )

    runner._register_becky_auto_close_after_delivery(
        event=event,
        source=event.source,
        session_key="agent:main:telegram:group:-1004476874933:3964",
        run_generation=7,
        agent_result={
            "completed": True,
            "failed": False,
            "partial": False,
            "interrupted": False,
            "final_response": "Done.",
            "turn_exit_reason": "text_response(finish_reason=stop)",
            "turn_tool_events": [
                {
                    "name": "mcp_todoist_add_tasks",
                    "requested_name": "mcp_todoist_add_tasks",
                    "success": True,
                    "arguments": {},
                }
            ],
        },
    )

    assert len(adapter.callbacks) == 1
    assert adapter.callbacks[0][0].endswith(":3964")
    assert adapter.callbacks[0][2] == 7
    event._hermes_delivery_succeeded = True
    await adapter.callbacks[0][1]()

    assert controller.calls == [("-1004476874933", "3964")]
    assert notifications[0]["source_ref"].startswith("loop_")


@pytest.mark.asyncio
async def test_runner_keeps_topic_open_when_delivery_was_not_confirmed() -> None:
    adapter = PostDeliveryAdapter()
    controller = CloseController()
    runner = object.__new__(GatewayRunner)
    session_key = "agent:main:telegram:group:-1004476874933:3964"
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._session_run_generation = {session_key: 7}
    runner._becky_loops_config = BeckyLoopsConfig(
        enabled=True,
        chat_id="-1004476874933",
        token="t" * 64,
    )
    runner._becky_loops_topic_controller = controller
    runner._session_db = None

    event = MessageEvent(
        text="Done.",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1004476874933",
            chat_type="group",
            thread_id="3964",
        ),
        metadata={
            "becky_dashboard_new_topic": True,
            "becky_auto_close_policy": "simple_calendar_todoist_success",
        },
        internal=True,
    )
    runner._register_becky_auto_close_after_delivery(
        event=event,
        source=event.source,
        session_key=session_key,
        run_generation=7,
        agent_result={
            "completed": True,
            "failed": False,
            "partial": False,
            "interrupted": False,
            "final_response": "Done.",
            "turn_exit_reason": "text_response(finish_reason=stop)",
            "turn_tool_events": [
                {
                    "name": "mcp_todoist_add_tasks",
                    "requested_name": "mcp_todoist_add_tasks",
                    "success": True,
                    "arguments": {},
                }
            ],
        },
    )

    assert len(adapter.callbacks) == 1
    await adapter.callbacks[0][1]()

    assert controller.calls == []


@pytest.mark.asyncio
async def test_runner_keeps_topic_open_when_a_newer_generation_exists() -> None:
    adapter = PostDeliveryAdapter()
    controller = CloseController()
    runner = object.__new__(GatewayRunner)
    session_key = "agent:main:telegram:group:-1004476874933:3964"
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._session_run_generation = {session_key: 8}
    runner._becky_loops_config = BeckyLoopsConfig(
        enabled=True,
        chat_id="-1004476874933",
        token="t" * 64,
    )
    runner._becky_loops_topic_controller = controller
    runner._session_db = None
    event = MessageEvent(
        text="Done.",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1004476874933",
            chat_type="group",
            thread_id="3964",
        ),
        metadata={
            "becky_dashboard_new_topic": True,
            "becky_auto_close_policy": "simple_calendar_todoist_success",
        },
        internal=True,
    )

    runner._register_becky_auto_close_after_delivery(
        event=event,
        source=event.source,
        session_key=session_key,
        run_generation=7,
        agent_result={
            "completed": True,
            "final_response": "Done.",
            "turn_exit_reason": "text_response(finish_reason=stop)",
            "turn_tool_events": [
                {
                    "name": "mcp_todoist_add_tasks",
                    "requested_name": "mcp_todoist_add_tasks",
                    "success": True,
                }
            ],
        },
    )
    event._hermes_delivery_succeeded = True

    await adapter.callbacks[0][1]()

    assert controller.calls == []


@pytest.mark.asyncio
async def test_runner_keeps_topic_open_when_a_followup_is_pending() -> None:
    adapter = PostDeliveryAdapter()
    adapter._pending_messages = {}
    controller = CloseController()
    runner = object.__new__(GatewayRunner)
    session_key = "agent:main:telegram:group:-1004476874933:3964"
    adapter._pending_messages[session_key] = object()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._session_run_generation = {session_key: 7}
    runner._becky_loops_config = BeckyLoopsConfig(
        enabled=True,
        chat_id="-1004476874933",
        token="t" * 64,
    )
    runner._becky_loops_topic_controller = controller
    runner._session_db = None
    event = MessageEvent(
        text="Done.",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1004476874933",
            chat_type="group",
            thread_id="3964",
        ),
        metadata={
            "becky_dashboard_new_topic": True,
            "becky_auto_close_policy": "simple_calendar_todoist_success",
        },
        internal=True,
    )

    runner._register_becky_auto_close_after_delivery(
        event=event,
        source=event.source,
        session_key=session_key,
        run_generation=7,
        agent_result={
            "completed": True,
            "final_response": "Done.",
            "turn_exit_reason": "text_response(finish_reason=stop)",
            "turn_tool_events": [
                {
                    "name": "mcp_todoist_add_tasks",
                    "requested_name": "mcp_todoist_add_tasks",
                    "success": True,
                }
            ],
        },
    )
    event._hermes_delivery_succeeded = True

    await adapter.callbacks[0][1]()

    assert controller.calls == []


@pytest.mark.asyncio
async def test_runner_does_not_register_auto_close_for_failed_action() -> None:
    adapter = PostDeliveryAdapter()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    event = MessageEvent(
        text="It failed.",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1004476874933",
            chat_type="group",
            thread_id="3964",
        ),
        internal=True,
    )

    runner._register_becky_auto_close_after_delivery(
        event=event,
        source=event.source,
        session_key="agent:main:telegram:group:-1004476874933:3964",
        run_generation=7,
        agent_result={
            "completed": True,
            "failed": False,
            "partial": False,
            "interrupted": False,
            "final_response": "It failed.",
            "turn_exit_reason": "text_response(finish_reason=stop)",
            "turn_tool_events": [
                {
                    "name": "mcp_todoist_add_tasks",
                    "requested_name": "mcp_todoist_add_tasks",
                    "success": False,
                    "arguments": {},
                }
            ],
        },
    )

    assert adapter.callbacks == []


@pytest.mark.asyncio
async def test_runner_does_not_register_auto_close_without_new_topic_provenance() -> None:
    adapter = PostDeliveryAdapter()
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    event = MessageEvent(
        text="Done.",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1004476874933",
            chat_type="group",
            thread_id="3964",
        ),
        metadata={
            "becky_auto_close_policy": "simple_calendar_todoist_success",
        },
        internal=True,
    )

    runner._register_becky_auto_close_after_delivery(
        event=event,
        source=event.source,
        session_key="agent:main:telegram:group:-1004476874933:3964",
        run_generation=7,
        agent_result={
            "completed": True,
            "failed": False,
            "partial": False,
            "interrupted": False,
            "final_response": "Done.",
            "turn_exit_reason": "text_response(finish_reason=stop)",
            "turn_tool_events": [
                {
                    "name": "mcp_todoist_add_tasks",
                    "requested_name": "mcp_todoist_add_tasks",
                    "success": True,
                    "arguments": {},
                }
            ],
        },
    )

    assert adapter.callbacks == []


@pytest.mark.asyncio
async def test_auto_close_keeps_topic_open_when_topic_control_fails() -> None:
    controller = FailingCloseController()
    runner = object.__new__(GatewayRunner)
    runner._becky_loops_config = BeckyLoopsConfig(
        enabled=True,
        chat_id="-1004476874933",
        token="t" * 64,
    )
    runner._becky_loops_topic_controller = controller
    runner._session_db = None
    runner._notify_becky_topic_closed = pytest.fail

    await runner._maybe_auto_close_becky_topic(
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1004476874933",
            chat_type="group",
            thread_id="3964",
        ),
        agent_result={
            "completed": True,
            "failed": False,
            "partial": False,
            "interrupted": False,
            "final_response": "Done.",
            "turn_exit_reason": "text_response(finish_reason=stop)",
            "turn_tool_events": [
                {
                    "name": "google_calendar_create_event",
                    "requested_name": "google_calendar_create_event",
                    "success": True,
                }
            ],
        },
    )

    assert controller.calls == [("-1004476874933", "3964")]


@pytest.mark.asyncio
async def test_internal_dashboard_reply_is_not_dropped_by_busy_auth_gate() -> None:
    runner = object.__new__(GatewayRunner)
    runner._is_user_authorized = lambda _source: False

    event = MessageEvent(
        text="follow up",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1004476874933",
            chat_type="group",
            thread_id="3964",
        ),
        internal=True,
    )

    assert await runner._handle_active_session_busy_message(event, "session-1") is False


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

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.native_turn_sources import NativeSessionView, NativeTurnAdmission, NativeTurnLease
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


class Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent = []

    @property
    def name(self):
        return "telegram"

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def native_event(tmp_path, source, *, commit=lambda _session: True, abort=lambda *_: None):
    event = MessageEvent(
        text="native continuation",
        message_type=MessageType.TEXT,
        source=source,
        internal=True,
        allow_gateway_control=False,
    )
    event._native_turn_admission = NativeTurnAdmission(
        NativeTurnLease("lease", event.text, commit, abort),
        NativeSessionView(
            profile="default",
            profile_home=tmp_path,
            session_id="session",
            surface="gateway",
            session_key="telegram:dm:123",
            owner_token="owner",
        ),
    )
    return event


@pytest.mark.asyncio
async def test_native_admission_reserves_guard_before_handler_and_commits_once(tmp_path):
    adapter = Adapter()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm")
    key = adapter._event_session_key(MessageEvent(text="", source=source))
    calls = []

    async def handler(event):
        assert key in adapter._active_sessions
        assert event._native_turn_admission.commit() is True
        calls.append(event.text)
        return None

    adapter.set_message_handler(handler)
    event = native_event(tmp_path, source)

    assert await adapter.admit_native_turn(event, key) is True
    assert event._gateway_accepted is True
    task = adapter._session_tasks[key]
    await task

    assert calls == ["native continuation"]
    assert event._native_turn_admission.state == "committed"
    assert key not in adapter._active_sessions


@pytest.mark.asyncio
@pytest.mark.parametrize("busy_kind", ["active", "pending"])
async def test_native_admission_never_queues_behind_busy_or_human_turn(tmp_path, busy_kind):
    adapter = Adapter()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm")
    key = adapter._event_session_key(MessageEvent(text="", source=source))
    human = MessageEvent(text="human", source=source)
    adapter.set_message_handler(lambda _event: None)
    if busy_kind == "active":
        adapter._active_sessions[key] = asyncio.Event()
    else:
        adapter._pending_messages[key] = human
    event = native_event(tmp_path, source)

    assert await adapter.admit_native_turn(event, key) is False
    assert event._gateway_accepted is False
    assert adapter._pending_messages.get(key) is (human if busy_kind == "pending" else None)
    assert event._native_turn_admission.state == "open"


@pytest.mark.asyncio
async def test_task_failure_before_commit_aborts_not_sent_once(tmp_path):
    adapter = Adapter()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm")
    key = adapter._event_session_key(MessageEvent(text="", source=source))
    aborts = []

    async def handler(_event):
        raise RuntimeError("preflight")

    adapter.set_message_handler(handler)
    event = native_event(tmp_path, source, abort=lambda outcome, reason: aborts.append((outcome, reason)))

    assert await adapter.admit_native_turn(event, key) is True
    await adapter._session_tasks[key]

    assert event._native_turn_admission.state == "not_sent"
    assert len(aborts) == 1
    assert aborts[0][0] == "not_sent"


@pytest.mark.asyncio
async def test_gateway_watcher_polls_registered_profile_and_reserves_host_slot(tmp_path):
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="room",
        user_id="user",
        user_name="User",
        profile="default",
    )
    adapter = Adapter()
    session_key = adapter._event_session_key(MessageEvent(text="", source=source))
    entry = SimpleNamespace(origin=source, session_id="session", session_key=session_key)
    view = NativeSessionView(
        profile="default",
        profile_home=tmp_path,
        session_id="session",
        surface="gateway",
        compression_lineage=("session",),
        session_key=session_key,
        owner_token=f"gateway:default:{session_key}",
        metadata={"platform": "discord"},
    )
    committed = []

    class Source:
        name = "gateway-test-source"

        def poll(self, exact):
            if exact != view:
                return None
            return NativeTurnLease(
                lease_id="lease",
                prompt="continue",
                commit=lambda session: committed.append(session) or True,
                abort=lambda _outcome, _reason: None,
            )

        def on_user_boundary(self, _session, _reason):
            return None

    registration = PluginContext(
        PluginManifest(name="gateway-native-test"),
        PluginManager(scope_key=str(tmp_path)),
    ).register_native_turn_source(Source(), surfaces=("gateway",))

    class Store:
        pass

    class AsyncStore:
        def __init__(self, store):
            self._store = store

        async def list_sessions(self):
            return [entry]

    runner = object.__new__(GatewayRunner)
    runner.session_store = Store()
    runner._async_session_store = AsyncStore(runner.session_store)
    runner._adapter_for_source = lambda _source: adapter

    async def exact_view(_entry):
        return view

    runner._native_gateway_view = exact_view

    async def handler(event):
        assert session_key in adapter._active_sessions
        assert event.metadata["gateway_session_key"] == session_key
        assert event.metadata["gateway_session_id"] == "session"
        assert event.metadata["gateway_session_strict"] is True
        assert event._native_turn_admission.commit() is True

    adapter.set_message_handler(handler)
    try:
        assert await runner._poll_native_turn_sources_once() is True
        await adapter._session_tasks[session_key]
    finally:
        registration.dispose()

    assert committed == [view]

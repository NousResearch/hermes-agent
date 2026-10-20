from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource


class _RecordingAdapter(BasePlatformAdapter):
    def __init__(self, calls: list[tuple]):
        super().__init__(PlatformConfig(enabled=True, typing_indicator=True), Platform.SLACK)
        self.calls = calls
        self.typing_started = asyncio.Event()
        self.stopped = asyncio.Event()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None) -> SendResult:
        self.calls.append(("adapter-send", chat_id, metadata))
        return SendResult(success=True, message_id="adapter-send")

    async def get_chat_info(self, chat_id: str):
        return {"name": chat_id, "type": "group"}

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        self.calls.append(("typing", chat_id, metadata))
        self.typing_started.set()

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        self.calls.append(("stop", chat_id, metadata))
        self.stopped.set()


class _Transport:
    def __init__(self, adapter: _RecordingAdapter, calls: list[tuple]):
        self.adapter = adapter
        self.is_relay = False
        self.calls = calls

    async def send(self, logical_platform, chat_id: str, content: str, metadata=None):
        self.calls.append(("send", logical_platform, chat_id, content, metadata))
        return SimpleNamespace(success=True)


def _runner(adapter: _RecordingAdapter, transport: _Transport) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.SLACK: PlatformConfig(enabled=True, typing_indicator=True)}
    )
    runner.config.platforms[Platform.SLACK].home_channel = HomeChannel(
        platform=Platform.SLACK, chat_id="C-home", name="home", thread_id="T-home"
    )
    runner.adapters = {Platform.SLACK: adapter}
    runner._profile_adapters = {}
    runner._session_db = None
    runner._evict_cached_agent = lambda _key: None
    runner._release_running_agent_state = lambda _key: None

    store = SimpleNamespace()
    store.get_or_create_session = AsyncMock(
        return_value=SessionEntry(
            session_key="old", session_id="old-session", created_at=datetime.now(),
            updated_at=datetime.now(), platform=Platform.SLACK, chat_type="thread",
        )
    )
    store.switch_session = AsyncMock(
        return_value=SessionEntry(
            session_key="agent:main:slack:thread:C-home:T-home", session_id="cli-session",
            created_at=datetime.now(), updated_at=datetime.now(),
            platform=Platform.SLACK, chat_type="thread",
        )
    )
    runner.session_store = store
    runner._async_session_store = SimpleNamespace(
        _store=store,
        get_or_create_session=store.get_or_create_session,
        switch_session=store.switch_session,
    )

    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C-home",
        chat_type="thread",
        user_id="system:handoff",
        user_name="Handoff",
        thread_id="T-new",
    )
    dest = GatewayRunner._HandoffDestination(
        platform=Platform.SLACK,
        platform_name="slack",
        transport=transport,
        home=runner.config.platforms[Platform.SLACK].home_channel,
        home_chat_id="C-home",
        effective_thread_id="T-new",
        source=source,
        handoff_config=runner.config,
    )
    runner._handoff_resolve_destination = __import__("unittest.mock").mock.AsyncMock(
        return_value=dest
    )
    return runner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected_send_count"),
    [("text", 1), ("empty", 0), ("error", 0)],
)
async def test_handoff_dispatch_uses_adapter_typing_lifecycle(outcome, expected_send_count):
    calls: list[tuple] = []
    adapter = _RecordingAdapter(calls)
    transport = _Transport(adapter, calls)
    runner = _runner(adapter, transport)

    async def _handle_message(event: MessageEvent):
        calls.append(("handle", event.source.chat_id, event.source.thread_id))
        await asyncio.wait_for(adapter.typing_started.wait(), timeout=0.2)
        if outcome == "error":
            raise RuntimeError("boom")
        return "handoff response" if outcome == "text" else None

    runner._handle_message = _handle_message

    if outcome == "error":
        with pytest.raises(RuntimeError, match="boom"):
            await runner._process_handoff({"id": "cli-session", "title": "work", "handoff_platform": "slack"})
    else:
        await runner._process_handoff({"id": "cli-session", "title": "work", "handoff_platform": "slack"})

    send_calls = [call for call in calls if call[0] == "send"]
    assert len(send_calls) == expected_send_count
    stop_calls = [call for call in calls if call[0] == "stop"]
    assert stop_calls, calls
    first_typing = next(i for i, call in enumerate(calls) if call[0] == "typing")
    first_handle = next(i for i, call in enumerate(calls) if call[0] == "handle")
    first_stop = next(i for i, call in enumerate(calls) if call[0] == "stop")
    assert first_typing > first_handle
    assert first_stop > first_typing
    if send_calls:
        assert first_stop < calls.index(send_calls[0])
    assert stop_calls[-1][1] == "C-home"
    assert stop_calls[-1][2]["thread_id"] == "T-new"


@pytest.mark.asyncio
async def test_handoff_dispatch_skips_cache_dependent_relay_typing_lifecycle():
    calls: list[tuple] = []
    adapter = _RecordingAdapter(calls)
    transport = _Transport(adapter, calls)
    transport.is_relay = True
    runner = _runner(adapter, transport)
    runner._handle_message = AsyncMock(return_value="handoff response")

    await runner._process_handoff(
        {"id": "cli-session", "title": "work", "handoff_platform": "slack"}
    )

    assert not [call for call in calls if call[0] in ("typing", "stop")]
    assert len([call for call in calls if call[0] == "send"]) == 1

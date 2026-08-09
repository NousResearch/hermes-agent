from __future__ import annotations

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.contextual_cron import ContextualCronGateway, ContextualCronQueueItem
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _StubAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _adapter() -> _StubAdapter:
    return _StubAdapter(
        PlatformConfig(enabled=True, token="test"),
        Platform.TELEGRAM,
    )


def _event(text: str = "human follow-up") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="42",
            chat_type="dm",
            user_id="42",
        ),
    )


def _key() -> str:
    return build_session_key(_event().source)


@pytest.mark.asyncio
async def test_contextual_guard_waits_on_adapter_guard_without_polling():
    adapter = _adapter()
    key = _key()
    human_guard = asyncio.Event()
    adapter._active_sessions[key] = human_guard
    owner = asyncio.current_task()
    assert owner is not None
    adapter._session_tasks[key] = owner

    waiter = asyncio.create_task(adapter.acquire_contextual_cron_guard(key))
    await asyncio.sleep(0)
    assert not waiter.done()

    adapter._release_session_guard(key, guard=human_guard)
    cron_guard = await asyncio.wait_for(waiter, timeout=1)

    assert adapter._active_sessions[key] is cron_guard
    await adapter.release_contextual_cron_guard(key, cron_guard)
    assert key not in adapter._active_sessions


@pytest.mark.asyncio
async def test_contextual_release_flushes_debounce_and_hands_human_turn_first(monkeypatch):
    adapter = _adapter()
    key = _key()
    cron_guard = await adapter.acquire_contextual_cron_guard(key)
    adapter._busy_text_debounce_seconds = 60.0
    adapter._busy_text_hard_cap_seconds = 60.0
    await adapter._queue_text_debounce(key, _event())

    handed_off: list[MessageEvent] = []

    def start_human(event: MessageEvent, session_key: str, **_kwargs) -> bool:
        assert session_key == key
        handed_off.append(event)
        adapter._active_sessions[key] = asyncio.Event()
        return True

    monkeypatch.setattr(adapter, "_start_session_processing", start_human)
    await adapter.release_contextual_cron_guard(key, cron_guard)

    assert [event.text for event in handed_off] == ["human follow-up"]
    assert key in adapter._active_sessions
    assert key not in adapter._pending_messages
    assert key not in adapter._text_debounce


def _queue_item() -> ContextualCronQueueItem:
    loop = asyncio.get_running_loop()
    return ContextualCronQueueItem(
        job_id="job",
        execution_id="execution",
        prompt="check",
        session_key=_key(),
        admitted_session_id="session-1",
        admitted_routing_revision=0,
        source=_event().source,
        future=loop.create_future(),
    )


@pytest.mark.asyncio
async def test_degraded_turn_lease_releases_contextual_adapter_guard():
    adapter = _adapter()

    class Leases:
        async def acquire(self, *_args, **_kwargs):
            return type("Token", (), {"degraded": True})()

    runner = type(
        "Runner",
        (),
        {
            "_turn_leases": Leases(),
            "_adapter_for_source": lambda self, _source: adapter,
        },
    )()
    gateway = ContextualCronGateway(runner)
    outcome = await gateway._run_queued_item(_queue_item())

    assert outcome.kind == "retryable"
    assert _key() not in adapter._active_sessions


@pytest.mark.asyncio
async def test_raising_turn_lease_releases_contextual_adapter_guard():
    adapter = _adapter()

    class Leases:
        async def acquire(self, *_args, **_kwargs):
            raise RuntimeError("lease failed")

    runner = type(
        "Runner",
        (),
        {
            "_turn_leases": Leases(),
            "_adapter_for_source": lambda self, _source: adapter,
        },
    )()
    gateway = ContextualCronGateway(runner)

    with pytest.raises(RuntimeError, match="lease failed"):
        await gateway._run_queued_item(_queue_item())
    assert _key() not in adapter._active_sessions


@pytest.mark.asyncio
async def test_cancelled_turn_lease_wait_releases_contextual_adapter_guard():
    adapter = _adapter()
    entered = asyncio.Event()

    class Leases:
        async def acquire(self, *_args, **_kwargs):
            entered.set()
            await asyncio.Event().wait()

    runner = type(
        "Runner",
        (),
        {
            "_turn_leases": Leases(),
            "_adapter_for_source": lambda self, _source: adapter,
        },
    )()
    gateway = ContextualCronGateway(runner)
    task = asyncio.create_task(gateway._run_queued_item(_queue_item()))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _key() not in adapter._active_sessions

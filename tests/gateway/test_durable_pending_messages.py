from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, PendingMessageStore
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _Adapter:
    platform = Platform.TELEGRAM

    def __init__(self):
        self._pending_messages = PendingMessageStore()
        self._text_debounce = {}
        self.handled: list[MessageEvent] = []

    def set_pending_message_change_callback(self, callback):
        self._pending_messages.set_change_callback(callback)

    async def handle_message(self, event: MessageEvent):
        self.handled.append(event)


class _BlockingAdapter(_Adapter):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def handle_message(self, event: MessageEvent):
        self.started.set()
        await self.release.wait()
        await super().handle_message(event)


class _FailingAdapter(_Adapter):
    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()

    async def handle_message(self, event: MessageEvent):
        self.started.set()
        raise RuntimeError("boom")


def _runner(tmp_path, adapter: _Adapter) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._queued_events = {}
    runner._running_agents = {}
    runner._background_tasks = set()
    runner._durable_pending_path = tmp_path / "gateway_pending_messages.json"
    return runner


def _event(text: str = "queued follow-up") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-1",
            chat_type="dm",
            user_id="u1",
        ),
        message_id="m1",
    )


def _payload_has_pending(runner: GatewayRunner, session_key: str) -> bool:
    return session_key in runner._read_durable_pending_payload().get("pending", {})


def test_pending_message_store_callback_persists_snapshot(tmp_path):
    adapter = _Adapter()
    runner = _runner(tmp_path, adapter)
    adapter.set_pending_message_change_callback(runner._save_durable_pending_messages)

    adapter._pending_messages["session-1"] = _event("do not lose this")

    payload = runner._read_durable_pending_payload()
    assert payload["pending"]["session-1"]["text"] == "do not lose this"
    assert payload["pending"]["session-1"]["source"]["platform"] == "telegram"


def test_restores_durable_pending_messages_for_adapter(tmp_path):
    adapter = _Adapter()
    runner = _runner(tmp_path, adapter)
    adapter._pending_messages["session-1"] = _event("restore me")
    runner._save_durable_pending_messages()

    restored_adapter = _Adapter()
    restored_runner = _runner(tmp_path, restored_adapter)

    assert restored_runner._restore_durable_pending_messages_for_adapter(restored_adapter) == 1
    assert restored_adapter._pending_messages["session-1"].text == "restore me"


def test_saves_text_debounce_buffer_as_durable_pending(tmp_path):
    adapter = _Adapter()
    runner = _runner(tmp_path, adapter)
    adapter._text_debounce["session-1"] = SimpleNamespace(event=_event("debounced"))

    runner._save_durable_pending_messages()

    payload = runner._read_durable_pending_payload()
    assert payload["pending"]["session-1"]["text"] == "debounced"


@pytest.mark.asyncio
async def test_schedules_restored_pending_when_session_not_resuming(tmp_path):
    adapter = _Adapter()
    runner = _runner(tmp_path, adapter)
    adapter._pending_messages["session-1"] = _event("run me")
    runner._is_session_resume_pending = lambda session_key: False

    assert runner._schedule_durable_pending_messages() == 1
    await asyncio.wait_for(asyncio.gather(*list(runner._background_tasks)), timeout=1)

    assert [event.text for event in adapter.handled] == ["run me"]
    assert "session-1" not in adapter._pending_messages


@pytest.mark.asyncio
async def test_restored_pending_snapshot_survives_until_handle_completes(tmp_path):
    adapter = _BlockingAdapter()
    runner = _runner(tmp_path, adapter)
    adapter.set_pending_message_change_callback(runner._save_durable_pending_messages)
    adapter._pending_messages["session-1"] = _event("run me durably")
    runner._is_session_resume_pending = lambda session_key: False

    assert runner._schedule_durable_pending_messages() == 1
    await asyncio.wait_for(adapter.started.wait(), timeout=1)

    assert "session-1" not in adapter._pending_messages
    assert _payload_has_pending(runner, "session-1")

    adapter.release.set()
    await asyncio.wait_for(asyncio.gather(*list(runner._background_tasks)), timeout=1)

    assert [event.text for event in adapter.handled] == ["run me durably"]
    assert not _payload_has_pending(runner, "session-1")


@pytest.mark.asyncio
async def test_restored_pending_reinserted_on_handle_failure(tmp_path):
    adapter = _FailingAdapter()
    runner = _runner(tmp_path, adapter)
    adapter.set_pending_message_change_callback(runner._save_durable_pending_messages)
    adapter._pending_messages["session-1"] = _event("retry me")
    runner._is_session_resume_pending = lambda session_key: False

    assert runner._schedule_durable_pending_messages() == 1
    await asyncio.wait_for(adapter.started.wait(), timeout=1)
    for _ in range(10):
        if "session-1" in adapter._pending_messages:
            break
        await asyncio.sleep(0)

    assert adapter._pending_messages["session-1"].text == "retry me"
    assert _payload_has_pending(runner, "session-1")


@pytest.mark.asyncio
async def test_resume_pending_skip_keeps_snapshot_then_later_drains(tmp_path):
    adapter = _Adapter()
    runner = _runner(tmp_path, adapter)
    adapter.set_pending_message_change_callback(runner._save_durable_pending_messages)
    adapter._pending_messages["session-1"] = _event("after resume")

    runner._is_session_resume_pending = lambda session_key: True
    assert runner._schedule_durable_pending_messages() == 0
    assert _payload_has_pending(runner, "session-1")
    assert adapter.handled == []

    runner._is_session_resume_pending = lambda session_key: False
    assert runner._schedule_durable_pending_messages() == 1
    await asyncio.wait_for(asyncio.gather(*list(runner._background_tasks)), timeout=1)

    assert [event.text for event in adapter.handled] == ["after resume"]
    assert not _payload_has_pending(runner, "session-1")


def test_corrupt_durable_pending_snapshot_is_quarantined(tmp_path):
    adapter = _Adapter()
    runner = _runner(tmp_path, adapter)
    runner._durable_pending_path.write_text("{not-json", encoding="utf-8")

    assert runner._read_durable_pending_payload() == {"pending": {}, "queued": {}}

    assert not runner._durable_pending_path.exists()
    assert list(tmp_path.glob("gateway_pending_messages.json.corrupt.*"))

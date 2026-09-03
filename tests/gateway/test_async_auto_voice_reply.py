"""Regression coverage for non-blocking, ordered gateway auto voice replies."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    SendResult,
)
from gateway.session import SessionEntry, SessionSource


SESSION_KEY = "agent:main:telegram:group:-1001:12345"


class _DeliveryAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.send_started = asyncio.Event()
        self.release_send = asyncio.Event()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.send_started.set()
        await self.release_send.wait()
        return SendResult(success=True, message_id="text-reply")


def _source(chat_id: str = "-1001") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="group",
        user_id="12345",
    )


def _event(chat_id: str = "-1001") -> MessageEvent:
    return MessageEvent(
        text="speak this reply",
        source=_source(chat_id),
        message_id=f"message:{chat_id}",
    )


def _handler_runner(monkeypatch, tmp_path):
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    monkeypatch.setattr(runner, "_is_user_authorized", lambda _source: True)
    monkeypatch.setattr(runner, "_set_session_env", lambda _context: None)
    monkeypatch.setattr(
        runner,
        "_handle_active_session_busy_message",
        AsyncMock(return_value=False),
    )
    runner._session_db = MagicMock()
    monkeypatch.setattr(
        runner, "_recover_telegram_topic_thread_id", lambda _source: None
    )
    monkeypatch.setattr(
        runner, "_cache_session_source", lambda _key, _source: None
    )
    monkeypatch.setattr(
        runner, "_is_session_run_current", lambda _key, _generation: True
    )
    monkeypatch.setattr(runner, "_reply_anchor_for_event", lambda _event: None)
    monkeypatch.setattr(runner, "_get_guild_id", lambda _event: None)
    monkeypatch.setattr(
        runner, "_should_send_voice_reply", lambda *_args, **_kwargs: True
    )
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=SESSION_KEY,
        session_id="session-async-voice",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    monkeypatch.setattr(
        runner,
        "_run_agent",
        AsyncMock(
            return_value={
                "final_response": "The text reply",
                "messages": [
                    {"role": "user", "content": "speak this reply"},
                    {"role": "assistant", "content": "The text reply"},
                ],
                "tools": [],
                "history_offset": 0,
                "last_prompt_tokens": 0,
                "api_calls": 1,
                "failed": False,
            }
        ),
    )

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


@pytest.mark.asyncio
async def test_agent_handler_returns_before_auto_voice_synthesis(
    monkeypatch, tmp_path
):
    runner = _handler_runner(monkeypatch, tmp_path)
    voice_started = asyncio.Event()
    release_voice = asyncio.Event()

    async def slow_voice(_event, _text):
        voice_started.set()
        await release_voice.wait()

    runner._send_voice_reply = AsyncMock(side_effect=slow_voice)
    event = _event()

    response = await asyncio.wait_for(
        runner._handle_message_with_agent(event, event.source, SESSION_KEY, 1),
        timeout=1,
    )

    assert response == "The text reply"
    assert not voice_started.is_set()

    getattr(event, "_hermes_auto_voice_text_delivered").set()
    await asyncio.wait_for(voice_started.wait(), timeout=1)
    release_voice.set()
    assert await runner._drain_voice_reply_tasks(timeout=1) is True


@pytest.mark.asyncio
async def test_base_adapter_releases_voice_only_after_text_send(monkeypatch):
    adapter = _DeliveryAdapter()
    event = _event()
    text_delivered = asyncio.Event()
    setattr(event, "_hermes_auto_voice_text_delivered", text_delivered)
    adapter.set_message_handler(AsyncMock(return_value="The text reply"))
    session_key = "agent:main:telegram:group:-1001"
    adapter._active_sessions[session_key] = asyncio.Event()
    monkeypatch.setattr("gateway.delivery_ledger.ledger_enabled", lambda: False)

    processing = asyncio.create_task(
        adapter._process_message_background(event, session_key)
    )
    await asyncio.wait_for(adapter.send_started.wait(), timeout=1)
    assert not text_delivered.is_set()

    adapter.release_send.set()
    await asyncio.wait_for(processing, timeout=1)
    assert text_delivered.is_set()


@pytest.mark.asyncio
async def test_auto_voice_reply_is_fifo_per_session():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._voice_reply_tasks = set()
    runner._voice_reply_tails = {}
    first_release = asyncio.Event()
    calls = []

    async def send_voice(_event, text):
        calls.append(f"start:{text}")
        if text == "first":
            await first_release.wait()
        calls.append(f"end:{text}")

    runner._send_voice_reply = send_voice
    first_gate = asyncio.Event()
    second_gate = asyncio.Event()
    first_gate.set()
    second_gate.set()

    first = runner._schedule_voice_reply(
        "same-session", _event(), "first", first_gate
    )
    second = runner._schedule_voice_reply(
        "same-session", _event(), "second", second_gate
    )
    await asyncio.sleep(0)

    assert calls == ["start:first"]
    first_release.set()
    await asyncio.gather(first, second)
    assert calls == ["start:first", "end:first", "start:second", "end:second"]
    await asyncio.sleep(0)
    assert runner._voice_reply_tasks == set()
    assert runner._voice_reply_tails == {}


def test_voice_delivery_key_ignores_per_user_session_partition():
    first = _event()
    second = _event()
    first.source.user_id = "user-one"
    second.source.user_id = "user-two"

    assert gateway_run.GatewayRunner._voice_reply_delivery_key(
        first
    ) == gateway_run.GatewayRunner._voice_reply_delivery_key(second)


@pytest.mark.asyncio
async def test_auto_voice_reply_does_not_serialize_unrelated_sessions():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._voice_reply_tasks = set()
    runner._voice_reply_tails = {}
    first_release = asyncio.Event()
    second_done = asyncio.Event()

    async def send_voice(event, _text):
        if event.source.chat_id == "first":
            await first_release.wait()
        else:
            second_done.set()

    runner._send_voice_reply = send_voice
    gate = asyncio.Event()
    gate.set()

    runner._schedule_voice_reply("session:first", _event("first"), "one", gate)
    runner._schedule_voice_reply("session:second", _event("second"), "two", gate)

    await asyncio.wait_for(second_done.wait(), timeout=1)
    first_release.set()
    assert await runner._drain_voice_reply_tasks(timeout=1) is True


@pytest.mark.asyncio
async def test_voice_reply_drain_cancels_tasks_at_deadline():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._voice_reply_tasks = set()
    runner._voice_reply_tails = {}
    started = asyncio.Event()

    async def send_voice(_event, _text):
        started.set()
        await asyncio.Event().wait()

    runner._send_voice_reply = send_voice
    gate = asyncio.Event()
    gate.set()
    task = runner._schedule_voice_reply("session", _event(), "slow", gate)
    await asyncio.wait_for(started.wait(), timeout=1)

    assert await runner._drain_voice_reply_tasks(timeout=0) is False
    assert task.cancelled()
    assert runner._voice_reply_tasks == set()
    assert runner._voice_reply_tails == {}

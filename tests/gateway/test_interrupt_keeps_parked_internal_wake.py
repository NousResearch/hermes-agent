"""Invariant: interrupting a session (/stop, /new, /reset) discards a parked human follow-up but
never a parked internal wake (async-delegation completion, notify+wake). See #114456.

Real ``BasePlatformAdapter`` pending slot + post-command drain, real
``GatewayRunner._interrupt_and_clear_session`` with the reason pairs its three callers pass.
"""
import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import _INTERRUPT_REASON_RESET, _INTERRUPT_REASON_STOP, GatewayRunner
from gateway.session import SessionSource

# (interrupt_reason, invalidation_reason) as passed by _busy_stop_command, _handle_stop_command
# (pending sentinel) and _busy_new_command.
_COMMAND_REASONS = [
    (_INTERRUPT_REASON_STOP, "stop_command"),
    (_INTERRUPT_REASON_STOP, "stop_command_pending"),
    (_INTERRUPT_REASON_RESET, "new_command"),
]


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.restarted = []

    @property
    def name(self):
        return "telegram"

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}

    def _start_session_processing(self, event, session_key, *, interrupt_event=None):
        self.restarted.append(event)
        return True


def _gateway():
    adapter = _Adapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.TELEGRAM: adapter}
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", chat_type="dm", user_id="u1")
    key = adapter._event_session_key(MessageEvent(text="", message_type=MessageType.TEXT, source=source))
    return adapter, runner, source, key


def _event(source, text, *, internal):
    event = MessageEvent(text=text, message_type=MessageType.TEXT, source=source, internal=internal)
    event._gateway_accepted = True
    return event


async def _run_command(runner, adapter, source, key, reasons):
    interrupt_reason, invalidation_reason = reasons
    command_guard = asyncio.Event()
    adapter._active_sessions[key] = command_guard
    await runner._interrupt_and_clear_session(
        key, source, interrupt_reason=interrupt_reason, invalidation_reason=invalidation_reason,
    )
    await adapter._drain_pending_after_session_command(key, command_guard)


@pytest.mark.asyncio
@pytest.mark.parametrize("reasons", _COMMAND_REASONS, ids=[r[1] for r in _COMMAND_REASONS])
async def test_interrupt_keeps_parked_internal_wake_and_discards_human_followup(reasons):
    adapter, runner, source, key = _gateway()
    wake = _event(source, "[ASYNC DELEGATION BATCH COMPLETE] 1 task done", internal=True)
    adapter._pending_messages[key] = wake
    await _run_command(runner, adapter, source, key, reasons)
    assert adapter.restarted == [wake]
    assert key not in adapter._pending_messages and key not in adapter._active_sessions

    adapter.restarted.clear()
    adapter._pending_messages[key] = _event(source, "stale human follow-up", internal=False)
    await _run_command(runner, adapter, source, key, reasons)
    assert adapter.restarted == []
    assert key not in adapter._pending_messages


@pytest.mark.asyncio
async def test_interrupt_promotes_internal_wake_queued_behind_discarded_human_head():
    adapter, runner, source, key = _gateway()
    adapter._pending_messages[key] = _event(source, "human head", internal=False)
    later_human = _event(source, "second human", internal=False)
    wake = _event(source, "[ASYNC DELEGATION BATCH COMPLETE] 1 task done", internal=True)
    runner._session_state(key).conversation.queued_events.extend([later_human, wake])
    await _run_command(runner, adapter, source, key, _COMMAND_REASONS[0])
    assert adapter.restarted == [wake]
    assert runner._overflow_queue(key) == [later_human]

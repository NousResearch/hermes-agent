"""Trusted wakes cross the real adapter busy boundary, not external-user auth."""
import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.wake import deliver_wake
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


class WakeAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.sent = []

    @property
    def name(self):
        return "telegram"

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(content)
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


async def unused_handler(event):
    raise AssertionError("busy event must not start a concurrent turn")


@pytest.mark.asyncio
@pytest.mark.parametrize("human_pending", [False, True])
async def test_completed_board_wake_is_admitted_without_changing_human_input(tmp_path, monkeypatch, human_pending):
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "board.db"))
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="wake receipt", assignee="worker")
        kb.complete_task(conn, tid, summary="completed result")
        task = kb.get_task(conn, tid)
    assert task is not None and task.status == "done"
    text = f"[Kanban task completed] {task.result}"
    adapter, runner, source, key = busy_gateway()
    human = MessageEvent(text="human follow-up", message_type=MessageType.TEXT, source=source)
    if human_pending:
        adapter._pending_messages[key] = human
    assert runner._is_user_authorized(source) is False
    await deliver_wake(adapter, source=source, text=text)
    if human_pending:
        assert adapter._pending_messages[key] is human
        assert human.text == "human follow-up"
        wake = runner._overflow_queue(key)[0]
    else:
        wake = adapter._pending_messages[key]
    assert wake.internal and wake.source.user_id is None
    assert wake.text == text
    assert not adapter.sent
    assert not adapter._active_sessions[key].is_set()


def busy_gateway():
    adapter = WakeAdapter()
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._draining = False
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm")
    key = adapter._event_session_key(MessageEvent(text="", message_type=MessageType.TEXT, source=source))
    adapter.set_message_handler(unused_handler)
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    adapter._active_sessions[key] = asyncio.Event()
    return adapter, runner, source, key


@pytest.mark.asyncio
async def test_identityless_external_input_is_still_rejected():
    adapter, runner, source, key = busy_gateway()
    for _ in range(2):
        await adapter.handle_message(MessageEvent(text="external", message_type=MessageType.TEXT, source=source))
    assert key not in adapter._pending_messages
    assert not runner._overflow_queue(key)
    assert not adapter.sent

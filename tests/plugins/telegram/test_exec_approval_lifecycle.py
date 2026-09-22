"""Real gateway approval waits and Telegram controls share one request lifetime."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.run_turn_runner import TurnRunner
from plugins.platforms.telegram.adapter import TelegramAdapter
from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision

SESSION = "agent:main:telegram:group:123:7"


@pytest.fixture
def surface(monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "1")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = Mock()
    monkeypatch.setattr(adapter, "pause_typing_for_chat", Mock())
    runner = object.__new__(TurnRunner)
    runner._ctx = SimpleNamespace(
        _status_adapter=adapter, _status_chat_id="123", _status_thread_metadata={"thread_id": "7"},
        session_key=SESSION, _run_still_current=lambda: False,
    )
    runner._close_native_stream_boundary = Mock()
    yield adapter, runner
    approval.unregister_gateway_notify(SESSION)


def query_for(data, message_id, *, user_id=1, chat_id=123):
    return SimpleNamespace(
        data=data, message=SimpleNamespace(message_id=message_id, chat_id=chat_id,
                                         chat=SimpleNamespace(type="group"), message_thread_id=7),
        from_user=SimpleNamespace(id=user_id, first_name="Tester"),
        answer=AsyncMock(), edit_message_text=AsyncMock(),
    )


async def start_wait(runner, name):
    runner._ctx._loop_for_step = asyncio.get_running_loop()
    return asyncio.create_task(asyncio.to_thread(
        _await_gateway_decision, SESSION, runner._approval_notify_sync,
        {"command": name, "description": "synthetic; never executed", "pattern_key": name},
    ))


@pytest.mark.asyncio
async def test_only_authorized_matching_card_resolves_its_request_after_parent_return(surface):
    adapter, runner = surface
    sent = asyncio.Queue()
    edits = asyncio.Queue()
    tasks = []

    async def send(**kwargs):
        message_id = 100 + sent.qsize() + len(adapter._approval_state)
        sent.put_nowait((message_id, kwargs["reply_markup"].inline_keyboard[0][0].callback_data))
        return SimpleNamespace(message_id=message_id)

    async def edit(chat_id, message_id, content, **kwargs):
        edits.put_nowait((message_id, content))
        return SimpleNamespace(success=True)

    adapter._bot.send_message.side_effect = send
    adapter.edit_message = edit
    try:
        for name in ("A", "B"):
            tasks.append(await start_wait(runner, name))
        cards = [await asyncio.wait_for(sent.get(), 5) for _ in tasks]
        pending = approval.list_gateway_approvals(SESSION)
        card_b = next((mid, data) for mid, data in cards
                      if adapter._approval_state[int(data.rsplit(":", 1)[1])].request_id == pending[1]["request_id"])
        mid, data = card_b
        for overrides in ({"user_id": 2}, {"chat_id": 456}, {"message_id": mid + 99}):
            q = query_for(data, **({"message_id": mid} | overrides))
            await adapter._handle_callback_query(SimpleNamespace(callback_query=q), None)
            assert len(approval.list_gateway_approvals(SESSION)) == 2
        q = query_for(data, mid)
        await adapter._handle_callback_query(SimpleNamespace(callback_query=q), None)
        assert [entry["request_id"] for entry in approval.list_gateway_approvals(SESSION)] == [pending[0]["request_id"]]
        await adapter._handle_callback_query(SimpleNamespace(callback_query=q), None)
        assert len(approval.list_gateway_approvals(SESSION)) == 1
        assert "no longer pending" in q.answer.call_args.kwargs["text"]
        approval.withdraw_gateway_approval(SESSION, pending[0]["request_id"], "owning child stopped")
        results = await asyncio.wait_for(asyncio.gather(*tasks), 5)
        assert sorted(result["choice"] or "withdrawn" for result in results) == ["once", "withdrawn"]
        _, notice = await asyncio.wait_for(edits.get(), 5)
        assert "withdrawn" in notice and "timed out" not in notice
        assert not adapter._approval_state
        adapter.pause_typing_for_chat.assert_not_called()
        runner._close_native_stream_boundary.assert_not_called()
    finally:
        approval.unregister_gateway_notify(SESSION)
        await asyncio.wait_for(asyncio.gather(*tasks), 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "withdrawn", "answered"])
async def test_late_card_ack_retires_to_committed_outcome_without_parent_turn(surface, monkeypatch, outcome):
    """A send can outlive the runner's send wait; terminal state must survive until its ack."""
    adapter, runner = surface
    started = threading.Event()
    release = asyncio.Event()
    edited = asyncio.Queue()
    task = None

    async def slow_send(**kwargs):
        started.set()
        await release.wait()
        return SimpleNamespace(message_id=321)

    async def edit(chat_id, message_id, content, **kwargs):
        edited.put_nowait(content)
        return SimpleNamespace(success=True)

    def ambiguous(_future, timeout):
        assert started.wait(5)
        return "ambiguous"

    adapter._bot.send_message.side_effect = slow_send
    adapter.edit_message = edit
    monkeypatch.setattr("gateway.run._approval_send_outcome", ambiguous)
    if outcome == "timeout":
        monkeypatch.setattr("tools.approval_context._get_approval_timeout", lambda: 0)
    try:
        task = await start_wait(runner, "late ack")
        assert await asyncio.to_thread(started.wait, 5)
        pending = approval.list_gateway_approvals(SESSION)
        if outcome == "withdrawn":
            approval.withdraw_gateway_approval(SESSION, pending[0]["request_id"], "child stopped")
        elif outcome == "answered":
            assert approval.resolve_gateway_approval(SESSION, "once", request_id=pending[0]["request_id"]) == 1
        result = await asyncio.wait_for(task, 5)
        release.set()
        notice = await asyncio.wait_for(edited.get(), 5)
        expected = {"timeout": "timed out", "withdrawn": "withdrawn", "answered": "answered elsewhere"}
        assert expected[outcome] in notice.lower()
        assert (result["choice"] == "once") is (outcome == "answered")
        assert not adapter._approval_state
        q = query_for("ea:once:1", 321)
        await adapter._handle_callback_query(SimpleNamespace(callback_query=q), None)
        assert "no longer pending" in q.answer.call_args.kwargs["text"]
        q.edit_message_text.assert_not_called()  # a stale tap must not overwrite the truthful terminal card
    finally:
        release.set()
        approval.unregister_gateway_notify(SESSION)
        if task is not None:
            await asyncio.wait_for(task, 5)

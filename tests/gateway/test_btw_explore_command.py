"""Messaging-gateway ``/btw --explore`` / ``--promote`` (gateway/slash_commands_btw.py).

An exploration forks the chat's live cached agent and sends its report to the chat; a promotion
becomes the chat's next turn through the adapter FIFO (or the local client's own submission), never
a write into the transcript.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import side_thread
from tests.gateway.test_background_command import _make_event, _make_runner


@pytest.fixture(autouse=True)
def _empty_registry():
    side_thread._registry.clear()
    yield
    side_thread._registry.clear()


def _runner(parent_agent):
    runner = _make_runner()
    store = AsyncMock()
    store.get_or_create_session.return_value = MagicMock(session_id="chat-sess")
    store._store = runner.session_store
    runner._async_session_store = store
    runner._cached_agent_for = MagicMock(return_value=parent_agent)
    runner._session_key_for_source = MagicMock(return_value="agent:main:telegram:dm:67890")
    runner._reply_metadata = MagicMock(return_value=None)
    runner.adapter = AsyncMock()
    runner.adapter._pending_messages = {}
    runner._delivery_adapter_for = MagicMock(return_value=runner.adapter)

    async def in_executor(func, *args):
        return func(*args)

    runner._run_in_executor_with_context = in_executor
    return runner


async def _drain(runner):
    for task in list(runner._background_tasks):
        await task


def _fake_run(thread, parent, history, **_kw):
    thread.tool_calls.append("search_files")
    thread._finish("done", answer="retry lives in net.py:40")
    return thread.answer


@pytest.mark.asyncio
async def test_explore_forks_the_live_agent_and_sends_the_report():
    parent = MagicMock(_session_messages=[{"role": "user", "content": "fix it"}])
    runner = _runner(parent)
    seen = {}

    def run(thread, agent, history, **kw):
        seen.update(agent=agent, history=history)
        return _fake_run(thread, agent, history)

    with patch("agent.side_thread.explore_refusal", return_value=None), \
         patch("agent.side_thread.run_side_exploration", side_effect=run):
        ack = await runner._handle_btw_command(_make_event(text="/btw --explore find the retry logic"))
        await _drain(runner)

    (thread,) = side_thread.list_threads("chat-sess")
    assert thread.thread_id in ack
    assert seen["agent"] is parent and seen["history"] == parent._session_messages
    sent = runner.adapter.send.call_args.args[1]
    assert "retry lives in net.py:40" in sent and f"/btw --promote {thread.thread_id}" in sent


@pytest.mark.asyncio
async def test_promote_enqueues_the_report_as_the_chats_next_turn():
    runner = _runner(MagicMock(_session_messages=[]))
    with patch("agent.side_thread.explore_refusal", return_value=None), \
         patch("agent.side_thread.run_side_exploration", side_effect=_fake_run):
        await runner._handle_btw_command(_make_event(text="/btw --explore find the retry logic"))
        await _drain(runner)
    (thread,) = side_thread.list_threads("chat-sess")

    ack = await runner._handle_btw_command(_make_event(text=f"/btw --promote {thread.thread_id} fix it"))

    assert thread.thread_id in ack
    (queued,) = runner.adapter._pending_messages.values()
    assert queued.text == side_thread.build_promotion_prompt(thread, "fix it")
    assert queued.source.chat_id == "67890"


@pytest.mark.asyncio
async def test_promote_inside_a_local_slash_exec_goes_to_the_clients_sink():
    from gateway.session_command_sink import capture

    runner = _runner(None)
    thread = side_thread.register_thread("chat-sess", "t")
    thread._finish("done", answer="report")
    with capture() as sink:
        await runner._handle_btw_command(_make_event(text="/btw --promote"))
    assert sink.prompt == side_thread.build_promotion_prompt(thread)
    assert runner.adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_explore_without_a_live_agent_is_refused_at_once():
    runner = _runner(None)
    reply = await runner._handle_btw_command(_make_event(text="/btw --explore look around"))
    assert "send a message first" in reply.lower()
    assert side_thread.list_threads("chat-sess") == []
    assert runner._background_tasks == set()


@pytest.mark.asyncio
async def test_open_tells_the_chat_side_chats_are_a_desktop_feature():
    runner = _runner(None)
    thread = side_thread.register_thread("chat-sess", "t")
    thread._finish("done", answer="report")
    reply = await runner._handle_btw_command(_make_event(text="/btw --open"))
    assert "Desktop" in reply and thread.thread_id in reply
    assert runner.adapter._pending_messages == {}

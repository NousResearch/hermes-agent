"""A turn that ends in intentional silence stops "typing" when the agent returns (#149948).

The adapter's keep-alive refreshes typing until the message handler returns. A silent turn sends
nothing that would end it, so the work after the agent finished (agent:end hooks, transcript
persistence) kept the chat showing "typing" for a reply that never comes.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from tests.gateway.test_gateway_silence_tokens import _event, _runner, _source
from tests.waiting import async_wait_until

KEY = "agent:main:telegram:group:-1001:12345"


class _CountingPaused(set):
    """``_typing_paused`` that counts the keep-alive's per-tick membership checks while ``armed``,
    so the test waits on keep-alive iterations, not on wall-clock time."""

    armed = False
    checks = 0

    def __contains__(self, item):
        if self.armed:
            self.checks += 1
        return super().__contains__(item)


class _Adapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {}


def _result(text):
    return {"final_response": text, "failed": False, "api_calls": 1, "tools": [], "history_offset": 0,
            "last_prompt_tokens": 0, "messages": [{"role": "user", "content": "side chatter"},
                                                  {"role": "assistant", "content": text}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("answer,typing_after_agent", [("[SILENT]", False), ("A real reply.", True)],
                         ids=["silent-turn", "control-reply"])
async def test_typing_stops_once_a_silent_turn_returns(monkeypatch, tmp_path, answer, typing_after_agent):
    runner = _runner(monkeypatch, tmp_path)
    adapter = _Adapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    adapter.gateway_runner = runner
    runner._delivery_adapter_for = lambda _source: adapter
    adapter._typing_paused = paused = _CountingPaused(adapter._typing_paused)
    keep_typing = adapter._keep_typing
    adapter._keep_typing = lambda chat_id, **kw: keep_typing(chat_id, interval=0.05, **kw)  # fast ticks
    ticks, in_hook, release_hook, release_agent = [], asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def send_typing(chat_id, metadata=None):
        ticks.append("hook" if in_hook.is_set() else "agent")

    adapter.send_typing = send_typing

    async def run_agent(**_kwargs):
        await release_agent.wait()  # the agent works until the keep-alive has ticked
        return _result(answer)

    async def agent_end_hook(event_type, _ctx):
        if event_type == "agent:end":  # post-agent work: nothing more will be shown
            in_hook.set()
            await release_hook.wait()

    runner._run_agent = AsyncMock(side_effect=run_agent)
    runner.hooks.emit = AsyncMock(side_effect=agent_end_hook)
    adapter._message_handler = lambda ev: runner._handle_message_with_agent(ev, _source(), KEY, 1)
    adapter._active_sessions[KEY] = asyncio.Event()
    turn = asyncio.create_task(adapter._process_message_background(_event(reply_expected=False), KEY))

    await async_wait_until(lambda: ticks.count("agent") >= 2, what="typing while the agent works", timeout=10)
    release_agent.set()
    await async_wait_until(in_hook.is_set, what="the agent:end hook to start", timeout=10)
    paused.armed = True  # the keep-alive gets three full iterations while the hook runs
    await async_wait_until(lambda: paused.checks >= 6, what="keep-alive iterations during the hook", timeout=10)
    release_hook.set()
    await asyncio.wait_for(turn, 10)

    assert (ticks.count("hook") > 0) is typing_after_agent
    assert "-1001" not in adapter._typing_paused  # the turn's finally lifted the pause


@pytest.mark.asyncio
async def test_a_silent_topic_leaves_another_topics_typing_on(monkeypatch, tmp_path):
    """The pause is the silent turn's own: a second forum topic of the same chat whose agent is
    still working keeps refreshing "typing" while the silent turn runs its hooks."""
    runner = _runner(monkeypatch, tmp_path)
    adapter = _Adapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    adapter.gateway_runner = runner
    runner._delivery_adapter_for = lambda _source: adapter
    adapter._typing_paused = paused = _CountingPaused(adapter._typing_paused)
    keep_typing = adapter._keep_typing
    adapter._keep_typing = lambda chat_id, **kw: keep_typing(chat_id, interval=0.05, **kw)
    ticks, in_hook, release_hook, release_agent = [], asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def send_typing(chat_id, metadata=None):
        if in_hook.is_set():
            ticks.append((metadata or {}).get("thread_id"))

    adapter.send_typing = send_typing

    async def run_agent(**_kwargs):
        await release_agent.wait()
        return _result("[SILENT]")

    async def agent_end_hook(event_type, _ctx):
        if event_type == "agent:end":
            in_hook.set()
            await release_hook.wait()

    runner._run_agent = AsyncMock(side_effect=run_agent)
    runner.hooks.emit = AsyncMock(side_effect=agent_end_hook)
    working_done = asyncio.Event()

    async def handler(ev):
        if ev.source.thread_id == "working":
            await working_done.wait()
            return None
        return await runner._handle_message_with_agent(ev, ev.source, KEY, 1)

    adapter._message_handler = handler
    silent, working = _event(reply_expected=False), _event(reply_expected=True)
    silent.source.thread_id, working.source.thread_id = "silent", "working"
    adapter._active_sessions[KEY] = asyncio.Event()
    adapter._active_sessions[KEY + ":working"] = asyncio.Event()
    other = asyncio.create_task(adapter._process_message_background(working, KEY + ":working"))
    turn = asyncio.create_task(adapter._process_message_background(silent, KEY))
    try:
        release_agent.set()
        await async_wait_until(in_hook.is_set, what="the silent turn's agent:end hook", timeout=10)
        paused.armed = True
        await async_wait_until(lambda: paused.checks >= 8, what="both loops checking the pause", timeout=10)
        assert "working" in ticks and "silent" not in ticks
    finally:
        release_hook.set()
        await asyncio.wait_for(turn, 10)
        working_done.set()
        await asyncio.wait_for(other, 10)
    assert not adapter._typing_paused  # both turns' finally lifted their pauses

"""The rich-card mixin changes nothing unless ``display.tool_progress_native`` is on.

Flag-off parity: a turn with the stock native card (``native_task_cards: true``) and no rich flag
queues exactly the stock events, wires no reasoning sink and drains through the stock rail. Flag-on:
the extra events, the shielded drain and the dead-turn fences. The stock card state, publish seam and
text fallback are reused, not copied.
"""

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from gateway.run_turn_progress import GatewayTurnProgressMixin
from gateway.run_turn_task_cards import TaskCardMixin
from tests.gateway.test_rich_slack_task_cards import NativeAdapter, direct_adapter, make_turn

FLAG = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
STOCK = {"display": {"platforms": {"slack": {"tool_progress": "all"}}}}
SUBAGENT = {"type": "subagent.start", "subagent_id": "s1", "goal": "look around"}


@pytest.mark.asyncio
async def test_flag_off_queues_stock_events_and_wires_no_reasoning_sink(monkeypatch):
    ctx, turn = await make_turn(monkeypatch, STOCK, NativeAdapter())
    assert ctx._native_slack_task_cards is True and ctx._rich_slack_task_cards is False
    turn.native_tool_start_callback("call-a", "terminal", {"command": "date"})
    turn.native_tool_complete_callback("call-a", "terminal", {"command": "date"}, "done")
    turn.progress_callback("subagent.start", None, "look around", None, subagent_id="s1")
    queued = [ctx.progress_queue.get_nowait() for _ in range(ctx.progress_queue.qsize())]
    assert queued == [
        {"type": "tool.started", "tool_call_id": "call-a", "tool_name": "terminal", "preview": "date"},
        {"type": "tool.completed", "tool_call_id": "call-a", "tool_name": "terminal", "is_error": False},
    ]
    assert turn.test_agent.reasoning_callback is None


@pytest.mark.asyncio
async def test_flag_off_drains_through_the_stock_rail(monkeypatch):
    calls = []

    async def stock(self, adapter):
        calls.append(adapter)

    monkeypatch.setattr(GatewayTurnProgressMixin, "_send_native_task_card_progress", stock)
    adapter = NativeAdapter()
    ctx, turn = await make_turn(monkeypatch, STOCK, adapter)
    await TaskCardMixin._send_native_task_card_progress(turn, adapter)
    assert calls == [adapter]
    assert not hasattr(ctx, "rich_session")


@pytest.mark.asyncio
async def test_flag_without_native_cards_does_not_enable_rich(monkeypatch):
    ctx, _ = await make_turn(monkeypatch, FLAG, NativeAdapter(basic=False))
    assert ctx._native_slack_task_cards is False and ctx._rich_slack_task_cards is False


@pytest.mark.asyncio
async def test_rich_needs_no_tool_progress_setting_but_respects_an_explicit_off(monkeypatch):
    only_flag = {"display": {"platforms": {"slack": {"tool_progress_native": True}}}}
    ctx, _ = await make_turn(monkeypatch, only_flag, NativeAdapter())
    assert ctx._native_slack_task_cards is True and ctx._rich_slack_task_cards is True
    explicit_off = {"display": {"platforms": {"slack": {"tool_progress": "off", "tool_progress_native": True}}}}
    ctx, _ = await make_turn(monkeypatch, explicit_off, NativeAdapter())
    assert ctx._native_slack_task_cards is False and ctx._rich_slack_task_cards is False


@pytest.mark.asyncio
async def test_flag_on_queues_rich_events_with_args_and_result(monkeypatch):
    ctx, turn = await make_turn(monkeypatch, FLAG, NativeAdapter())
    assert ctx._rich_slack_task_cards is True
    turn.native_tool_start_callback("call-a", "terminal", {"command": "date"})
    turn.native_tool_complete_callback("call-a", "terminal", {"command": "date"}, "done")
    turn.progress_callback("subagent.start", None, "look around", None, subagent_id="s1")
    started, completed, child = (ctx.progress_queue.get_nowait() for _ in range(3))
    assert started["args"] == {"command": "date"}
    assert completed["args"] == {"command": "date"} and completed["result"] == "done"
    assert child["type"] == "subagent.start" and child["subagent_id"] == "s1"


@pytest.mark.asyncio
async def test_closed_cards_stop_accepting_events(monkeypatch):
    ctx, turn = await make_turn(monkeypatch, FLAG, NativeAdapter())
    ctx._task_cards_closed = True
    turn.native_tool_start_callback("call-a", "terminal", {})
    turn.progress_callback("subagent.start", None, "late", None, subagent_id="s1")
    turn.test_agent.reasoning_callback("late thought")
    assert ctx.progress_queue.empty()


@pytest.mark.asyncio
async def test_muted_notification_turn_opens_no_rich_card(monkeypatch):
    ctx, turn = await make_turn(monkeypatch, FLAG, NativeAdapter())
    ctx.mute_notification_reply = True
    turn.native_tool_start_callback("call-a", "terminal", {})
    assert ctx.progress_queue.empty()


@pytest.mark.asyncio
async def test_reasoning_and_subagent_events_are_dropped_after_native_failure(monkeypatch):
    _ctx, turn = await make_turn(monkeypatch, FLAG, NativeAdapter())
    state = turn._RichTaskCardState(adapter=NativeAdapter(), rich=object())
    state.native_failed = True
    for _ in range(1000):
        assert not state.apply_event({"type": "reasoning.delta", "text": "fragment"})
        assert not state.apply_event(SUBAGENT)
    assert state.pending == [] and state.tasks == {}
    # Tool rows keep feeding the stock text fallback.
    assert state.apply_event({"type": "tool.started", "tool_call_id": "a", "tool_name": "terminal"})
    assert state.pending == [] and list(state.tasks) == ["a"]


@pytest.mark.asyncio
async def test_rich_state_redacts_tool_events_before_the_text_fallback(monkeypatch):
    _ctx, turn = await make_turn(monkeypatch, FLAG, NativeAdapter())
    state = turn._RichTaskCardState(adapter=NativeAdapter(), rich=object())
    secret = "sk-" + "a1B2c3D4" * 5
    state.apply_event({"type": "tool.started", "tool_call_id": "a", "tool_name": "terminal",
                       "preview": f"curl -H 'Authorization: Bearer {secret}'"})
    assert secret not in str(state.pending) and secret not in str(state.tasks)


@pytest.mark.asyncio
async def test_dead_turn_publishes_nothing_and_is_not_a_transport_failure(monkeypatch):
    adapter, client = direct_adapter()
    ctx, turn = await make_turn(monkeypatch, FLAG, adapter)
    from gateway.slack_task_stream import RichTaskCardSession
    state = turn._RichTaskCardState(adapter=adapter, rich=RichTaskCardSession(adapter, ctx))
    state.apply_event({"type": "tool.started", "tool_call_id": "a", "tool_name": "read_file"})
    ctx._run_still_current = lambda: False
    await turn._task_card_publish(state)
    assert client.calls == [] and state.native_failed is False and state.pending == []


@pytest.mark.asyncio
async def test_cleanup_timeout_never_falls_back_after_unknown_native_acceptance(monkeypatch):
    from gateway import run_turn_task_cards as cards
    from gateway import slack_task_stream
    entered, never, stopped, posts = asyncio.Event(), asyncio.Event(), [], []

    class Rich:
        def __init__(self, adapter, ctx):
            self.main = object()
            self.client = SimpleNamespace(is_live=lambda: True)

        async def publish(self, pending):
            entered.set()
            await never.wait()

        async def flush_pending(self):
            return None

        async def stop(self, *, flush_reasoning):
            stopped.append(flush_reasoning)

    adapter = NativeAdapter()
    adapter.send = lambda **kw: posts.append(kw)
    ctx, turn = await make_turn(monkeypatch, FLAG, adapter)
    monkeypatch.setattr(slack_task_stream, "RichTaskCardSession", Rich)

    async def timed_out(awaitable, timeout):
        awaitable.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await awaitable
        raise TimeoutError("synthetic unknown acceptance")

    monkeypatch.setattr(cards.asyncio, "wait_for", timed_out)
    ctx.progress_queue.put({"type": "tool.started", "tool_name": "read_file", "tool_call_id": "one"})
    consumer = asyncio.create_task(turn._send_native_task_card_progress(adapter))
    await entered.wait()
    consumer.cancel()
    await consumer
    assert posts == [] and stopped == [False]


@pytest.mark.asyncio
async def test_rich_session_that_cannot_be_built_uses_the_stock_rail(monkeypatch):
    calls = []

    async def stock(self, adapter):
        calls.append(adapter)

    monkeypatch.setattr(GatewayTurnProgressMixin, "_send_native_task_card_progress", stock)
    adapter = NativeAdapter()  # no direct transport seams: the session has no main stream
    _ctx, turn = await make_turn(monkeypatch, FLAG, adapter)
    await turn._send_native_task_card_progress(adapter)
    assert calls == [adapter]

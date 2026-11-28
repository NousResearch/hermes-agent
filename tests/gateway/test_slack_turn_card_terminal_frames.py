"""Slack task cards end in a terminal frame and every status change reaches the card.

#138193: a row outside the newest-8 window that changes status is still published.
#141643: the closed card gets a terminal title and no running row, on a normal end, on /stop, and
when the card is rotated before Slack's stream lifetime seals it. Run rollup and the reasoning row
are opt-in behaviour on top (``task_card_rollup``, ``thinking_cards`` + ``show_reasoning``).
"""
from __future__ import annotations

import asyncio
import queue
from types import SimpleNamespace as NS

import pytest

from gateway.config import PlatformConfig
from gateway.progress_rollup import summarize_run
from gateway.run_turn_progress import GatewayTurnProgressMixin
from gateway.run_turn_runner import TurnRunner
from plugins.platforms.slack.adapter import SlackAdapter
from tests.waiting import async_wait_until


class _Client:
    def __init__(self):
        self.calls = []

    async def api_call(self, method, json=None):
        self.calls.append((method, json or {}))
        return {"ok": True, "ts": f"s{sum(1 for m, _ in self.calls if m == 'chat.startStream')}"}

    def frames(self):
        return [(m, [(c.get("title"), c.get("status")) for c in j.get("chunks", [])]) for m, j in self.calls]


def _adapter(extra=None):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-x", extra=extra or {}))
    client = _Client()
    adapter._app, adapter._get_client = object(), lambda *a, **k: client
    return adapter, client


def _ctx(events, current, agent):
    return NS(progress_queue=events, _run_still_current=lambda: current["v"], source=NS(chat_id="C1"),
              _progress_reply_to="1.1", _progress_metadata={"thread_id": "1.1"}, tool_progress_enabled=False,
              agent_holder=[agent], _cleanup_progress=False)


async def _drive(adapter, events, *, stop=False, late=()):
    current, agent = {"v": True}, NS(is_interrupted=False)
    runner = GatewayTurnProgressMixin.__new__(GatewayTurnProgressMixin)
    runner._ctx = _ctx(events, current, agent)
    loop = asyncio.create_task(runner._send_native_task_card_progress(adapter))
    expected = events.qsize()  # every queued event changes the card: one append each
    await async_wait_until(lambda: _appends(adapter) == expected, what="every event published")
    if not stop:
        # A turn that ends normally: cleanup cancels the drain while the run is still current.
        loop.cancel()
        await asyncio.gather(loop, return_exceptions=True)
        return
    agent.is_interrupted, current["v"] = True, False  # what /stop does
    for event in late:
        events.put(event)
    await asyncio.wait_for(loop, timeout=3)


def _appends(adapter):
    return sum(1 for method, _ in adapter._get_client().calls if method == "chat.appendStream")


def _events(*items):
    q = queue.Queue()
    for item in items:
        q.put(item)
    return q


def _started(n, name="web_search"):
    return {"type": "tool.started", "tool_call_id": f"call-{n}", "tool_name": name}


def _done(n, name="web_search", error=False):
    return {"type": "tool.completed", "tool_call_id": f"call-{n}", "tool_name": name, "is_error": error}


def test_status_change_outside_newest_eight_is_published():
    card = TurnRunner._TaskCardState(adapter=None)
    for n in range(9):
        card.apply_event(_started(n))
    card.mark_published(card.visible_tasks())
    card.apply_event(_done(0))
    frame = {task["id"]: task["status"] for task in card.visible_tasks()}
    assert frame["call-0"] == "complete"
    card.mark_published(card.visible_tasks())
    assert "call-0" not in {task["id"] for task in card.visible_tasks()}


@pytest.mark.asyncio
async def test_closed_card_gets_terminal_title_and_no_running_row():
    adapter, client = _adapter()
    await _drive(adapter, _events(_started(1, "read_file"), _done(1, "read_file"), _started(2, "terminal"),
                                  _done(2, "terminal", error=True)))
    method, chunks = client.frames()[-1]
    assert method == "chat.stopStream"
    assert chunks[0] == ("Ran 2 steps · 1 failed", None)
    assert all(status in {"complete", "error"} for _, status in chunks[1:])


@pytest.mark.asyncio
async def test_stop_mid_tool_closes_with_stopped_title_and_no_running_row():
    adapter, client = _adapter()
    await _drive(adapter, _events(_started(1, "delegate_task")), stop=True,
                 late=[_done(1, "delegate_task")])
    method, chunks = client.frames()[-1]
    assert method == "chat.stopStream"
    assert chunks[0][0].startswith("Stopped")
    assert "in_progress" not in {status for _, status in chunks[1:]}


@pytest.mark.asyncio
async def test_card_rotates_before_slack_lifetime_with_closing_frame(monkeypatch):
    adapter, client = _adapter({"native_task_card_rotate_seconds": 60})
    md = {"thread_id": "1.1"}
    await adapter.send_native_task_card_progress("C1", [{"id": "a", "title": "terminal", "status": "in_progress"}],
                                                 metadata=md)
    (stream,) = adapter._native_task_card_streams.values()
    stream.started_at -= 61
    await adapter.send_native_task_card_progress("C1", [{"id": "a", "title": "terminal", "status": "complete"}],
                                                 metadata=md)
    methods = [m for m, _ in client.frames()]
    assert methods == ["chat.startStream", "chat.appendStream", "chat.stopStream", "chat.startStream",
                       "chat.appendStream"]
    closing = client.frames()[2][1]
    assert closing[0][0] == "Continued below ⤵"
    assert "in_progress" not in {status for _, status in closing[1:]}


def test_rollup_reads_ran_n_steps_and_a_recovered_failure_keeps_no_error_badge():
    steps = [{"title": "read_file - a", "name": "read_file", "status": "complete"},
             {"title": "terminal - x", "name": "terminal", "status": "error"},
             {"title": "terminal - y", "name": "terminal", "status": "complete"}]
    live = summarize_run([*steps, {"title": "terminal - z", "name": "terminal", "status": "in_progress"}],
                         closed=False)
    assert live.title == "terminal - z"
    assert live.earlier_trail == "✓ read_file · ✗ terminal · ✓ terminal"
    closed = summarize_run(steps, closed=True)
    assert (closed.tool_count, closed.failed, closed.status) == (3, 1, "complete")


@pytest.mark.asyncio
async def test_rollup_card_folds_runs_between_text():
    adapter, client = _adapter({"task_card_rollup": True})
    await _drive(adapter, _events(_started(1, "read_file"), _done(1, "read_file"), _started(2, "terminal"),
                                  _done(2, "terminal"), {"type": "text.boundary"}, _started(3, "terminal")))
    rows = [chunks for method, chunks in client.frames() if method == "chat.appendStream"]
    assert ("terminal - ", "in_progress") not in rows[-1]
    assert rows[-1][-1][0].startswith("terminal")
    stop = client.frames()[-1][1]
    assert stop[0] == ("Ran 3 steps", None)
    assert stop[1:] == [("Ran 2 steps", "complete"), ("terminal", "complete")]


@pytest.mark.asyncio
async def test_reasoning_block_shows_thinking_then_thought_for_seconds(monkeypatch):
    import gateway.run_turn_runner_thinking as thinking

    monkeypatch.setattr(thinking, "REASONING_CARD_AFTER_S", 0.0)
    adapter, client = _adapter({"thinking_cards": True})
    q = queue.Queue()
    ctx = NS(source=NS(chat_id="C1"), thinking_queue=q, _native_slack_thinking_cards=True, _progress_reply_to="1.1",
             _progress_metadata={"thread_id": "1.1"}, _cleanup_progress=False, agent_holder=[None])
    alive = {"v": True}
    ctx._run_still_current = lambda: alive["v"]
    runner = TurnRunner(NS(_delivery_adapter_for=lambda source: adapter), ctx)
    task = asyncio.create_task(runner.send_thinking_card_messages())
    q.put({"type": "reasoning", "text": "Compare both config paths before editing."})
    await async_wait_until(lambda: _titles(client, "Thinking…"), what="Thinking row")
    q.put({"type": "reasoning.end"})
    await async_wait_until(lambda: _titles(client, "Thought for "), what="Thought-for row")
    alive["v"] = False
    await asyncio.wait_for(task, timeout=3)
    assert _titles(client, "Thinking…") and _titles(client, "Thought for ")
    assert client.frames()[-1][0] == "chat.stopStream"


def _titles(client, prefix):
    return [title for method, chunks in client.frames() if method == "chat.appendStream"
            for title, _ in chunks if str(title).startswith(prefix)]

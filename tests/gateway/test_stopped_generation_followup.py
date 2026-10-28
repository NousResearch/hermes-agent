"""Regression for #124646: a turn invalidated by /stop must not start queued work.

After the worker returns, ``_run_agent_inner`` drains the adapter's pending slot (or a leftover
``/steer``) and runs it as a recursive follow-up that inherits the parent's run generation. /stop
can land while the interrupted worker finishes, while streaming TTS finalizes, or while the
follow-up is being prepared. Without a generation check the stale turn drains a queue that now
belongs to its successor and starts a worker that ``_run_agent_track_agent`` refuses to track
("Skipping stale agent promotion"), so no later /stop can reach it while it holds the session's
durable turn lease.

A follow-up discarded during preparation never ran, so it is reported CANCELLED. An internal wake
is re-parked for the successor; a human follow-up is dropped, as /stop drops one from the slot.

Real: ``GatewayRunner._run_agent`` drain and follow-up recursion, ``BasePlatformAdapter`` pending
slot, run-generation bookkeeping.
Fake: the agent (records which message started it), and /stop, applied as the generation
invalidation ``_interrupt_and_clear_session`` performs, at each await where a real /stop can land.
"""

import pytest

from gateway.platforms.base import MessageEvent, ProcessingOutcome
from tests.gateway.test_queued_followup_processing_hooks import (
    SESSION_KEY,
    HookRecordingAdapter,
    _install_fake_agent,
    _make_runner,
    _source,
    _TwoTurnAgent,
)


def _setup(monkeypatch, tmp_path):
    _TwoTurnAgent.calls = []
    _install_fake_agent(monkeypatch, tmp_path, _TwoTurnAgent)
    adapter = HookRecordingAdapter()
    return _make_runner(adapter), adapter


async def _turn(runner, message, generation):
    return await runner._run_agent(
        message=message, context_prompt="", history=[], source=_source(),
        session_id="sess-stop", session_key=SESSION_KEY, run_generation=generation,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("step", [
    pytest.param("_run_agent_await_turn_worker", id="worker-completion"),
    pytest.param("_run_agent_finalize_streaming_tts", id="streaming-tts-finalization"),
])
async def test_stopped_turn_leaves_the_queue_to_its_successor(monkeypatch, tmp_path, step):
    runner, adapter = _setup(monkeypatch, tmp_path)
    successor = MessageEvent(text="successor's message", source=_source(), message_id="successor")
    original = getattr(runner, step)
    landed = []

    async def step_then_stop(*args, **kwargs):
        result = await original(*args, **kwargs)
        if not landed:
            landed.append(runner._invalidate_session_run_generation(SESSION_KEY, reason="stop_command"))
            adapter._pending_messages[SESSION_KEY] = successor
        return result

    monkeypatch.setattr(runner, step, step_then_stop)
    await _turn(runner, "first", runner._begin_session_run_generation(SESSION_KEY))

    assert _TwoTurnAgent.calls == ["first"]
    assert adapter._pending_messages[SESSION_KEY] is successor

    # The successor's own turn still runs the message, exactly once.
    await _turn(runner, "second", landed[0])
    assert _TwoTurnAgent.calls == ["first", "second", "successor's message"]


@pytest.mark.asyncio
async def test_stop_during_followup_preparation_does_not_start_it(monkeypatch, tmp_path):
    runner, adapter = _setup(monkeypatch, tmp_path)
    adapter._pending_messages[SESSION_KEY] = MessageEvent(
        text="queued follow-up", source=_source(), message_id="queued",
    )

    async def stop_on_processing_start(event):
        runner._invalidate_session_run_generation(SESSION_KEY, reason="stop_command")

    monkeypatch.setattr(adapter, "on_processing_start", stop_on_processing_start)
    await _turn(runner, "first", runner._begin_session_run_generation(SESSION_KEY))

    assert _TwoTurnAgent.calls == ["first"]
    assert adapter.completed == [("queued", ProcessingOutcome.CANCELLED)]
    assert SESSION_KEY not in adapter._pending_messages


@pytest.mark.asyncio
@pytest.mark.parametrize("successor_queued", [False, True], ids=["empty-slot", "successor-in-slot"])
async def test_stop_during_followup_preparation_reparks_an_internal_wake(monkeypatch, tmp_path, successor_queued):
    runner, adapter = _setup(monkeypatch, tmp_path)
    wake = MessageEvent(text="delegation finished", source=_source(), internal=True)
    successor = MessageEvent(text="successor's message", source=_source(), message_id="successor")
    adapter._pending_messages[SESSION_KEY] = wake
    landed = []

    # /stop lands on the last await before the follow-up starts: an internal wake has no processing
    # hooks to land on.
    async def stop_before_followup_starts(*_args):
        if not landed:
            landed.append(runner._invalidate_session_run_generation(SESSION_KEY, reason="stop_command"))
            if successor_queued:
                adapter._pending_messages[SESSION_KEY] = successor

    monkeypatch.setattr(runner, "_refresh_agent_cache_message_count", stop_before_followup_starts)
    await _turn(runner, "first", runner._begin_session_run_generation(SESSION_KEY))

    assert _TwoTurnAgent.calls == ["first"]
    assert adapter._pending_messages[SESSION_KEY] is (successor if successor_queued else wake)

    await _turn(runner, "second", landed[0])
    assert _TwoTurnAgent.calls == ["first", "second"] + (
        ["successor's message", "delegation finished"] if successor_queued else ["delegation finished"])


@pytest.mark.asyncio
async def test_only_the_discarded_followup_reports_cancelled(monkeypatch, tmp_path):
    runner, adapter = _setup(monkeypatch, tmp_path)
    adapter._pending_messages[SESSION_KEY] = MessageEvent(text="ran", source=_source(), message_id="ran")

    async def on_processing_start(event):
        if event.message_id == "ran":
            adapter._pending_messages[SESSION_KEY] = MessageEvent(
                text="discarded", source=_source(), message_id="discarded",
            )
        else:
            runner._invalidate_session_run_generation(SESSION_KEY, reason="stop_command")

    monkeypatch.setattr(adapter, "on_processing_start", on_processing_start)
    await _turn(runner, "first", runner._begin_session_run_generation(SESSION_KEY))

    assert _TwoTurnAgent.calls == ["first", "ran"]
    assert adapter.completed == [
        ("discarded", ProcessingOutcome.CANCELLED), ("ran", ProcessingOutcome.SUCCESS),
    ]

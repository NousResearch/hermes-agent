"""HermesAgentExecutor maps AIAgent turn outcomes onto A2A task states."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time

import pytest
from a2a.types import (
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from a2a.utils.errors import ServerError

from plugins.platforms.a2a.executor import HermesAgentExecutor
from plugins.platforms.a2a.sessions import ContextSessionStore


class _OutcomeAgent:
    """Agent stub that returns a caller-specified ``run_conversation`` result."""

    def __init__(self, result: dict):
        self._result = result
        self.stream_delta_callback = None
        self.reasoning_callback = None
        self.tool_progress_callback = None
        self.step_callback = None
        self.thinking_callback = None
        self.runs: list[str] = []

    def run_conversation(
        self, *, user_message, conversation_history=None, task_id=None, **kw
    ):
        self.runs.append(user_message)
        return self._result

    def interrupt(self, message=None):
        pass


def _run(fakes, agent, user_text="do the thing", context_id="ctx-out"):
    store = ContextSessionStore(agent_factory=lambda: agent)
    executor = HermesAgentExecutor(store)
    message = fakes.make_user_message(user_text, context_id=context_id)
    context = fakes.FakeContext(
        user_text, message, current_task=None, context_id=context_id
    )
    queue = fakes.RecordingQueue()
    asyncio.run(executor.execute(context, queue))
    return queue.events


def _terminal_state(events):
    statuses = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
    return statuses[-1].status.state if statuses else None


# --- happy path ------------------------------------------------------------


def test_execute_event_sequence(fakes):
    agent = fakes.FakeAgent()
    events = _run(fakes, agent, user_text="hello")

    assert isinstance(events[0], Task)  # Task enqueued first
    statuses = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
    artifacts = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]

    assert any(s.status.state == TaskState.working for s in statuses)
    assert statuses[-1].status.state == TaskState.completed
    assert len(artifacts) == 1
    assert artifacts[0].artifact.parts[0].root.text == "echo: hello"
    assert agent.runs == ["hello"]


def test_callback_bridge_surfaces_tool_activity(fakes):
    events = _run(fakes, fakes.FakeAgent(), user_text="hello")
    statuses = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
    kinds = {s.metadata.get("hermes/kind") for s in statuses if s.metadata}
    assert "tool-call" in kinds
    assert "tool-result" in kinds


# --- outcome mapping (regression for the "failures look like success" bug) --


def test_failed_result_marks_task_failed(fakes):
    events = _run(
        fakes,
        _OutcomeAgent({
            "final_response": None,
            "failed": True,
            "error": "provider 500",
        }),
    )
    assert _terminal_state(events) == TaskState.failed
    assert not any(isinstance(e, TaskArtifactUpdateEvent) for e in events)


def test_failed_result_carries_explanatory_text(fakes):
    events = _run(
        fakes,
        _OutcomeAgent({"final_response": "blocked by content policy", "failed": True}),
    )
    assert _terminal_state(events) == TaskState.failed
    statuses = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
    msg = statuses[-1].status.message
    assert msg is not None and msg.parts[0].root.text == "blocked by content policy"
    assert not any(isinstance(e, TaskArtifactUpdateEvent) for e in events)


def test_interrupted_result_marks_canceled(fakes):
    events = _run(
        fakes, _OutcomeAgent({"final_response": "partial", "interrupted": True})
    )
    assert _terminal_state(events) == TaskState.canceled
    assert not any(isinstance(e, TaskArtifactUpdateEvent) for e in events)


def test_empty_final_response_marks_failed(fakes):
    events = _run(fakes, _OutcomeAgent({"final_response": "", "messages": []}))
    assert _terminal_state(events) == TaskState.failed
    assert not any(isinstance(e, TaskArtifactUpdateEvent) for e in events)


def test_error_result_without_failed_flag_marks_failed(fakes):
    """Degraded turns carry ``error`` but no ``failed``/``interrupted`` flag.

    ``run_conversation`` has early-return paths (e.g. thinking-budget exhausted,
    truncation) that set ``error`` + ``partial`` + a human-readable
    ``final_response`` but never reach ``finalize_turn``, so the dict has no
    ``failed`` key. These must be reported to the peer as a failure, not as a
    successful completion whose artifact is actually an error notice.
    """
    events = _run(
        fakes,
        _OutcomeAgent({
            "final_response": "⚠️ Thinking Budget Exhausted",
            "error": "thinking budget exhausted before any response",
            "partial": True,
            "completed": False,
        }),
    )
    assert _terminal_state(events) == TaskState.failed
    assert not any(isinstance(e, TaskArtifactUpdateEvent) for e in events)


def test_failed_no_response_preserves_error_detail(fakes):
    """When there is no usable text, the specific error must survive.

    Previously an empty ``final_response`` collapsed every distinct failure into
    a generic "Agent produced no response." — discarding the diagnostic the peer
    needs.
    """
    events = _run(
        fakes,
        _OutcomeAgent({
            "final_response": None,
            "error": "Response truncated due to output length limit",
        }),
    )
    assert _terminal_state(events) == TaskState.failed
    statuses = [e for e in events if isinstance(e, TaskStatusUpdateEvent)]
    text = statuses[-1].status.message.parts[0].root.text
    assert "Response truncated due to output length limit" in text


# --- concurrency bound -----------------------------------------------------


class _ConcurrencyProbe:
    """Records the peak number of turns running its loop simultaneously."""

    _state_lock = threading.Lock()
    live = 0
    peak = 0

    def __init__(self) -> None:
        self.stream_delta_callback = None
        self.reasoning_callback = None
        self.tool_progress_callback = None
        self.step_callback = None
        self.thinking_callback = None

    def run_conversation(
        self, *, user_message, conversation_history=None, task_id=None, **kw
    ):
        with _ConcurrencyProbe._state_lock:
            _ConcurrencyProbe.live += 1
            _ConcurrencyProbe.peak = max(_ConcurrencyProbe.peak, _ConcurrencyProbe.live)
        try:
            time.sleep(0.05)
        finally:
            with _ConcurrencyProbe._state_lock:
                _ConcurrencyProbe.live -= 1
        return {"final_response": "ok", "messages": []}

    def interrupt(self, message=None):
        pass

    def clear_interrupt(self):
        pass


def test_concurrent_turns_are_bounded_and_overload_is_rejected(fakes):
    """The limit bounds admitted work rather than merely worker threads."""
    _ConcurrencyProbe.live = 0
    _ConcurrencyProbe.peak = 0
    store = ContextSessionStore(agent_factory=_ConcurrencyProbe)
    executor = HermesAgentExecutor(store, max_concurrency=1)

    async def drive():
        async def one(ctx):
            message = fakes.make_user_message("go", context_id=ctx)
            context = fakes.FakeContext(
                "go", message, current_task=None, context_id=ctx
            )
            await executor.execute(context, fakes.RecordingQueue())

        first = asyncio.create_task(one("ctx-a"))
        for _ in range(100):
            if _ConcurrencyProbe.live:
                break
            await asyncio.sleep(0.001)
        assert _ConcurrencyProbe.live == 1

        with pytest.raises(ServerError, match="at turn capacity"):
            await one("ctx-b")

        await first
        await executor.aclose()

    asyncio.run(drive())
    assert _ConcurrencyProbe.peak == 1


def test_canceled_request_keeps_capacity_reserved_until_worker_exits(fakes):
    started = threading.Event()
    release = threading.Event()

    class StubbornAgent(_OutcomeAgent):
        def __init__(self):
            super().__init__({"final_response": "done", "messages": []})

        def run_conversation(self, **_kwargs):
            started.set()
            release.wait(5)
            return self._result

    executor = HermesAgentExecutor(
        ContextSessionStore(agent_factory=StubbornAgent), max_concurrency=1
    )

    async def one(context_id):
        message = fakes.make_user_message("go", context_id=context_id)
        context = fakes.FakeContext(
            "go", message, current_task=None, context_id=context_id
        )
        await executor.execute(context, fakes.RecordingQueue())

    async def drive():
        first = asyncio.create_task(one("ctx-a"))
        assert await asyncio.to_thread(started.wait, 5)
        first.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first

        with pytest.raises(ServerError, match="at turn capacity"):
            await one("ctx-b")

        release.set()
        await executor.aclose()

    asyncio.run(drive())


def test_close_prevents_admitted_request_from_starting_after_await(fakes):
    """Shutdown cannot be bypassed by a pre-worker protocol await.

    A request reserves capacity before publishing its initial task. If that
    publish blocks while ``aclose()`` runs, resuming it must not recreate the
    closed session store or lazy worker pool.
    """
    enqueue_started = asyncio.Event()
    release_enqueue = asyncio.Event()
    agent = _OutcomeAgent({"final_response": "late", "messages": []})
    store = ContextSessionStore(agent_factory=lambda: agent)
    executor = HermesAgentExecutor(store)

    class BlockingQueue(fakes.RecordingQueue):
        async def enqueue_event(self, event):
            enqueue_started.set()
            await release_enqueue.wait()
            await super().enqueue_event(event)

    async def drive():
        message = fakes.make_user_message("late", context_id="ctx-late")
        context = fakes.FakeContext(
            "late", message, current_task=None, context_id="ctx-late"
        )
        queue = BlockingQueue()
        request = asyncio.create_task(executor.execute(context, queue))
        await enqueue_started.wait()

        await executor.aclose()
        release_enqueue.set()

        await request
        assert _terminal_state(queue.events) == TaskState.failed

    asyncio.run(drive())
    assert executor._turn_pool is None
    assert not store._sessions
    assert agent.runs == []


# --- invalid input ---------------------------------------------------------


def test_blank_or_empty_message_rejected(fakes):
    """Empty/whitespace input is a JSON-RPC error, not a 'completed' task.

    Both whitespace ("   ") and a truly-empty TextPart ("") must be rejected
    before ``new_task`` (which itself raises on an empty TextPart).
    """
    store = ContextSessionStore(agent_factory=fakes.FakeAgent)
    executor = HermesAgentExecutor(store)
    for text in ("   ", ""):
        message = fakes.make_user_message(text, context_id="ctx-empty")
        context = fakes.FakeContext(
            text, message, current_task=None, context_id="ctx-empty"
        )
        queue = fakes.RecordingQueue()
        with pytest.raises(ServerError):
            asyncio.run(executor.execute(context, queue))
        assert queue.events == []  # nothing enqueued for invalid input

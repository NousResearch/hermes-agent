"""Behavior tests for the provider-neutral realtime voice orchestrator.

A scripted fake session stands in for every provider; a recording host
stands in for the CLI. No network, no audio devices, no OpenAI key.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from agent.realtime_voice_orchestrator import (
    INTERRUPTED_TEXT,
    TOOLS_UNAVAILABLE_TEXT,
    UNPARSEABLE_ARGS_TEXT,
    RealtimeVoiceError,
    RealtimeVoiceHost,
    RealtimeVoiceOrchestrator,
    run_blocking_tool,
)
from agent.realtime_voice_provider import (
    InputSpeechStarted,
    InputTranscript,
    OutputAudio,
    OutputTranscript,
    RealtimeCapability,
    RealtimeVoiceSession,
    ResponseCompleted,
    ResponseStarted,
    SessionClosed,
    SessionFailure,
    SessionReady,
    SessionResumptionUpdate,
    ToolCall,
    ToolCallCancelled,
)

FULL_CAPABILITIES = frozenset(
    {
        RealtimeCapability.TOOL_CALLING,
        RealtimeCapability.RESPONSE_CANCELLATION,
        RealtimeCapability.OUTPUT_TRUNCATION,
        RealtimeCapability.INPUT_TRANSCRIPTION,
        RealtimeCapability.OUTPUT_TRANSCRIPTION,
    }
)


class FakeSession(RealtimeVoiceSession):
    """Scripted provider: tests push events in, the fake records commands out."""

    def __init__(self, capabilities=FULL_CAPABILITIES) -> None:
        super().__init__(capabilities)
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent_audio: list[bytes] = []
        self.cancels: list[str | None] = []
        self.truncations: list[tuple[str, int]] = []
        self.submissions: list[tuple[tuple, bool]] = []
        self.close_count = 0
        self.submit_error: Exception | None = None
        self.cancel_error: Exception | None = None

    def push(self, *events) -> None:
        for event in events:
            self.inbox.put_nowait(event)

    async def send_audio(self, audio: bytes) -> None:
        self.sent_audio.append(audio)

    def _events(self):
        async def stream():
            while True:
                event = await self.inbox.get()
                yield event

        return stream()

    async def _submit_tool_results(self, results, continue_response) -> None:
        if self.submit_error is not None:
            raise self.submit_error
        self.submissions.append((results, continue_response))

    async def _cancel_response(self, response_id) -> None:
        if self.cancel_error is not None:
            raise self.cancel_error
        self.cancels.append(response_id)

    async def _truncate_output(self, item_id, audio_end_ms) -> None:
        self.truncations.append((item_id, audio_end_ms))

    async def _close(self) -> None:
        self.close_count += 1


class RecordingHost(RealtimeVoiceHost):
    def __init__(self, played_ms: int | None = 750) -> None:
        self.played_ms = played_ms
        self.audio: list[bytes] = []
        self.items: list[str] = []
        self.barge_ins = 0
        self.transcripts: list[tuple[str, str, bool]] = []
        self.errors: list[tuple[str, bool]] = []
        self.tool_calls: list[str] = []
        self.tool_results: list[tuple[str, str, str]] = []
        self.responses: list[tuple[str, str | None]] = []
        self.ready: list[str] = []
        self.closed: list[str] = []
        self.resumption: list[str | None] = []

    def on_session_ready(self, session_id):
        self.ready.append(session_id)

    def on_session_closed(self, reason):
        self.closed.append(reason)

    def on_error(self, message, *, terminal):
        self.errors.append((message, terminal))

    def on_resumption_update(self, handle):
        self.resumption.append(handle)

    def on_input_transcript(self, text, final):
        self.transcripts.append(("user", text, final))

    def on_output_transcript(self, text, final):
        self.transcripts.append(("assistant", text, final))

    def on_output_item_started(self, item_id):
        self.items.append(item_id)

    def on_output_audio(self, pcm):
        self.audio.append(pcm)

    def on_response_started(self, response_id):
        self.responses.append(("started", response_id))

    def on_response_completed(self, response_id):
        self.responses.append(("completed", response_id))

    def on_barge_in(self):
        self.barge_ins += 1
        return self.played_ms

    def on_tool_call(self, call):
        self.tool_calls.append(call.call_id)

    def on_tool_result(self, call, output, outcome):
        self.tool_results.append((call.call_id, output, outcome))


class RecordingExecutor:
    def __init__(
        self,
        *,
        block: bool = False,
        block_names: frozenset[str] = frozenset(),
        error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.block = block
        self.block_names = block_names
        self.error = error
        self.cancelled = 0
        self.started = asyncio.Event()

    async def __call__(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        self.started.set()
        if self.error is not None:
            raise self.error
        if self.block or name in self.block_names:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        return f"{name}:{arguments.get('q', '')}"


async def settle(rounds: int = 6) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


def make(
    session: FakeSession | None = None,
    host: RecordingHost | None = None,
    executor=None,
    **kwargs,
):
    session = session or FakeSession()
    host = host or RecordingHost()
    orchestrator = RealtimeVoiceOrchestrator(session, host, tool_executor=executor, **kwargs)
    return session, host, orchestrator


async def finish(session: FakeSession, run_task: asyncio.Task) -> None:
    session.push(SessionClosed(reason="done"))
    await asyncio.wait_for(run_task, timeout=2)


# -- construction ------------------------------------------------------------


def test_constructor_validates_inputs() -> None:
    with pytest.raises(TypeError, match="RealtimeVoiceSession"):
        RealtimeVoiceOrchestrator(object(), RecordingHost())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RealtimeVoiceHost"):
        RealtimeVoiceOrchestrator(FakeSession(), object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tool_timeout_s"):
        RealtimeVoiceOrchestrator(FakeSession(), RecordingHost(), tool_timeout_s=0)


@pytest.mark.asyncio
async def test_run_can_only_be_called_once() -> None:
    session, host, orchestrator = make()
    run_task = asyncio.create_task(orchestrator.run())
    await settle()
    with pytest.raises(RuntimeError, match="only be called once"):
        await orchestrator.run()
    await finish(session, run_task)


# -- lifecycle ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifecycle_events_reach_host_and_session_closes() -> None:
    session, host, orchestrator = make()
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        SessionReady(session_id="sess"),
        SessionResumptionUpdate(handle="h1", resumable=True),
        SessionResumptionUpdate(handle="h1", resumable=False),
        InputTranscript(text="hello", final=True),
        SessionFailure(code="frame", message="one bad frame", terminal=False),
    )
    await finish(session, run_task)

    assert host.ready == ["sess"]
    assert host.resumption == ["h1", None]
    assert host.transcripts == [("user", "hello", True)]
    assert host.errors == [("one bad frame", False)]
    assert host.closed == ["done"]
    assert session.close_count == 1
    assert session.closed is True


@pytest.mark.asyncio
async def test_terminal_failure_raises_after_closing_the_session() -> None:
    session, host, orchestrator = make()
    run_task = asyncio.create_task(orchestrator.run())
    session.push(SessionFailure(code="transport", message="socket died"))

    with pytest.raises(RealtimeVoiceError, match="transport: socket died"):
        await asyncio.wait_for(run_task, timeout=2)

    assert host.errors == [("socket died", True)]
    assert session.close_count == 1


@pytest.mark.asyncio
async def test_send_audio_forwards_to_session() -> None:
    session, host, orchestrator = make()
    await orchestrator.send_audio(b"\x00\x01")
    assert session.sent_audio == [b"\x00\x01"]


# -- response identity -------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_and_transcripts_flow_while_response_is_live() -> None:
    session, host, orchestrator = make()
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ResponseStarted(response_id="r1"),
        OutputAudio(data=b"a", item_id="i1", response_id="r1"),
        OutputAudio(data=b"b", item_id="i1", response_id="r1"),
        OutputTranscript(text="hi", final=False, response_id="r1"),
        OutputTranscript(text="hi there", final=True, response_id="r1"),
        ResponseCompleted(response_id="r1"),
    )
    await finish(session, run_task)

    assert host.audio == [b"a", b"b"]
    assert host.items == ["i1"]
    assert host.transcripts == [("assistant", "hi", False), ("assistant", "hi there", True)]
    assert host.responses == [("started", "r1"), ("completed", "r1")]
    assert orchestrator.response_active is False


@pytest.mark.asyncio
async def test_barge_in_cancels_once_truncates_and_fences_the_tail() -> None:
    session, host, orchestrator = make(host=RecordingHost(played_ms=750))
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ResponseStarted(response_id="r1"),
        OutputAudio(data=b"a", item_id="i1", response_id="r1"),
        InputSpeechStarted(item_id="in1", audio_start_ms=10),
        OutputAudio(data=b"tail", item_id="i1", response_id="r1"),
        OutputTranscript(text="tail text", final=True, response_id="r1"),
        InputSpeechStarted(item_id="in1", audio_start_ms=400),
        ResponseCompleted(response_id="r1", status="cancelled"),
        ResponseStarted(response_id="r2"),
        OutputAudio(data=b"next", item_id="i2", response_id="r2"),
        OutputAudio(data=b"late", item_id="i1", response_id="r1"),
        ResponseCompleted(response_id="r2"),
    )
    await finish(session, run_task)

    assert host.barge_ins == 2
    assert session.cancels == ["r1"]
    assert session.truncations == [("i1", 750)]
    assert host.audio == [b"a", b"next"]
    assert host.transcripts == []


@pytest.mark.asyncio
async def test_barge_in_during_trailing_playback_truncates_completed_item() -> None:
    session, host, orchestrator = make(host=RecordingHost(played_ms=1200))
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ResponseStarted(response_id="r1"),
        OutputAudio(data=b"a", item_id="i1", response_id="r1"),
        ResponseCompleted(response_id="r1"),
        InputSpeechStarted(),
    )
    await finish(session, run_task)

    assert session.cancels == []  # nothing in flight: no cancel on the wire
    assert session.truncations == [("i1", 1200)]


@pytest.mark.asyncio
async def test_barge_in_with_nothing_playing_sends_nothing() -> None:
    session, host, orchestrator = make(host=RecordingHost(played_ms=None))
    run_task = asyncio.create_task(orchestrator.run())
    session.push(InputSpeechStarted())
    await finish(session, run_task)

    assert host.barge_ins == 1
    assert session.cancels == []
    assert session.truncations == []


@pytest.mark.asyncio
async def test_provider_without_cancel_or_truncate_degrades_to_local_drop() -> None:
    session = FakeSession(capabilities={RealtimeCapability.TOOL_CALLING})
    session, host, orchestrator = make(session=session, host=RecordingHost(played_ms=300))
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ResponseStarted(response_id="r1"),
        OutputAudio(data=b"a", item_id="i1", response_id="r1"),
        InputSpeechStarted(),
        OutputAudio(data=b"tail", item_id="i1", response_id="r1"),
        ResponseCompleted(response_id="r1"),
    )
    await finish(session, run_task)

    assert host.barge_ins == 1
    assert session.cancels == []
    assert session.truncations == []
    assert host.audio == [b"a"]  # the local fence still drops the tail


@pytest.mark.asyncio
async def test_unnamed_responses_are_fenced_after_cancel() -> None:
    session, host, orchestrator = make(host=RecordingHost(played_ms=100))
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ResponseStarted(),
        OutputAudio(data=b"a", item_id="i1"),
        InputSpeechStarted(),
        OutputAudio(data=b"tail", item_id="i1"),
        ResponseCompleted(),
        ResponseStarted(),
        OutputAudio(data=b"next", item_id="i2"),
        ResponseCompleted(),
    )
    await finish(session, run_task)

    assert session.cancels == [None]
    assert host.audio == [b"a", b"next"]


# -- tools -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_round_trip_submits_ordered_batch_with_one_continuation() -> None:
    executor = RecordingExecutor()
    session, host, orchestrator = make(executor=executor)
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ResponseStarted(response_id="r1"),
        ToolCall(call_id="c1", name="lookup", arguments='{"q": "one"}', response_id="r1"),
        ToolCall(call_id="c2", name="lookup", arguments='{"q": "two"}', response_id="r1"),
    )
    await settle()
    assert session.submissions == []  # nothing goes out before the response completes

    session.push(ResponseCompleted(response_id="r1"))
    await settle()
    await finish(session, run_task)

    assert executor.calls == [("lookup", {"q": "one"}), ("lookup", {"q": "two"})]
    assert len(session.submissions) == 1
    results, continue_response = session.submissions[0]
    assert [(r.call_id, r.output) for r in results] == [
        ("c1", "lookup:one"),
        ("c2", "lookup:two"),
    ]
    assert continue_response is True
    assert host.tool_calls == ["c1", "c2"]
    assert host.tool_results == [("c1", "lookup:one", "ok"), ("c2", "lookup:two", "ok")]
    assert orchestrator.pending_tool_calls == 0


@pytest.mark.asyncio
async def test_redelivered_call_id_runs_exactly_once() -> None:
    executor = RecordingExecutor()
    session, host, orchestrator = make(executor=executor)
    run_task = asyncio.create_task(orchestrator.run())
    call = ToolCall(call_id="c1", name="lookup", arguments='{"q": "one"}', response_id="r1")
    session.push(ResponseStarted(response_id="r1"), call, call, ResponseCompleted(response_id="r1"))
    await settle()
    session.push(call)
    await settle()
    await finish(session, run_task)

    assert len(executor.calls) == 1
    assert len(session.submissions) == 1
    assert [r.call_id for r in session.submissions[0][0]] == ["c1"]


@pytest.mark.asyncio
async def test_malformed_arguments_answer_with_an_error_result() -> None:
    executor = RecordingExecutor()
    session, host, orchestrator = make(executor=executor)
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ResponseStarted(response_id="r1"),
        ToolCall(call_id="c1", name="lookup", arguments="not json", response_id="r1"),
        ToolCall(call_id="c2", name="lookup", arguments="[1, 2]", response_id="r1"),
        ToolCall(call_id="c3", name="lookup", arguments="", response_id="r1"),
        ResponseCompleted(response_id="r1"),
    )
    await settle()
    await finish(session, run_task)

    assert executor.calls == [("lookup", {})]  # only the empty payload ran
    results, _ = session.submissions[0]
    assert [(r.call_id, r.output) for r in results] == [
        ("c1", UNPARSEABLE_ARGS_TEXT),
        ("c2", UNPARSEABLE_ARGS_TEXT),
        ("c3", "lookup:"),
    ]
    assert host.tool_results[0][2] == "error"


@pytest.mark.asyncio
async def test_without_executor_tools_are_refused_not_hung() -> None:
    session, host, orchestrator = make(executor=None)
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ToolCall(call_id="c1", name="lookup", arguments="{}", response_id="r1"),
        ResponseCompleted(response_id="r1"),
    )
    await settle()
    await finish(session, run_task)

    results, continue_response = session.submissions[0]
    assert [(r.call_id, r.output) for r in results] == [("c1", TOOLS_UNAVAILABLE_TEXT)]
    assert continue_response is True
    assert host.tool_results == [("c1", TOOLS_UNAVAILABLE_TEXT, "refused")]


@pytest.mark.asyncio
async def test_executor_exception_becomes_a_spoken_error_result() -> None:
    executor = RecordingExecutor(error=RuntimeError("boom"))
    session, host, orchestrator = make(executor=executor)
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ToolCall(call_id="c1", name="lookup", arguments="{}", response_id="r1"),
        ResponseCompleted(response_id="r1"),
    )
    await settle()
    await finish(session, run_task)

    results, _ = session.submissions[0]
    assert results[0].output == "The lookup tool failed: RuntimeError: boom"
    assert host.tool_results[0][2] == "error"


@pytest.mark.asyncio
async def test_tool_timeout_yields_an_honest_result_and_cancels_the_task() -> None:
    executor = RecordingExecutor(block=True)
    session, host, orchestrator = make(executor=executor, tool_timeout_s=0.05)
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ToolCall(call_id="c1", name="slow", arguments="{}", response_id="r1"),
        ResponseCompleted(response_id="r1"),
    )
    await asyncio.wait_for(executor.started.wait(), timeout=1)
    await asyncio.sleep(0.2)
    await finish(session, run_task)

    assert executor.cancelled == 1
    results, continue_response = session.submissions[0]
    assert "did not finish within 0.05 seconds" in results[0].output
    assert continue_response is True
    assert host.tool_results[0][2] == "timeout"


@pytest.mark.asyncio
async def test_barge_in_cancels_in_flight_tools_and_submits_without_continuation() -> None:
    executor = RecordingExecutor(block=True)
    session, host, orchestrator = make(executor=executor)
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ResponseStarted(response_id="r1"),
        ToolCall(call_id="c1", name="slow", arguments="{}", response_id="r1"),
        ResponseCompleted(response_id="r1"),
    )
    await asyncio.wait_for(executor.started.wait(), timeout=1)
    await settle()
    assert session.submissions == []

    session.push(InputSpeechStarted())
    await settle(12)
    await finish(session, run_task)

    assert executor.cancelled == 1
    results, continue_response = session.submissions[0]
    assert [(r.call_id, r.output) for r in results] == [("c1", INTERRUPTED_TEXT)]
    assert continue_response is False
    assert host.tool_results == [("c1", INTERRUPTED_TEXT, "interrupted")]


@pytest.mark.asyncio
async def test_barge_in_can_leave_tools_running_when_configured() -> None:
    executor = RecordingExecutor()
    session, host, orchestrator = make(executor=executor, cancel_tools_on_barge_in=False)
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ToolCall(call_id="c1", name="lookup", arguments='{"q": "x"}', response_id="r1"),
        InputSpeechStarted(),
        ResponseCompleted(response_id="r1"),
    )
    await settle()
    await finish(session, run_task)

    assert executor.cancelled == 0
    assert session.submissions[0][0][0].output == "lookup:x"


@pytest.mark.asyncio
async def test_provider_cancelled_calls_are_never_submitted() -> None:
    executor = RecordingExecutor(block_names=frozenset({"slow"}))
    session, host, orchestrator = make(executor=executor)
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ToolCall(call_id="c1", name="slow", arguments="{}", response_id="r1"),
        ToolCall(call_id="c2", name="lookup", arguments='{"q": "y"}'),
    )
    await asyncio.wait_for(executor.started.wait(), timeout=1)
    session.push(ToolCallCancelled(call_ids=("c1", "unknown")), ResponseCompleted(response_id="r1"))
    await settle(12)
    await finish(session, run_task)

    assert executor.cancelled == 1
    assert [(r.call_id, r.output) for results, _ in session.submissions for r in results] == [
        ("c2", "lookup:y")
    ]
    assert ("c1", "", "provider_cancelled") in host.tool_results


@pytest.mark.asyncio
async def test_call_without_response_id_settles_on_its_own() -> None:
    executor = RecordingExecutor()
    session, host, orchestrator = make(executor=executor)
    run_task = asyncio.create_task(orchestrator.run())
    session.push(ToolCall(call_id="c1", name="lookup", arguments='{"q": "solo"}'))
    await settle()
    await finish(session, run_task)

    assert [(r.call_id, r.output) for results, _ in session.submissions for r in results] == [
        ("c1", "lookup:solo")
    ]


@pytest.mark.asyncio
async def test_long_tool_output_is_bounded() -> None:
    async def executor(name, arguments):
        return {"blob": "x" * 500}

    session, host, orchestrator = make(executor=executor, max_tool_output_chars=100)
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ToolCall(call_id="c1", name="big", arguments="{}", response_id="r1"),
        ResponseCompleted(response_id="r1"),
    )
    await settle()
    await finish(session, run_task)

    output = session.submissions[0][0][0].output
    assert output.endswith(" …[truncated]")
    assert len(output) == 100 + len(" …[truncated]")
    assert output.startswith('{"blob": "xxx')


@pytest.mark.asyncio
async def test_failed_result_delivery_is_reported_not_fatal() -> None:
    executor = RecordingExecutor()
    session, host, orchestrator = make(executor=executor)
    session.submit_error = RuntimeError("socket gone")
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ToolCall(call_id="c1", name="lookup", arguments="{}", response_id="r1"),
        ResponseCompleted(response_id="r1"),
    )
    await settle()
    await finish(session, run_task)

    assert host.errors == [
        ("Tool results could not be delivered: RuntimeError: socket gone", False)
    ]


# -- close cleanup -----------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelling_run_stops_tools_and_closes_session() -> None:
    executor = RecordingExecutor(block=True)
    session, host, orchestrator = make(executor=executor)
    run_task = asyncio.create_task(orchestrator.run())
    session.push(ToolCall(call_id="c1", name="slow", arguments="{}", response_id="r1"))
    await asyncio.wait_for(executor.started.wait(), timeout=1)

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    assert executor.cancelled == 1
    assert session.close_count == 1
    assert session.submissions == []  # a closing session never receives late results
    leftovers = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and task.get_name().startswith("realtime-voice-")
    ]
    assert leftovers == []


@pytest.mark.asyncio
async def test_session_close_after_provider_closed_is_still_exactly_once() -> None:
    session, host, orchestrator = make()
    run_task = asyncio.create_task(orchestrator.run())
    await finish(session, run_task)
    await session.close()

    assert session.close_count == 1


@pytest.mark.asyncio
async def test_failed_barge_in_write_is_reported_and_the_call_continues() -> None:
    session, host, orchestrator = make(host=RecordingHost(played_ms=100))
    session.cancel_error = OSError("socket write failed")
    run_task = asyncio.create_task(orchestrator.run())
    session.push(
        ResponseStarted(response_id="r1"),
        OutputAudio(data=b"a", item_id="i1", response_id="r1"),
        InputSpeechStarted(),
        ResponseCompleted(response_id="r1"),
        ResponseStarted(response_id="r2"),
        OutputAudio(data=b"next", item_id="i2", response_id="r2"),
    )
    await finish(session, run_task)

    assert host.errors == [
        ("Could not cancel the response: OSError: socket write failed", False)
    ]
    assert session.truncations == [("i1", 100)]  # the second write still happened
    assert host.audio == [b"a", b"next"]


@pytest.mark.asyncio
async def test_host_callback_failure_inside_a_tool_task_is_logged_not_orphaned(caplog) -> None:
    class ExplodingHost(RecordingHost):
        def on_tool_result(self, call, output, outcome):
            raise RuntimeError("host callback broke")

    executor = RecordingExecutor()
    session, host, orchestrator = make(host=ExplodingHost(), executor=executor)
    run_task = asyncio.create_task(orchestrator.run())
    with caplog.at_level("WARNING", logger="agent.realtime_voice_orchestrator"):
        session.push(
            ToolCall(call_id="c1", name="lookup", arguments="{}", response_id="r1"),
            ResponseCompleted(response_id="r1"),
        )
        await settle(12)
        await finish(session, run_task)

    assert "realtime-voice-tool:c1 failed" in caplog.text
    assert "host callback broke" in caplog.text


# -- blocking tool bridge ----------------------------------------------------


@pytest.mark.asyncio
async def test_run_blocking_tool_returns_results_and_propagates_errors() -> None:
    def add(a, b):
        return a + b

    def fail():
        raise ValueError("nope")

    assert await run_blocking_tool(add, 1, b=2) == 3
    with pytest.raises(ValueError, match="nope"):
        await run_blocking_tool(fail)


@pytest.mark.asyncio
async def test_run_blocking_tool_runs_on_a_daemon_thread_and_survives_cancellation() -> None:
    release = threading.Event()
    seen: list[bool] = []

    def wedged():
        seen.append(threading.current_thread().daemon)
        release.wait(timeout=5)
        return "late"

    task = asyncio.create_task(run_blocking_tool(wedged))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.sleep(0.05)

    assert seen == [True]

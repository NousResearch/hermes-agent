"""Behavior tests for the typed, provider-neutral realtime voice contract."""

from __future__ import annotations

import asyncio

import pytest

import agent.realtime_voice_provider as realtime_voice_provider
from agent.realtime_voice_provider import (
    MAX_IDENTIFIER_LENGTH,
    PCM16_24K,
    InputAudioCommitted,
    InputSpeechStarted,
    InputSpeechStopped,
    InputTranscript,
    OutputAudio,
    OutputTranscript,
    RealtimeAudioFormat,
    RealtimeCapability,
    RealtimeTool,
    RealtimeToolResult,
    RealtimeVoiceEvent,
    RealtimeVoiceSession,
    RealtimeVoiceSetup,
    ResponseCompleted,
    ResponseStarted,
    SessionClosed,
    SessionFailure,
    SessionReady,
    SessionResumptionUpdate,
    ToolCall,
    ToolCallCancelled,
    UnsupportedRealtimeCapability,
)


# -- setup -------------------------------------------------------------------


def test_setup_copies_and_freezes_provider_options() -> None:
    tags = {"local"}
    options = {"region": "local", "tags": tags}

    setup = RealtimeVoiceSetup(model="model", voice="voice", provider_options=options)
    options["region"] = "changed"
    tags.add("mutated")

    assert setup.provider_options == {"region": "local", "tags": frozenset({"local"})}


def test_opaque_mappings_reject_values_that_cannot_be_frozen() -> None:
    class Mutable:
        pass

    with pytest.raises(TypeError, match="provider-neutral immutable value"):
        RealtimeVoiceSetup(provider_options={"native": Mutable()})
    with pytest.raises(TypeError, match="provider-neutral immutable value"):
        SessionReady(session_id="session", provider_data={"native": Mutable()})


def test_setup_rejects_provider_options_that_shadow_shared_fields() -> None:
    with pytest.raises(ValueError, match="shared setup field"):
        RealtimeVoiceSetup(model="model", provider_options={"model": "shadow"})
    with pytest.raises(ValueError, match="shared setup field"):
        RealtimeVoiceSetup(provider_options={"automatic_response": False})


@pytest.mark.parametrize("value", ["", " padded", "padded "])
def test_setup_validates_declared_model_voice_and_tool_names(value: str) -> None:
    with pytest.raises(ValueError, match="model"):
        RealtimeVoiceSetup(model=value)
    with pytest.raises(ValueError, match="voice"):
        RealtimeVoiceSetup(voice=value)
    with pytest.raises(ValueError, match="name"):
        RealtimeTool(name=value, description="description", parameters={})


def test_setup_declares_shared_audio_instructions_and_tools_immutably() -> None:
    schema = {"type": "object"}
    tools = [RealtimeTool(name="lookup", description="Look up", parameters=schema)]
    audio = RealtimeAudioFormat(mime_type="audio/pcm", sample_rate_hz=16_000, channels=1)

    setup = RealtimeVoiceSetup(
        model="model",
        voice="voice",
        instructions="Be concise",
        tools=tools,
        input_audio=audio,
        automatic_response=False,
    )
    schema["type"] = "changed"
    tools.clear()

    assert setup.tools[0].parameters == {"type": "object"}
    assert setup.input_audio is audio
    assert setup.output_audio is None
    assert setup.automatic_response is False


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: SessionReady(session_id="session", provider_data=value),
        lambda value: RealtimeVoiceSetup(provider_options=value),
        lambda value: ResponseStarted(response_id="response", metadata=value),
        lambda value: RealtimeTool("lookup", "description", value),
    ],
)
def test_mapping_contract_fields_require_mapping_shape_and_string_keys(factory) -> None:
    with pytest.raises(TypeError, match="Mapping"):
        factory([])
    with pytest.raises(TypeError, match="string keys"):
        factory({1: "value"})


@pytest.mark.parametrize("mime_type", ["", " audio/pcm", "audio/pcm ", "x" * 513])
def test_audio_format_rejects_blank_padded_or_oversized_mime_type(mime_type) -> None:
    with pytest.raises(ValueError, match="mime_type"):
        RealtimeAudioFormat(mime_type=mime_type, sample_rate_hz=24_000, channels=1)


@pytest.mark.parametrize("field_name", ["sample_rate_hz", "channels"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "1", None])
def test_audio_format_requires_positive_non_bool_integer_primitives(
    field_name, invalid
) -> None:
    values = {"mime_type": "audio/pcm", "sample_rate_hz": 24_000, "channels": 1}
    values[field_name] = invalid

    with pytest.raises((TypeError, ValueError), match=field_name):
        RealtimeAudioFormat(**values)


def test_audio_format_defaults_to_pcm16_mono_24k() -> None:
    assert PCM16_24K == RealtimeAudioFormat("audio/pcm", 24_000, 1)
    assert PCM16_24K.bytes_per_second == 48_000


@pytest.mark.parametrize("instructions", [None, 1, b"text", object()])
def test_setup_instructions_must_be_string(instructions) -> None:
    with pytest.raises(TypeError, match="instructions"):
        RealtimeVoiceSetup(instructions=instructions)


@pytest.mark.parametrize("audio", [{}, "audio/pcm", object()])
def test_setup_audio_must_be_none_or_realtime_audio_format(audio) -> None:
    with pytest.raises(TypeError, match="input_audio"):
        RealtimeVoiceSetup(input_audio=audio)
    with pytest.raises(TypeError, match="output_audio"):
        RealtimeVoiceSetup(output_audio=audio)


@pytest.mark.parametrize("tools", ["lookup", b"lookup", [object()]])
def test_setup_tools_reject_strings_and_non_tools(tools) -> None:
    with pytest.raises(TypeError, match="tools"):
        RealtimeVoiceSetup(tools=tools)


def test_setup_copies_tool_iterables_to_tuple() -> None:
    tool = RealtimeTool("lookup", "description", {})
    setup = RealtimeVoiceSetup(tools=(item for item in [tool]))

    assert setup.tools == (tool,)


def test_capabilities_are_explicit_and_immutable() -> None:
    assert {capability.value for capability in RealtimeCapability} == {
        "tool_calling",
        "input_transcription",
        "output_transcription",
        "input_commit_events",
        "explicit_response",
        "response_cancellation",
        "output_truncation",
        "tool_call_cancellation",
        "dynamic_context",
        "session_resumption",
    }


# -- events ------------------------------------------------------------------


def test_shared_event_vocabulary_is_typed_and_provider_data_is_opaque() -> None:
    tags = {"tag"}
    provider_data = {"native": {"values": ["value"], "tags": tags}}
    events = [
        SessionReady(session_id="session", provider_data=provider_data),
        SessionClosed(reason="normal"),
        SessionFailure(code="transport", message="lost"),
        SessionFailure(code="frame", message="one bad frame", terminal=False),
        SessionResumptionUpdate(handle="handle", resumable=True),
        InputSpeechStarted(item_id="input", audio_start_ms=120),
        InputSpeechStopped(item_id="input", audio_end_ms=900),
        InputAudioCommitted(item_id="input"),
        InputTranscript(text="hello", final=True, item_id="input"),
        ResponseStarted(response_id="response", metadata={"correlation": "abc"}),
        OutputTranscript(text="hi", final=False, item_id="output", response_id="response"),
        OutputAudio(data=b"pcm", item_id="output", response_id="response"),
        ToolCall(call_id="call", name="lookup", arguments='{"q": "x"}', response_id="response"),
        ToolCallCancelled(call_ids=("call",)),
        ResponseCompleted(response_id="response", status="completed"),
    ]
    provider_data["native"]["values"].append("changed")
    tags.add("mutated")

    assert len(events) == 15
    assert all(isinstance(event, RealtimeVoiceEvent) for event in events)
    assert events[0].provider_data == {
        "native": {"values": ("value",), "tags": frozenset({"tag"})}
    }
    assert events[9].metadata == {"correlation": "abc"}


def test_output_audio_copies_mutable_buffers() -> None:
    source = bytearray(b"abc")
    event = OutputAudio(data=source, item_id="item", response_id="response")

    source[0] = ord("z")

    assert event.data == b"abc"
    assert isinstance(event.data, bytes)


def test_provider_data_cannot_shadow_shared_event_fields() -> None:
    with pytest.raises(ValueError, match="provider_data.*session_id"):
        SessionReady(session_id="session", provider_data={"session_id": "spoof"})


@pytest.mark.parametrize("session_id", ["", " padded", "padded "])
def test_shared_events_reject_blank_or_malformed_identifiers(session_id: str) -> None:
    with pytest.raises(ValueError, match="session_id"):
        SessionReady(session_id=session_id)


def test_shared_events_reject_oversized_identifiers() -> None:
    with pytest.raises(ValueError, match="session_id"):
        SessionReady(session_id="x" * (MAX_IDENTIFIER_LENGTH + 1))


@pytest.mark.parametrize(
    ("factory", "field_name"),
    [
        (lambda value: ToolCall(value, "lookup", "{}"), "call_id"),
        (lambda value: ToolCall("call", value, "{}"), "name"),
        (lambda value: ToolCallCancelled((value,)), "call_id"),
        (lambda value: SessionFailure(value, "message"), "code"),
        (lambda value: RealtimeToolResult(value, "output"), "call_id"),
    ],
)
def test_all_required_shared_identifiers_and_names_are_validated(factory, field_name) -> None:
    for invalid in ("", " padded", "padded ", "x" * (MAX_IDENTIFIER_LENGTH + 1)):
        with pytest.raises(ValueError, match=field_name):
            factory(invalid)


@pytest.mark.parametrize(
    ("factory", "field_name"),
    [
        (lambda value: InputTranscript("text", True, item_id=value), "item_id"),
        (lambda value: OutputTranscript("text", True, response_id=value), "response_id"),
        (lambda value: OutputAudio(b"pcm", item_id=value), "item_id"),
        (lambda value: ToolCall("call", "lookup", "{}", response_id=value), "response_id"),
        (lambda value: ResponseStarted(value), "response_id"),
        (lambda value: ResponseCompleted(value), "response_id"),
        (lambda value: InputSpeechStarted(item_id=value), "item_id"),
        (lambda value: SessionResumptionUpdate(value, True), "handle"),
    ],
)
def test_optional_identifiers_accept_none_but_reject_malformed_strings(
    factory, field_name
) -> None:
    factory(None)
    for invalid in ("", " padded", "padded "):
        with pytest.raises(ValueError, match=field_name):
            factory(invalid)


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "10"])
def test_speech_offsets_must_be_none_or_non_negative_integers(invalid) -> None:
    with pytest.raises((TypeError, ValueError), match="audio_start_ms"):
        InputSpeechStarted(audio_start_ms=invalid)
    with pytest.raises((TypeError, ValueError), match="audio_end_ms"):
        InputSpeechStopped(audio_end_ms=invalid)


def test_tool_call_cancellation_requires_identifier_sequence() -> None:
    with pytest.raises(TypeError, match="sequence"):
        ToolCallCancelled(call_ids="call")
    with pytest.raises(ValueError, match="at least one"):
        ToolCallCancelled(call_ids=())

    cancelled = ToolCallCancelled(call_ids=["a", "b"])
    assert cancelled.call_ids == ("a", "b")


def test_tool_call_carries_raw_argument_text_for_the_host_to_parse() -> None:
    call = ToolCall(call_id="call", name="lookup", arguments="not json")

    assert call.arguments == "not json"
    with pytest.raises(TypeError, match="arguments"):
        ToolCall(call_id="call", name="lookup", arguments={"q": 1})  # type: ignore[arg-type]


def test_failure_terminal_flag_and_metadata_values_are_typed() -> None:
    with pytest.raises(TypeError, match="terminal"):
        SessionFailure(code="x", message="m", terminal="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="metadata values"):
        ResponseStarted(response_id="r", metadata={"n": 1})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="final"):
        InputTranscript(text="t", final="yes")  # type: ignore[arg-type]


# -- session -----------------------------------------------------------------


class _Session(RealtimeVoiceSession):
    def __init__(self, capabilities=(), **formats) -> None:
        super().__init__(capabilities, **formats)
        self.closed_count = 0
        self.submitted = []
        self.event_values = []

    async def send_audio(self, audio: bytes) -> None:
        return None

    async def _submit_tool_results(self, results, continue_response) -> None:
        self.submitted.append((results, continue_response))

    def _events(self):
        async def stream():
            for event in self.event_values:
                yield event

        return stream()

    async def _close(self) -> None:
        await asyncio.sleep(0)
        self.closed_count += 1


class _ControlledCloseSession(_Session):
    def __init__(self, outcomes=()) -> None:
        super().__init__()
        self.close_entered = asyncio.Event()
        self.release_close = asyncio.Event()
        self.outcomes = list(outcomes)
        self.completed_close_count = 0

    async def _close(self) -> None:
        self.closed_count += 1
        self.close_entered.set()
        await self.release_close.wait()
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if outcome is not None:
                raise outcome
        self.completed_close_count += 1


class _FinalizingEventSession(_Session):
    def __init__(self, terminal) -> None:
        super().__init__()
        self.terminal = terminal
        self.stream_finalized = asyncio.Event()

    def _events(self):
        async def stream():
            try:
                yield self.terminal
                yield SessionReady(session_id="late")
            finally:
                self.stream_finalized.set()

        return stream()


class _NoOptionalHooksSession(RealtimeVoiceSession):
    async def send_audio(self, audio: bytes) -> None:
        return None

    def _events(self):
        async def stream():
            if False:
                yield RealtimeVoiceEvent()

        return stream()

    async def _close(self) -> None:
        return None


def test_provider_api_version_is_two() -> None:
    assert realtime_voice_provider.REALTIME_VOICE_PROVIDER_API_VERSION == 2


def test_session_reports_negotiated_audio_formats() -> None:
    narrow = RealtimeAudioFormat(sample_rate_hz=16_000)
    session = _Session(input_audio=narrow)

    assert session.input_audio_format is narrow
    assert session.output_audio_format is PCM16_24K
    with pytest.raises(TypeError, match="output_audio"):
        _Session(output_audio="pcm")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_optional_operations_fail_with_typed_capability_error() -> None:
    session = _Session()
    operations = [
        (session.commit_audio(), RealtimeCapability.INPUT_COMMIT_EVENTS),
        (session.create_response(), RealtimeCapability.EXPLICIT_RESPONSE),
        (session.cancel_response(), RealtimeCapability.RESPONSE_CANCELLATION),
        (session.truncate_output("item", 10), RealtimeCapability.OUTPUT_TRUNCATION),
        (session.add_context("item", "text"), RealtimeCapability.DYNAMIC_CONTEXT),
        (session.remove_context("item"), RealtimeCapability.DYNAMIC_CONTEXT),
        (
            session.submit_tool_results([RealtimeToolResult("call", "done")]),
            RealtimeCapability.TOOL_CALLING,
        ),
    ]

    for operation, capability in operations:
        with pytest.raises(UnsupportedRealtimeCapability) as exc:
            await operation
        assert exc.value.capability is capability
        assert session.supports(capability) is False


@pytest.mark.asyncio
async def test_optional_operation_arguments_are_validated_before_capability() -> None:
    session = _Session()

    with pytest.raises(ValueError, match="item_id"):
        await session.truncate_output("", 10)
    with pytest.raises(TypeError, match="audio_end_ms"):
        await session.truncate_output("item", True)
    with pytest.raises(ValueError, match="audio_end_ms"):
        await session.truncate_output("item", -1)
    with pytest.raises(ValueError, match="response_id"):
        await session.cancel_response(" padded")
    with pytest.raises(TypeError, match="metadata values"):
        await session.create_response(metadata={"n": 1})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    "capability",
    [
        RealtimeCapability.TOOL_CALLING,
        RealtimeCapability.INPUT_COMMIT_EVENTS,
        RealtimeCapability.EXPLICIT_RESPONSE,
        RealtimeCapability.RESPONSE_CANCELLATION,
        RealtimeCapability.OUTPUT_TRUNCATION,
        RealtimeCapability.DYNAMIC_CONTEXT,
    ],
)
def test_operational_capability_requires_subclass_hook_override(capability) -> None:
    with pytest.raises(ValueError, match=rf"{capability.value}.*override"):
        _NoOptionalHooksSession({capability})


def test_dynamic_context_requires_both_hooks() -> None:
    class _HalfContext(_NoOptionalHooksSession):
        async def _add_context(self, item_id, text):
            return None

    with pytest.raises(ValueError, match="_remove_context"):
        _HalfContext({RealtimeCapability.DYNAMIC_CONTEXT})


def test_passive_capabilities_do_not_require_hooks() -> None:
    session = _NoOptionalHooksSession(
        {
            RealtimeCapability.INPUT_TRANSCRIPTION,
            RealtimeCapability.OUTPUT_TRANSCRIPTION,
            RealtimeCapability.TOOL_CALL_CANCELLATION,
            RealtimeCapability.SESSION_RESUMPTION,
        }
    )

    assert RealtimeCapability.SESSION_RESUMPTION in session.capabilities


@pytest.mark.asyncio
async def test_event_stream_rejects_non_contract_values() -> None:
    session = _Session()
    session.event_values = [{"type": "response.started"}]

    with pytest.raises(TypeError, match="RealtimeVoiceEvent"):
        await anext(session.events())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal",
    [SessionClosed(reason="normal"), SessionFailure(code="transport", message="lost")],
)
async def test_event_stream_stops_after_first_terminal_session_event(terminal) -> None:
    session = _FinalizingEventSession(terminal)

    assert [event async for event in session.events()] == [terminal]
    assert session.stream_finalized.is_set()


@pytest.mark.asyncio
async def test_non_terminal_failure_keeps_the_stream_open() -> None:
    session = _Session()
    session.event_values = [
        SessionFailure(code="frame", message="one bad frame", terminal=False),
        SessionReady(session_id="session"),
        SessionClosed(),
    ]

    events = [event async for event in session.events()]

    assert [type(event) for event in events] == [SessionFailure, SessionReady, SessionClosed]


@pytest.mark.asyncio
async def test_close_is_idempotent_under_concurrent_callers_and_context_exit() -> None:
    session = _Session()

    async with session:
        await asyncio.gather(session.close(), session.close(), session.close())

    assert session.closed_count == 1
    assert session.closed is True


@pytest.mark.asyncio
async def test_cancelled_close_caller_leaves_cleanup_owned_for_second_close() -> None:
    session = _ControlledCloseSession()
    first = asyncio.create_task(session.close())
    await session.close_entered.wait()

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second = asyncio.create_task(session.close())
    session.release_close.set()
    await second
    await session.close()

    assert session.closed_count == 1
    assert session.completed_close_count == 1


@pytest.mark.asyncio
async def test_cancel_racing_success_does_not_repeat_cleanup() -> None:
    session = _ControlledCloseSession()
    first = asyncio.create_task(session.close())
    await session.close_entered.wait()

    session.release_close.set()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    await session.close()

    assert session.closed_count == 1
    assert session.completed_close_count == 1


@pytest.mark.asyncio
async def test_cancelled_waiter_cleanup_failure_is_observed_and_retryable() -> None:
    session = _ControlledCloseSession([RuntimeError("cleanup failed"), None])
    loop = asyncio.get_running_loop()
    orphaned = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: orphaned.append(context))
    try:
        first = asyncio.create_task(session.close())
        await session.close_entered.wait()

        session.release_close.set()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await asyncio.sleep(0)

        await session.close()
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert session.closed_count == 2
    assert session.completed_close_count == 1
    assert orphaned == []


@pytest.mark.asyncio
async def test_close_exception_allows_later_retry() -> None:
    session = _ControlledCloseSession([RuntimeError("cleanup failed"), None])
    session.release_close.set()

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await session.close()

    await session.close()
    await session.close()

    assert session.closed_count == 2
    assert session.completed_close_count == 1

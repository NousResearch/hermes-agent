"""Behavior tests for the typed, provider-neutral realtime voice contract."""

from __future__ import annotations

import asyncio

import pytest

from agent.realtime_voice_provider import (
    InputTranscript,
    Interruption,
    OutputAudio,
    OutputTranscript,
    RealtimeCapability,
    RealtimeAudioFormat,
    ResponseCompleted,
    ResponseStarted,
    SessionClosed,
    SessionFailure,
    SessionReady,
    ToolCall,
    ToolCallCancelled,
    RealtimeTool,
    RealtimeToolResult,
    RealtimeVoiceEvent,
    RealtimeVoiceSession,
    RealtimeVoiceSetup,
    TranscriptProvenance,
    TranscriptRole,
    TurnCompleted,
    TurnStarted,
    UnsupportedRealtimeCapability,
    MAX_IDENTIFIER_LENGTH,
)


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
    audio = RealtimeAudioFormat(mime_type="audio/pcm", sample_rate_hz=24_000, channels=1)

    setup = RealtimeVoiceSetup(
        model="model",
        voice="voice",
        instructions="Be concise",
        tools=tools,
        audio=audio,
    )
    schema["type"] = "changed"
    tools.clear()

    assert setup.tools[0].parameters == {"type": "object"}
    assert setup.audio is audio


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: SessionReady(session_id="session", provider_data=value),
        lambda value: RealtimeVoiceSetup(provider_options=value),
        lambda value: ToolCall(
            "call", "batch", "turn", "response", "lookup", value
        ),
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


@pytest.mark.parametrize("instructions", [1, b"text", object()])
def test_setup_instructions_must_be_none_or_string(instructions) -> None:
    with pytest.raises(TypeError, match="instructions"):
        RealtimeVoiceSetup(instructions=instructions)


@pytest.mark.parametrize("audio", [{}, "audio/pcm", object()])
def test_setup_audio_must_be_none_or_realtime_audio_format(audio) -> None:
    with pytest.raises(TypeError, match="audio"):
        RealtimeVoiceSetup(audio=audio)


@pytest.mark.parametrize("tools", ["lookup", b"lookup", [object()]])
def test_setup_tools_reject_strings_and_non_tools(tools) -> None:
    with pytest.raises(TypeError, match="tools"):
        RealtimeVoiceSetup(tools=tools)


def test_setup_copies_tool_iterables_to_tuple() -> None:
    tool = RealtimeTool("lookup", "description", {})
    setup = RealtimeVoiceSetup(tools=(item for item in [tool]))

    assert setup.tools == (tool,)


def test_capabilities_are_explicit_and_immutable() -> None:
    capabilities = frozenset(RealtimeCapability)

    assert {capability.value for capability in capabilities} == {
        "tool_calling",
        "explicit_interruption",
        "output_truncation",
        "input_commit_events",
        "response_metadata_echo",
        "tool_call_cancellation",
        "dynamic_context",
        "session_resumption",
        "input_transcription",
        "output_transcription",
        "continuation",
    }


def test_shared_event_vocabulary_is_typed_and_provider_data_is_opaque() -> None:
    tags = {"tag"}
    provider_data = {"native": {"values": ["value"], "tags": tags}}
    events = [
        SessionReady(session_id="session", provider_data=provider_data),
        SessionClosed(reason="normal"),
        SessionFailure(code="transport", message="lost"),
        InputTranscript(
            item_id="input",
            turn_id="turn",
            text="hello",
            final=True,
            role=TranscriptRole.PARTICIPANT,
            provenance=TranscriptProvenance.PARTICIPANT_INPUT_AUDIO,
        ),
        OutputTranscript(
            item_id="output",
            turn_id="turn",
            response_id="response",
            text="hi",
            final=False,
        ),
        OutputAudio(
            data=b"pcm",
            item_id="output",
            turn_id="turn",
            response_id="response",
        ),
        ToolCall(call_id="call", batch_id="batch", turn_id="turn", response_id="response", name="lookup", arguments={"q": "x"}),
        ToolCallCancelled(call_id="call", batch_id="batch"),
        TurnStarted(turn_id="turn"),
        TurnCompleted(turn_id="turn"),
        ResponseStarted(response_id="response", turn_id="turn"),
        ResponseCompleted(response_id="response", turn_id="turn"),
        Interruption(response_id="response", turn_id="turn"),
    ]
    provider_data["native"]["values"].append("changed")
    tags.add("mutated")

    assert len(events) == 13
    assert events[0].session_id == "session"
    assert events[0].provider_data == {
        "native": {"values": ("value",), "tags": frozenset({"tag"})}
    }


def test_output_audio_copies_mutable_buffers() -> None:
    source = bytearray(b"abc")
    event = OutputAudio(
        data=source, item_id="item", turn_id="turn", response_id="response"
    )

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
        (
            lambda value: InputTranscript(
                value,
                "turn",
                "text",
                True,
                TranscriptRole.OPERATOR,
                TranscriptProvenance.OPERATOR_INPUT,
            ),
            "item_id",
        ),
        (
            lambda value: OutputTranscript(
                value, "turn", "response", "text", True
            ),
            "item_id",
        ),
        (
            lambda value: OutputAudio(b"pcm", value, "turn", "response"),
            "item_id",
        ),
        (lambda value: ToolCall(value, "batch", "turn", "response", "lookup", {}), "call_id"),
        (lambda value: ToolCall("call", "batch", "turn", "response", value, {}), "name"),
        (lambda value: ToolCallCancelled(value, "batch"), "call_id"),
        (lambda value: TurnStarted(value), "turn_id"),
        (lambda value: TurnCompleted(value), "turn_id"),
        (lambda value: ResponseStarted(value, "turn"), "response_id"),
        (lambda value: ResponseCompleted(value, "turn"), "response_id"),
        (lambda value: Interruption(value, "turn"), "response_id"),
        (lambda value: SessionFailure(value, "message"), "code"),
    ],
)
def test_all_required_shared_identifiers_and_names_are_validated(factory, field_name) -> None:
    for invalid in ("", " padded", "padded ", "x" * (MAX_IDENTIFIER_LENGTH + 1)):
        with pytest.raises(ValueError, match=field_name):
            factory(invalid)


def test_transcript_attribution_rejects_contradictory_provenance() -> None:
    with pytest.raises(ValueError, match="role.*provenance"):
        InputTranscript(
            item_id="input",
            turn_id="turn",
            text="spoofed",
            final=True,
            role=TranscriptRole.ASSISTANT,
            provenance=TranscriptProvenance.PARTICIPANT_INPUT_AUDIO,
        )

    with pytest.raises(ValueError, match="role.*provenance"):
        OutputTranscript(
            item_id="output",
            turn_id="turn",
            response_id="response",
            text="spoofed",
            final=True,
            role=TranscriptRole.OPERATOR,
            provenance=TranscriptProvenance.ASSISTANT_OUTPUT_AUDIO,
        )


def test_tool_work_has_stable_neutral_batch_and_response_identity() -> None:
    arguments = {"nested": ["value"]}
    output = {"items": [1]}

    call = ToolCall(
        call_id="call",
        batch_id="batch",
        turn_id="turn",
        response_id="response",
        name="lookup",
        arguments=arguments,
    )
    cancelled = ToolCallCancelled(call_id="call", batch_id="batch")
    result = RealtimeToolResult(
        call_id="call", batch_id="batch", name="lookup", output=output
    )
    continuation = ResponseStarted(
        response_id="next-response",
        turn_id="turn",
        continuation_of_batch_id="batch",
    )
    arguments["nested"].append("mutated")
    output["items"].append(2)

    assert call.arguments == {"nested": ("value",)}
    assert cancelled.batch_id == result.batch_id == "batch"
    assert result.output == {"items": (1,)}
    assert continuation.continuation_of_batch_id == "batch"


def test_transcript_audio_and_terminal_events_have_explicit_association() -> None:
    input_event = InputTranscript(
        item_id="input",
        turn_id="turn",
        text="hello",
        final=True,
        role=TranscriptRole.OPERATOR,
        provenance=TranscriptProvenance.OPERATOR_INPUT,
    )
    transcript = OutputTranscript(
        item_id="output", turn_id="turn", response_id="response", text="hi", final=True
    )
    audio = OutputAudio(
        data=b"pcm", item_id="output", turn_id="turn", response_id="response"
    )
    completed = ResponseCompleted(
        response_id="response", turn_id="turn", continuation_of_batch_id="batch"
    )
    interrupted = Interruption(response_id="response", turn_id="turn")

    assert input_event.turn_id == transcript.turn_id == audio.turn_id == "turn"
    assert transcript.response_id == audio.response_id == completed.response_id
    assert completed.continuation_of_batch_id == "batch"
    assert interrupted.turn_id == "turn"


class _Session(RealtimeVoiceSession):
    def __init__(self, capabilities=()) -> None:
        super().__init__(capabilities)
        self.closed_count = 0
        self.submitted = []
        self.continuations = []

    async def send_audio(self, audio: bytes, *, mime_type: str | None = None) -> None:
        return None

    async def _submit_tool_results(self, batch_id, results) -> None:
        self.submitted.append((batch_id, results))

    def events(self):
        async def stream():
            if False:
                yield RealtimeVoiceEvent()

        return stream()

    async def _continue_response(self, batch_id: str) -> None:
        self.continuations.append(batch_id)

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


class _NoOptionalHooksSession(RealtimeVoiceSession):
    async def send_audio(self, audio: bytes, *, mime_type: str | None = None) -> None:
        return None

    async def _submit_tool_results(self, batch_id, results) -> None:
        return None

    def events(self):
        async def stream():
            if False:
                yield RealtimeVoiceEvent()

        return stream()

    async def _close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_tool_results_require_tool_calling_and_continuation_is_separate() -> None:
    result = RealtimeToolResult(
        call_id="call", batch_id="batch", name="lookup", output="done"
    )
    unsupported = _Session()

    with pytest.raises(UnsupportedRealtimeCapability) as exc:
        await unsupported.submit_tool_results("batch", [result])
    assert exc.value.capability is RealtimeCapability.TOOL_CALLING

    session = _Session(
        {RealtimeCapability.TOOL_CALLING, RealtimeCapability.CONTINUATION}
    )
    await session.submit_tool_results("batch", [result])
    assert session.submitted == [("batch", (result,))]
    assert session.continuations == []

    await session.continue_response("batch")
    assert session.continuations == ["batch"]


@pytest.mark.asyncio
async def test_optional_operations_fail_with_typed_capability_error() -> None:
    session = _Session()
    operations = [
        (session.commit_audio(), RealtimeCapability.INPUT_COMMIT_EVENTS),
        (session.interrupt(), RealtimeCapability.EXPLICIT_INTERRUPTION),
        (
            session.truncate_output("response", "item"),
            RealtimeCapability.OUTPUT_TRUNCATION,
        ),
        (
            session.cancel_tool_call("call", "batch"),
            RealtimeCapability.TOOL_CALL_CANCELLATION,
        ),
        (session.resume_session("session"), RealtimeCapability.SESSION_RESUMPTION),
        (session.update_context("instructions"), RealtimeCapability.DYNAMIC_CONTEXT),
        (session.continue_response("batch"), RealtimeCapability.CONTINUATION),
    ]

    for operation, capability in operations:
        with pytest.raises(UnsupportedRealtimeCapability) as exc:
            await operation
        assert exc.value.capability is capability


@pytest.mark.parametrize(
    "capability",
    [
        RealtimeCapability.EXPLICIT_INTERRUPTION,
        RealtimeCapability.OUTPUT_TRUNCATION,
        RealtimeCapability.INPUT_COMMIT_EVENTS,
        RealtimeCapability.TOOL_CALL_CANCELLATION,
        RealtimeCapability.DYNAMIC_CONTEXT,
        RealtimeCapability.SESSION_RESUMPTION,
        RealtimeCapability.CONTINUATION,
    ],
)
def test_operational_capability_requires_subclass_hook_override(capability) -> None:
    with pytest.raises(ValueError, match=rf"{capability.value}.*override"):
        _NoOptionalHooksSession({capability})


def test_passive_capabilities_do_not_require_hooks() -> None:
    session = _NoOptionalHooksSession(
        {
            RealtimeCapability.RESPONSE_METADATA_ECHO,
            RealtimeCapability.INPUT_TRANSCRIPTION,
            RealtimeCapability.OUTPUT_TRANSCRIPTION,
            RealtimeCapability.TOOL_CALLING,
        }
    )

    assert RealtimeCapability.TOOL_CALLING in session.capabilities


@pytest.mark.asyncio
async def test_close_is_idempotent_under_concurrent_callers_and_context_exit() -> None:
    session = _Session()

    async with session:
        await asyncio.gather(session.close(), session.close(), session.close())

    assert session.closed_count == 1


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
async def test_close_exception_allows_later_retry() -> None:
    session = _ControlledCloseSession([RuntimeError("cleanup failed"), None])
    session.release_close.set()

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await session.close()

    await session.close()
    await session.close()

    assert session.closed_count == 2
    assert session.completed_close_count == 1

"""
Realtime Voice Provider ABC
===========================

Defines the provider-neutral contract for low-latency, bidirectional voice
sessions. Realtime voice is intentionally separate from
:mod:`agent.transports`: model transports execute one request/response turn,
while a realtime session is a long-lived async channel carrying audio, text,
tool calls, interruptions, and lifecycle events.

Providers own their wire protocol and SDK. Concrete backends translate native
events into the frozen typed events below; provider-specific state stays under
``provider_data`` instead of leaking into Hermes callers.

Event contract
--------------
:meth:`RealtimeVoiceSession.events` yields :class:`RealtimeVoiceEvent`
instances. Shared event families are:

``SessionReady`` / ``SessionClosed`` / ``SessionFailure``
    Session lifecycle.
``InputTranscript``
    User speech transcription in ``text``.
``OutputTranscript``
    Assistant text in ``text``.
``OutputAudio``
    Assistant audio bytes in ``audio``.
``ToolCall`` / ``ToolCallCancelled``
    Stable batch-bound tool requests and cancellation signals.
``TurnStarted`` / ``TurnCompleted`` / ``ResponseStarted`` /
``ResponseCompleted`` / ``Interruption``
    Turn, response, continuation, and barge-in lifecycle signals.

Unknown provider-native fields belong in ``provider_data``. New normalized
event names may be added without changing the provider API version.
"""

from __future__ import annotations

import abc
import asyncio
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import StrEnum
from types import MappingProxyType
from typing import Any, AsyncIterator, Iterable

REALTIME_VOICE_PROVIDER_API_VERSION = 2
MAX_IDENTIFIER_LENGTH = 512
MAX_OUTSTANDING_CONTINUATIONS = 128
MAX_TRACKED_CONTINUATION_RESPONSES = 1024
MAX_TRACKED_ORDINARY_RESPONSES = 1024


class RealtimeCapability(StrEnum):
    TOOL_CALLING = "tool_calling"
    EXPLICIT_INTERRUPTION = "explicit_interruption"
    OUTPUT_TRUNCATION = "output_truncation"
    INPUT_COMMIT_EVENTS = "input_commit_events"
    RESPONSE_METADATA_ECHO = "response_metadata_echo"
    TOOL_CALL_CANCELLATION = "tool_call_cancellation"
    DYNAMIC_CONTEXT = "dynamic_context"
    SESSION_RESUMPTION = "session_resumption"
    INPUT_TRANSCRIPTION = "input_transcription"
    OUTPUT_TRANSCRIPTION = "output_transcription"
    CONTINUATION = "continuation"


class UnsupportedRealtimeCapability(RuntimeError):
    """Raised when a session cannot provide requested optional semantics."""

    def __init__(self, capability: RealtimeCapability) -> None:
        self.capability = capability
        super().__init__(f"realtime capability is unsupported: {capability.value}")


class TranscriptRole(StrEnum):
    OPERATOR = "operator"
    PARTICIPANT = "participant"
    ASSISTANT = "assistant"


class TranscriptProvenance(StrEnum):
    OPERATOR_INPUT = "operator_input"
    PARTICIPANT_INPUT_AUDIO = "participant_input_audio"
    ASSISTANT_OUTPUT_AUDIO = "assistant_output_audio"


def _validate_identifier(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > MAX_IDENTIFIER_LENGTH
    ):
        raise ValueError(
            f"{field_name} must be a nonblank, trimmed identifier no longer than "
            f"{MAX_IDENTIFIER_LENGTH} characters"
        )


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("provider-neutral mappings require string keys")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    raise TypeError(
        f"provider-neutral immutable value required, got {type(value).__name__}"
    )


def _freeze_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a Mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} must have string keys")
    return _freeze(value)


def _freeze_provider_data(event: Any) -> None:
    provider_data = _freeze_mapping(event.provider_data, "provider_data")
    shared_fields = {item.name for item in fields(event)} - {"provider_data"}
    shadowed = shared_fields.intersection(provider_data)
    if shadowed:
        raise ValueError(
            f"provider_data cannot shadow shared event field: {min(shadowed)}"
        )
    object.__setattr__(event, "provider_data", provider_data)


class RealtimeVoiceEvent:
    """Marker for normalized provider-neutral session events.

    ``provider_data`` is diagnostic-only. Hosts own tool authorization,
    participant identity, and memory policy; opaque data is never authority.
    """


@dataclass(frozen=True, slots=True)
class SessionReady(RealtimeVoiceEvent):
    session_id: str
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.session_id, "session_id")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class SessionClosed(RealtimeVoiceEvent):
    reason: str = ""
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class SessionFailure(RealtimeVoiceEvent):
    code: str
    message: str
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.code, "code")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class InputTranscript(RealtimeVoiceEvent):
    item_id: str
    turn_id: str
    text: str
    final: bool
    role: TranscriptRole
    provenance: TranscriptProvenance
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.item_id, "item_id")
        _validate_identifier(self.turn_id, "turn_id")
        expected_provenance = {
            TranscriptRole.OPERATOR: TranscriptProvenance.OPERATOR_INPUT,
            TranscriptRole.PARTICIPANT: TranscriptProvenance.PARTICIPANT_INPUT_AUDIO,
        }.get(self.role)
        if self.provenance is not expected_provenance:
            raise ValueError("input transcript role contradicts provenance")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class OutputTranscript(RealtimeVoiceEvent):
    item_id: str
    turn_id: str
    response_id: str
    text: str
    final: bool
    role: TranscriptRole = TranscriptRole.ASSISTANT
    provenance: TranscriptProvenance = TranscriptProvenance.ASSISTANT_OUTPUT_AUDIO
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.item_id, "item_id")
        _validate_identifier(self.turn_id, "turn_id")
        _validate_identifier(self.response_id, "response_id")
        if (
            self.role is not TranscriptRole.ASSISTANT
            or self.provenance is not TranscriptProvenance.ASSISTANT_OUTPUT_AUDIO
        ):
            raise ValueError("output transcript role contradicts provenance")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class OutputAudio(RealtimeVoiceEvent):
    data: bytes
    item_id: str
    turn_id: str
    response_id: str
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.item_id, "item_id")
        _validate_identifier(self.turn_id, "turn_id")
        _validate_identifier(self.response_id, "response_id")
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise TypeError("output audio data must be bytes-like")
        object.__setattr__(self, "data", bytes(self.data))
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class ToolCall(RealtimeVoiceEvent):
    call_id: str
    batch_id: str
    turn_id: str
    response_id: str
    name: str
    arguments: Mapping[str, Any]
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.call_id, "call_id")
        _validate_identifier(self.batch_id, "batch_id")
        _validate_identifier(self.turn_id, "turn_id")
        _validate_identifier(self.response_id, "response_id")
        _validate_identifier(self.name, "name")
        object.__setattr__(
            self, "arguments", _freeze_mapping(self.arguments, "arguments")
        )
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class ToolCallCancelled(RealtimeVoiceEvent):
    call_id: str
    batch_id: str
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.call_id, "call_id")
        _validate_identifier(self.batch_id, "batch_id")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class RealtimeToolResult:
    """One immutable orchestrator-owned result in a stable ordered batch."""

    call_id: str
    batch_id: str
    name: str
    output: Any

    def __post_init__(self) -> None:
        _validate_identifier(self.call_id, "call_id")
        _validate_identifier(self.batch_id, "batch_id")
        _validate_identifier(self.name, "name")
        object.__setattr__(self, "output", _freeze(self.output))


@dataclass(frozen=True, slots=True)
class TurnStarted(RealtimeVoiceEvent):
    turn_id: str
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.turn_id, "turn_id")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class TurnCompleted(RealtimeVoiceEvent):
    turn_id: str
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.turn_id, "turn_id")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class ResponseStarted(RealtimeVoiceEvent):
    response_id: str
    turn_id: str
    continuation_of_batch_id: str | None = None
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.response_id, "response_id")
        _validate_identifier(self.turn_id, "turn_id")
        if self.continuation_of_batch_id is not None:
            _validate_identifier(self.continuation_of_batch_id, "continuation_of_batch_id")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class ResponseCompleted(RealtimeVoiceEvent):
    response_id: str
    turn_id: str
    continuation_of_batch_id: str | None = None
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.response_id, "response_id")
        _validate_identifier(self.turn_id, "turn_id")
        if self.continuation_of_batch_id is not None:
            _validate_identifier(self.continuation_of_batch_id, "continuation_of_batch_id")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class Interruption(RealtimeVoiceEvent):
    response_id: str
    turn_id: str
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.response_id, "response_id")
        _validate_identifier(self.turn_id, "turn_id")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class RealtimeAudioFormat:
    mime_type: str
    sample_rate_hz: int
    channels: int

    def __post_init__(self) -> None:
        _validate_identifier(self.mime_type, "mime_type")
        for field_name in ("sample_rate_hz", "channels"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class RealtimeTool:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "name")
        object.__setattr__(
            self, "parameters", _freeze_mapping(self.parameters, "parameters")
        )


@dataclass(frozen=True, slots=True)
class RealtimeVoiceSetup:
    """Shared setup with opaque, immutable, non-authoritative provider options.

    Providers may use ``provider_options`` to configure their own transport,
    but hosts must never treat those values as identity or tool authority.
    """

    model: str | None = None
    voice: str | None = None
    instructions: str | None = None
    tools: tuple[RealtimeTool, ...] = ()
    audio: RealtimeAudioFormat | None = None
    provider_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model is not None:
            _validate_identifier(self.model, "model")
        if self.voice is not None:
            _validate_identifier(self.voice, "voice")
        if self.instructions is not None and not isinstance(self.instructions, str):
            raise TypeError("instructions must be None or str")
        if self.audio is not None and not isinstance(self.audio, RealtimeAudioFormat):
            raise TypeError("audio must be None or RealtimeAudioFormat")
        provider_options = _freeze_mapping(self.provider_options, "provider_options")
        shared_fields = {"audio", "instructions", "model", "tools", "voice"}
        shadowed = shared_fields.intersection(provider_options)
        if shadowed:
            raise ValueError(
                f"provider_options cannot shadow shared setup field: {min(shadowed)}"
            )
        if isinstance(self.tools, (str, bytes, bytearray, memoryview)):
            raise TypeError("tools must be an iterable of RealtimeTool instances")
        try:
            tools = tuple(self.tools)
        except TypeError as exc:
            raise TypeError(
                "tools must be an iterable of RealtimeTool instances"
            ) from exc
        if any(not isinstance(tool, RealtimeTool) for tool in tools):
            raise TypeError("tools must contain only RealtimeTool instances")
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "provider_options", provider_options)


class RealtimeVoiceSession(abc.ABC):
    """One provider-neutral bidirectional session with immutable capabilities."""

    _CAPABILITY_HOOKS = {
        RealtimeCapability.EXPLICIT_INTERRUPTION: "_interrupt",
        RealtimeCapability.OUTPUT_TRUNCATION: "_truncate_output",
        RealtimeCapability.INPUT_COMMIT_EVENTS: "_commit_audio",
        RealtimeCapability.TOOL_CALL_CANCELLATION: "_cancel_tool_call",
        RealtimeCapability.DYNAMIC_CONTEXT: "_update_context",
        RealtimeCapability.SESSION_RESUMPTION: "_resume_session",
        RealtimeCapability.CONTINUATION: "_continue_response",
    }

    def __init__(self, capabilities: Iterable[RealtimeCapability] = ()) -> None:
        self._capabilities = frozenset(capabilities)
        for capability, hook_name in self._CAPABILITY_HOOKS.items():
            if capability in self._capabilities and getattr(
                type(self), hook_name
            ) is getattr(RealtimeVoiceSession, hook_name):
                raise ValueError(
                    f"advertised capability {capability.value} requires a subclass "
                    f"override of {hook_name}"
                )
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._pending_continuation_batch_ids: OrderedDict[str, None] = OrderedDict()
        self._continuation_responses: dict[str, str] = {}
        self._completed_continuation_responses: OrderedDict[str, str] = OrderedDict()
        self._ordinary_response_ids: OrderedDict[str, None] = OrderedDict()

    @property
    def capabilities(self) -> frozenset[RealtimeCapability]:
        return self._capabilities

    async def __aenter__(self) -> "RealtimeVoiceSession":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    def _require(self, capability: RealtimeCapability) -> None:
        if capability not in self._capabilities:
            raise UnsupportedRealtimeCapability(capability)

    @abc.abstractmethod
    async def send_audio(self, audio: bytes, *, mime_type: str | None = None) -> None:
        """Send one immutable audio chunk."""

    async def submit_tool_results(
        self, batch_id: str, results: Sequence[RealtimeToolResult]
    ) -> None:
        """Submit one ordered result batch without requesting continuation."""
        _validate_identifier(batch_id, "batch_id")
        frozen_results = tuple(results)
        if not frozen_results:
            raise ValueError("tool result batch must contain at least one result")
        if any(result.batch_id != batch_id for result in frozen_results):
            raise ValueError("tool result batch_id must match every result batch_id")
        self._require(RealtimeCapability.TOOL_CALLING)
        await self._submit_tool_results(batch_id, frozen_results)

    @abc.abstractmethod
    async def _submit_tool_results(
        self, batch_id: str, results: tuple[RealtimeToolResult, ...]
    ) -> None:
        """Provider-specific ordered result batch submission."""

    def events(self) -> AsyncIterator[RealtimeVoiceEvent]:
        """Return the normalized stream with capability invariants enforced."""
        return self._validated_events()

    async def _validated_events(self) -> AsyncIterator[RealtimeVoiceEvent]:
        try:
            async for event in self._events():
                self._validate_continuation_event(event)
                yield event
        except BaseException:
            self._pending_continuation_batch_ids.clear()
            self._continuation_responses.clear()
            raise
        if self._pending_continuation_batch_ids or self._continuation_responses:
            raise ValueError(
                "event stream ended with unresolved continuation state "
                f"({len(self._pending_continuation_batch_ids)} pending, "
                f"{len(self._continuation_responses)} active)"
            )

    def _validate_continuation_event(self, event: RealtimeVoiceEvent) -> None:
        if isinstance(event, (SessionFailure, SessionClosed)):
            self._pending_continuation_batch_ids.clear()
            self._continuation_responses.clear()
            return
        if not isinstance(event, (ResponseStarted, ResponseCompleted)):
            return
        linked_batch = event.continuation_of_batch_id
        supports_continuation = RealtimeCapability.CONTINUATION in self._capabilities
        if not supports_continuation:
            if linked_batch is not None:
                raise ValueError(
                    "continuation response requires the continuation capability"
                )
            return

        if isinstance(event, ResponseStarted):
            known_batch = self._continuation_responses.get(event.response_id)
            if known_batch is None:
                known_batch = self._completed_continuation_responses.get(
                    event.response_id
                )
            if known_batch is not None:
                if linked_batch != known_batch:
                    raise ValueError("continuation response linkage changed")
                return
            if event.response_id in self._ordinary_response_ids:
                if linked_batch is not None:
                    raise ValueError("ordinary response linkage changed")
                self._remember_ordinary_response(event.response_id)
                return
            if linked_batch is None:
                self._remember_ordinary_response(event.response_id)
                return
            if linked_batch not in self._pending_continuation_batch_ids:
                raise ValueError("unsolicited continuation response linkage")
            self._pending_continuation_batch_ids.pop(linked_batch)
            self._continuation_responses[event.response_id] = linked_batch
            return

        if event.response_id in self._ordinary_response_ids:
            if linked_batch is not None:
                raise ValueError("ordinary response completion linkage changed")
            self._remember_ordinary_response(event.response_id)
            return

        expected_batch = self._continuation_responses.get(event.response_id)
        if expected_batch is None:
            completed_batch = self._completed_continuation_responses.get(
                event.response_id
            )
            if completed_batch is not None:
                if linked_batch != completed_batch:
                    raise ValueError("continuation completion linkage changed")
            elif linked_batch is not None:
                raise ValueError("unsolicited continuation completion linkage")
            else:
                self._remember_ordinary_response(event.response_id)
            return
        if linked_batch != expected_batch:
            raise ValueError(
                "continuation completion must link requested batch "
                f"{expected_batch}"
            )
        del self._continuation_responses[event.response_id]
        self._completed_continuation_responses[event.response_id] = expected_batch
        self._completed_continuation_responses.move_to_end(event.response_id)
        if (
            len(self._completed_continuation_responses)
            > MAX_TRACKED_CONTINUATION_RESPONSES
        ):
            self._completed_continuation_responses.popitem(last=False)

    def _remember_ordinary_response(self, response_id: str) -> None:
        self._ordinary_response_ids[response_id] = None
        self._ordinary_response_ids.move_to_end(response_id)
        if len(self._ordinary_response_ids) > MAX_TRACKED_ORDINARY_RESPONSES:
            self._ordinary_response_ids.popitem(last=False)

    @abc.abstractmethod
    def _events(self) -> AsyncIterator[RealtimeVoiceEvent]:
        """Return provider events translated to normalized typed events."""

    async def commit_audio(self) -> None:
        self._require(RealtimeCapability.INPUT_COMMIT_EVENTS)
        await self._commit_audio()

    async def _commit_audio(self) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.INPUT_COMMIT_EVENTS)

    async def interrupt(self) -> None:
        self._require(RealtimeCapability.EXPLICIT_INTERRUPTION)
        await self._interrupt()

    async def _interrupt(self) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.EXPLICIT_INTERRUPTION)

    async def truncate_output(self, response_id: str, item_id: str) -> None:
        _validate_identifier(response_id, "response_id")
        _validate_identifier(item_id, "item_id")
        self._require(RealtimeCapability.OUTPUT_TRUNCATION)
        await self._truncate_output(response_id, item_id)

    async def _truncate_output(self, response_id: str, item_id: str) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.OUTPUT_TRUNCATION)

    async def cancel_tool_call(self, call_id: str, batch_id: str) -> None:
        _validate_identifier(call_id, "call_id")
        _validate_identifier(batch_id, "batch_id")
        self._require(RealtimeCapability.TOOL_CALL_CANCELLATION)
        await self._cancel_tool_call(call_id, batch_id)

    async def _cancel_tool_call(self, call_id: str, batch_id: str) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.TOOL_CALL_CANCELLATION)

    async def resume_session(self, session_id: str) -> None:
        _validate_identifier(session_id, "session_id")
        self._require(RealtimeCapability.SESSION_RESUMPTION)
        await self._resume_session(session_id)

    async def _resume_session(self, session_id: str) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.SESSION_RESUMPTION)

    async def update_context(
        self, instructions: str, tools: Sequence[RealtimeTool] = ()
    ) -> None:
        self._require(RealtimeCapability.DYNAMIC_CONTEXT)
        await self._update_context(instructions, tuple(tools))

    async def _update_context(
        self, instructions: str, tools: tuple[RealtimeTool, ...]
    ) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.DYNAMIC_CONTEXT)

    async def continue_response(self, batch_id: str) -> None:
        _validate_identifier(batch_id, "batch_id")
        self._require(RealtimeCapability.CONTINUATION)
        if batch_id in self._pending_continuation_batch_ids or batch_id in (
            self._continuation_responses.values()
        ) or batch_id in self._completed_continuation_responses.values():
            raise ValueError(f"continuation batch is already pending: {batch_id}")
        if (
            len(self._pending_continuation_batch_ids)
            + len(self._continuation_responses)
            >= MAX_OUTSTANDING_CONTINUATIONS
        ):
            raise ValueError("outstanding continuation limit reached")
        self._pending_continuation_batch_ids[batch_id] = None
        try:
            await self._continue_response(batch_id)
        except BaseException:
            self._pending_continuation_batch_ids.pop(batch_id, None)
            failed_response_ids = [
                response_id
                for response_id, active_batch_id in self._continuation_responses.items()
                if active_batch_id == batch_id
            ]
            for response_id in failed_response_ids:
                del self._continuation_responses[response_id]
            raise

    async def _continue_response(self, batch_id: str) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.CONTINUATION)

    def _finalize_close_task(self, task: asyncio.Task[None]) -> None:
        """Record one cleanup outcome even when its waiting caller is cancelled."""
        if task.cancelled():
            if self._close_task is task:
                self._close_task = None
            return
        try:
            task.result()
        except BaseException:
            # Retrieving the exception prevents an orphaned-task warning when
            # the sole waiter was cancelled. A later close may retry cleanup.
            if self._close_task is task:
                self._close_task = None
        else:
            if self._close_task is task:
                self._closed = True
                self._close_task = None

    async def close(self) -> None:
        """Release resources once; repeated calls are successful no-ops."""
        async with self._close_lock:
            if self._closed:
                return
            task = self._close_task
            if task is None:
                task = asyncio.create_task(self._close())
                self._close_task = task
                task.add_done_callback(self._finalize_close_task)

        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except BaseException:
            async with self._close_lock:
                if self._close_task is task:
                    self._close_task = None
            raise

        async with self._close_lock:
            if self._close_task is task:
                self._closed = True
                self._close_task = None

    @abc.abstractmethod
    async def _close(self) -> None:
        """Provider-specific release, retried only after failed cleanup."""


class RealtimeVoiceProvider(abc.ABC):
    """Abstract factory and immutable metadata surface for realtime sessions."""

    api_version: int = REALTIME_VOICE_PROVIDER_API_VERSION
    capabilities: frozenset[RealtimeCapability] = frozenset()

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier used by the realtime voice registry."""

    @property
    def display_name(self) -> str:
        return self.name.title()

    def is_available(self) -> bool:
        """Check readiness without mutating configuration or environment."""
        return True

    def list_models(self) -> tuple[Mapping[str, Any], ...]:
        return ()

    def list_voices(self) -> tuple[Mapping[str, Any], ...]:
        return ()

    def default_model(self) -> str | None:
        models = self.list_models()
        if not models:
            return None
        model_id = models[0].get("id")
        if model_id is not None:
            _validate_identifier(model_id, "model id")
        return model_id

    def default_voice(self) -> str | None:
        voices = self.list_voices()
        if not voices:
            return None
        voice_id = voices[0].get("id")
        if voice_id is not None:
            _validate_identifier(voice_id, "voice id")
        return voice_id

    def get_setup_schema(self) -> Mapping[str, Any]:
        """Return immutable, non-mutating picker/setup metadata."""
        return _freeze(
            {"name": self.display_name, "badge": "", "tag": "", "env_vars": ()}
        )

    @abc.abstractmethod
    async def open_session(self, setup: RealtimeVoiceSetup) -> RealtimeVoiceSession:
        """Open a connected session from the complete shared typed setup."""

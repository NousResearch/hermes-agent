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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import StrEnum
from types import MappingProxyType
from typing import Any, AsyncIterator, Iterable

REALTIME_VOICE_PROVIDER_API_VERSION = 1
MAX_IDENTIFIER_LENGTH = 512


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
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_provider_data(event: Any) -> None:
    shared_fields = {item.name for item in fields(event)} - {"provider_data"}
    shadowed = shared_fields.intersection(event.provider_data)
    if shadowed:
        raise ValueError(
            f"provider_data cannot shadow shared event field: {min(shadowed)}"
        )
    object.__setattr__(event, "provider_data", _freeze(event.provider_data))


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
        object.__setattr__(self, "arguments", _freeze(self.arguments))
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


@dataclass(frozen=True, slots=True)
class RealtimeTool:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "name")
        object.__setattr__(self, "parameters", _freeze(self.parameters))


@dataclass(frozen=True, slots=True)
class RealtimeVoiceSetup:
    """Shared session setup; provider-specific options remain opaque."""

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
        shared_fields = {"audio", "instructions", "model", "tools", "voice"}
        shadowed = shared_fields.intersection(self.provider_options)
        if shadowed:
            raise ValueError(
                f"provider_options cannot shadow shared setup field: {min(shadowed)}"
            )
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "provider_options", _freeze(self.provider_options))


class RealtimeVoiceSession(abc.ABC):
    """One provider-neutral bidirectional session with immutable capabilities."""

    def __init__(self, capabilities: Iterable[RealtimeCapability] = ()) -> None:
        self._capabilities = frozenset(capabilities)
        self._closed = False

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

    @abc.abstractmethod
    def events(self) -> AsyncIterator[RealtimeVoiceEvent]:
        """Return the normalized typed event stream."""

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
        await self._continue_response(batch_id)

    async def _continue_response(self, batch_id: str) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.CONTINUATION)

    async def close(self) -> None:
        """Release resources once; repeated calls are successful no-ops."""
        if self._closed:
            return
        self._closed = True
        await self._close()

    @abc.abstractmethod
    async def _close(self) -> None:
        """Provider-specific release, called at most once."""


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

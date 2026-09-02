"""
Realtime Voice Provider ABC
===========================

Defines the provider-neutral contract for low-latency, bidirectional voice
sessions (RFC #77111). Realtime voice is intentionally separate from
:mod:`agent.transports`: model transports execute one request/response turn,
while a realtime session is a long-lived async channel carrying audio, text,
tool calls, interruptions, and lifecycle events in both directions.

Ownership split
---------------
The **provider** owns its SDK, credentials, wire protocol, audio codecs,
turn detection, and transport lifecycle. It translates native wire events
into the frozen typed events below and native commands out of the typed
session methods.

**Hermes** (the host, via :mod:`agent.realtime_voice_orchestrator`) owns tool
authorization and execution, approvals, conversation history, and memory. A
provider never executes a tool: it emits :class:`ToolCall` and later receives
:class:`RealtimeToolResult` values through :meth:`RealtimeVoiceSession.submit_tool_results`.

Event contract
--------------
:meth:`RealtimeVoiceSession.events` yields :class:`RealtimeVoiceEvent`
instances until a terminal event (:class:`SessionClosed`, or a
:class:`SessionFailure` with ``terminal=True``) closes the stream.

``SessionReady`` / ``SessionClosed`` / ``SessionFailure`` / ``SessionResumptionUpdate``
    Session lifecycle. A non-terminal ``SessionFailure`` is a wire-level
    complaint the host should surface but the call survives (one malformed
    frame must never end a conversation).
``InputSpeechStarted`` / ``InputSpeechStopped`` / ``InputAudioCommitted``
    Server-side turn detection on the operator's microphone. ``InputSpeechStarted``
    is the barge-in signal.
``InputTranscript`` / ``OutputTranscript``
    Operator speech and assistant speech as text (partial and final).
``OutputAudio``
    Assistant audio in the session's negotiated ``output_audio_format``.
``ResponseStarted`` / ``ResponseCompleted``
    One assistant response. Every audio/transcript/tool event between them
    carries the same ``response_id`` when the provider names responses.
``ToolCall`` / ``ToolCallCancelled``
    Function calls the host must answer exactly once per ``call_id``, and the
    provider-side retraction of calls it no longer wants answered.

Every event carries an immutable, diagnostic-only ``provider_data`` mapping
for provider-native fields. Hosts must never treat it as authority.

Capabilities
------------
A session advertises a frozen set of :class:`RealtimeCapability`. Optional
operations (``commit_audio``, ``create_response``, ``cancel_response``,
``truncate_output``, ``add_context`` / ``remove_context``,
``submit_tool_results``) raise :class:`UnsupportedRealtimeCapability` unless
the matching capability is advertised, and advertising a capability without
overriding its hook is rejected at construction. Hosts degrade explicitly:
a provider without ``OUTPUT_TRUNCATION`` gets its playback dropped locally
instead of a faked truncation.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import StrEnum
from types import MappingProxyType
from typing import Any, AsyncIterator, Iterable

REALTIME_VOICE_PROVIDER_API_VERSION = 2
MAX_IDENTIFIER_LENGTH = 512


class RealtimeCapability(StrEnum):
    TOOL_CALLING = "tool_calling"
    INPUT_TRANSCRIPTION = "input_transcription"
    OUTPUT_TRANSCRIPTION = "output_transcription"
    INPUT_COMMIT_EVENTS = "input_commit_events"
    EXPLICIT_RESPONSE = "explicit_response"
    RESPONSE_CANCELLATION = "response_cancellation"
    OUTPUT_TRUNCATION = "output_truncation"
    TOOL_CALL_CANCELLATION = "tool_call_cancellation"
    DYNAMIC_CONTEXT = "dynamic_context"
    SESSION_RESUMPTION = "session_resumption"


class UnsupportedRealtimeCapability(RuntimeError):
    """Raised when a session cannot provide requested optional semantics."""

    def __init__(self, capability: RealtimeCapability) -> None:
        self.capability = capability
        super().__init__(f"realtime capability is unsupported: {capability.value}")


def _validate_identifier(value: Any, field_name: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
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


def _validate_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be str")


def _validate_optional_ms(value: Any, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be None or a non-negative integer")
    if value < 0:
        raise ValueError(f"{field_name} must be None or a non-negative integer")


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


# -- lifecycle ---------------------------------------------------------------


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
        _validate_text(self.reason, "reason")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class SessionFailure(RealtimeVoiceEvent):
    """A provider-reported problem. ``terminal`` failures end the stream."""

    code: str
    message: str
    terminal: bool = True
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.code, "code")
        _validate_text(self.message, "message")
        if not isinstance(self.terminal, bool):
            raise TypeError("terminal must be bool")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class SessionResumptionUpdate(RealtimeVoiceEvent):
    """The provider issued (or invalidated) a resumption handle.

    ``resumable=False`` retracts any earlier handle; reusing it would be
    silent data loss. Hosts persist the latest confirmed handle only.
    """

    handle: str | None
    resumable: bool
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.handle, "handle", optional=True)
        if not isinstance(self.resumable, bool):
            raise TypeError("resumable must be bool")
        _freeze_provider_data(self)


# -- operator input ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InputSpeechStarted(RealtimeVoiceEvent):
    """Server-side turn detection heard the operator start speaking (barge-in)."""

    item_id: str | None = None
    audio_start_ms: int | None = None
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.item_id, "item_id", optional=True)
        _validate_optional_ms(self.audio_start_ms, "audio_start_ms")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class InputSpeechStopped(RealtimeVoiceEvent):
    item_id: str | None = None
    audio_end_ms: int | None = None
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.item_id, "item_id", optional=True)
        _validate_optional_ms(self.audio_end_ms, "audio_end_ms")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class InputAudioCommitted(RealtimeVoiceEvent):
    item_id: str | None = None
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.item_id, "item_id", optional=True)
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class InputTranscript(RealtimeVoiceEvent):
    """Operator speech as text. Belongs to no response."""

    text: str
    final: bool
    item_id: str | None = None
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_text(self.text, "text")
        if not isinstance(self.final, bool):
            raise TypeError("final must be bool")
        _validate_identifier(self.item_id, "item_id", optional=True)
        _freeze_provider_data(self)


# -- assistant output --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResponseStarted(RealtimeVoiceEvent):
    response_id: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.response_id, "response_id", optional=True)
        metadata = _freeze_mapping(self.metadata, "metadata")
        if any(not isinstance(value, str) for value in metadata.values()):
            raise TypeError("metadata values must be str")
        object.__setattr__(self, "metadata", metadata)
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class ResponseCompleted(RealtimeVoiceEvent):
    """The provider finished (or cancelled) one response."""

    response_id: str | None = None
    status: str = ""
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.response_id, "response_id", optional=True)
        _validate_text(self.status, "status")
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class OutputAudio(RealtimeVoiceEvent):
    """Assistant audio bytes in the session's ``output_audio_format``."""

    data: bytes
    item_id: str | None = None
    response_id: str | None = None
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, (bytes, bytearray, memoryview)):
            raise TypeError("output audio data must be bytes-like")
        object.__setattr__(self, "data", bytes(self.data))
        _validate_identifier(self.item_id, "item_id", optional=True)
        _validate_identifier(self.response_id, "response_id", optional=True)
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class OutputTranscript(RealtimeVoiceEvent):
    """Assistant speech as text. ``final`` carries the complete transcript."""

    text: str
    final: bool
    item_id: str | None = None
    response_id: str | None = None
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_text(self.text, "text")
        if not isinstance(self.final, bool):
            raise TypeError("final must be bool")
        _validate_identifier(self.item_id, "item_id", optional=True)
        _validate_identifier(self.response_id, "response_id", optional=True)
        _freeze_provider_data(self)


# -- tools -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolCall(RealtimeVoiceEvent):
    """One function call the host must answer exactly once.

    ``arguments`` is the JSON object text as the provider delivered it. The
    host parses it and answers a malformed payload with an error *result*
    so the turn completes instead of hanging on an unanswered call.
    """

    call_id: str
    name: str
    arguments: str
    response_id: str | None = None
    item_id: str | None = None
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.call_id, "call_id")
        _validate_identifier(self.name, "name")
        _validate_text(self.arguments, "arguments")
        _validate_identifier(self.response_id, "response_id", optional=True)
        _validate_identifier(self.item_id, "item_id", optional=True)
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class ToolCallCancelled(RealtimeVoiceEvent):
    """The provider retracted calls; their results must never be submitted."""

    call_ids: tuple[str, ...]
    provider_data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.call_ids, (str, bytes)):
            raise TypeError("call_ids must be a sequence of identifiers")
        call_ids = tuple(self.call_ids)
        if not call_ids:
            raise ValueError("call_ids must contain at least one identifier")
        for call_id in call_ids:
            _validate_identifier(call_id, "call_id")
        object.__setattr__(self, "call_ids", call_ids)
        _freeze_provider_data(self)


@dataclass(frozen=True, slots=True)
class RealtimeToolResult:
    """One host-owned result bound to the provider's ``call_id``."""

    call_id: str
    output: str

    def __post_init__(self) -> None:
        _validate_identifier(self.call_id, "call_id")
        _validate_text(self.output, "output")


# -- setup -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RealtimeAudioFormat:
    """Raw PCM description. Samples are always 16-bit little-endian signed."""

    mime_type: str = "audio/pcm"
    sample_rate_hz: int = 24_000
    channels: int = 1

    def __post_init__(self) -> None:
        _validate_identifier(self.mime_type, "mime_type")
        for field_name in ("sample_rate_hz", "channels"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be a positive integer")
            if value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate_hz * self.channels * 2


PCM16_24K = RealtimeAudioFormat()


@dataclass(frozen=True, slots=True)
class RealtimeTool:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "name")
        _validate_text(self.description, "description")
        object.__setattr__(
            self, "parameters", _freeze_mapping(self.parameters, "parameters")
        )


@dataclass(frozen=True, slots=True)
class RealtimeVoiceSetup:
    """Shared setup with opaque, immutable, non-authoritative provider options.

    ``input_audio`` / ``output_audio`` of ``None`` ask for the provider's
    native format; the opened session reports what was actually negotiated.
    ``automatic_response`` lets server-side turn detection start a response
    on its own; hosts that gate every response set it ``False`` and call
    :meth:`RealtimeVoiceSession.create_response` themselves.
    """

    model: str | None = None
    voice: str | None = None
    instructions: str = ""
    tools: tuple[RealtimeTool, ...] = ()
    input_audio: RealtimeAudioFormat | None = None
    output_audio: RealtimeAudioFormat | None = None
    automatic_response: bool = True
    provider_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identifier(self.model, "model", optional=True)
        _validate_identifier(self.voice, "voice", optional=True)
        _validate_text(self.instructions, "instructions")
        for field_name in ("input_audio", "output_audio"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, RealtimeAudioFormat):
                raise TypeError(f"{field_name} must be None or RealtimeAudioFormat")
        if not isinstance(self.automatic_response, bool):
            raise TypeError("automatic_response must be bool")
        provider_options = _freeze_mapping(self.provider_options, "provider_options")
        shared_fields = {item.name for item in fields(self)} - {"provider_options"}
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


# -- session -----------------------------------------------------------------


class RealtimeVoiceSession(abc.ABC):
    """One provider-neutral bidirectional session with immutable capabilities.

    Subclasses implement :meth:`send_audio`, :meth:`_events`, and
    :meth:`_close`, plus the ``_``-prefixed hook for every capability they
    advertise. The public methods own validation and capability gating so a
    host can rely on them regardless of provider.
    """

    _CAPABILITY_HOOKS: Mapping[RealtimeCapability, tuple[str, ...]] = {
        RealtimeCapability.TOOL_CALLING: ("_submit_tool_results",),
        RealtimeCapability.INPUT_COMMIT_EVENTS: ("_commit_audio",),
        RealtimeCapability.EXPLICIT_RESPONSE: ("_create_response",),
        RealtimeCapability.RESPONSE_CANCELLATION: ("_cancel_response",),
        RealtimeCapability.OUTPUT_TRUNCATION: ("_truncate_output",),
        RealtimeCapability.DYNAMIC_CONTEXT: ("_add_context", "_remove_context"),
    }

    def __init__(
        self,
        capabilities: Iterable[RealtimeCapability] = (),
        *,
        input_audio: RealtimeAudioFormat = PCM16_24K,
        output_audio: RealtimeAudioFormat = PCM16_24K,
    ) -> None:
        self._capabilities = frozenset(capabilities)
        for capability, hook_names in self._CAPABILITY_HOOKS.items():
            if capability not in self._capabilities:
                continue
            for hook_name in hook_names:
                if getattr(type(self), hook_name) is getattr(
                    RealtimeVoiceSession, hook_name
                ):
                    raise ValueError(
                        f"advertised capability {capability.value} requires a "
                        f"subclass override of {hook_name}"
                    )
        for field_name, value in (("input_audio", input_audio), ("output_audio", output_audio)):
            if not isinstance(value, RealtimeAudioFormat):
                raise TypeError(f"{field_name} must be RealtimeAudioFormat")
        self._input_audio = input_audio
        self._output_audio = output_audio
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None

    @property
    def capabilities(self) -> frozenset[RealtimeCapability]:
        return self._capabilities

    @property
    def input_audio_format(self) -> RealtimeAudioFormat:
        """The PCM format :meth:`send_audio` expects."""
        return self._input_audio

    @property
    def output_audio_format(self) -> RealtimeAudioFormat:
        """The PCM format :class:`OutputAudio` carries."""
        return self._output_audio

    @property
    def closed(self) -> bool:
        return self._closed

    def supports(self, capability: RealtimeCapability) -> bool:
        return capability in self._capabilities

    async def __aenter__(self) -> "RealtimeVoiceSession":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    def _require(self, capability: RealtimeCapability) -> None:
        if capability not in self._capabilities:
            raise UnsupportedRealtimeCapability(capability)

    # -- required ----------------------------------------------------------

    @abc.abstractmethod
    async def send_audio(self, audio: bytes) -> None:
        """Send one chunk of operator audio in :attr:`input_audio_format`."""

    @abc.abstractmethod
    def _events(self) -> AsyncIterator[RealtimeVoiceEvent]:
        """Return provider events translated to normalized typed events."""

    @abc.abstractmethod
    async def _close(self) -> None:
        """Provider-specific release, retried only after failed cleanup."""

    # -- event stream ------------------------------------------------------

    def events(self) -> AsyncIterator[RealtimeVoiceEvent]:
        """Return the normalized stream; ends after the first terminal event."""
        return self._validated_events()

    async def _validated_events(self) -> AsyncIterator[RealtimeVoiceEvent]:
        events = self._events()
        try:
            async for event in events:
                if not isinstance(event, RealtimeVoiceEvent):
                    raise TypeError(
                        "provider event stream must yield RealtimeVoiceEvent instances"
                    )
                yield event
                if isinstance(event, SessionClosed) or (
                    isinstance(event, SessionFailure) and event.terminal
                ):
                    return
        finally:
            close_events = getattr(events, "aclose", None)
            if close_events is not None:
                await close_events()

    # -- tools -------------------------------------------------------------

    async def submit_tool_results(
        self,
        results: Sequence[RealtimeToolResult],
        *,
        continue_response: bool = True,
    ) -> None:
        """Deliver one ordered batch of results, optionally requesting a reply.

        Providers whose protocol answers tool results on its own ignore
        ``continue_response``; the others start exactly one response after
        the batch when it is ``True``.
        """
        frozen_results = tuple(results)
        if not frozen_results:
            raise ValueError("tool result batch must contain at least one result")
        if any(not isinstance(result, RealtimeToolResult) for result in frozen_results):
            raise TypeError("tool results must be RealtimeToolResult instances")
        if len({result.call_id for result in frozen_results}) != len(frozen_results):
            raise ValueError("tool result batch cannot repeat a call_id")
        self._require(RealtimeCapability.TOOL_CALLING)
        await self._submit_tool_results(frozen_results, continue_response)

    async def _submit_tool_results(
        self, results: tuple[RealtimeToolResult, ...], continue_response: bool
    ) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.TOOL_CALLING)

    # -- optional operations ----------------------------------------------

    async def commit_audio(self) -> None:
        """Close the operator's input buffer when turn detection is manual."""
        self._require(RealtimeCapability.INPUT_COMMIT_EVENTS)
        await self._commit_audio()

    async def _commit_audio(self) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.INPUT_COMMIT_EVENTS)

    async def create_response(self, *, metadata: Mapping[str, str] | None = None) -> None:
        """Ask the provider to start a response now."""
        frozen = _freeze_mapping(metadata or {}, "metadata")
        if any(not isinstance(value, str) for value in frozen.values()):
            raise TypeError("metadata values must be str")
        self._require(RealtimeCapability.EXPLICIT_RESPONSE)
        await self._create_response(frozen)

    async def _create_response(self, metadata: Mapping[str, str]) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.EXPLICIT_RESPONSE)

    async def cancel_response(self, response_id: str | None = None) -> None:
        """Stop the in-flight response (the named one when the wire supports it)."""
        _validate_identifier(response_id, "response_id", optional=True)
        self._require(RealtimeCapability.RESPONSE_CANCELLATION)
        await self._cancel_response(response_id)

    async def _cancel_response(self, response_id: str | None) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.RESPONSE_CANCELLATION)

    async def truncate_output(self, item_id: str, audio_end_ms: int) -> None:
        """Tell the provider how much of ``item_id`` the operator actually heard."""
        _validate_identifier(item_id, "item_id")
        if isinstance(audio_end_ms, bool) or not isinstance(audio_end_ms, int):
            raise TypeError("audio_end_ms must be a non-negative integer")
        if audio_end_ms < 0:
            raise ValueError("audio_end_ms must be a non-negative integer")
        self._require(RealtimeCapability.OUTPUT_TRUNCATION)
        await self._truncate_output(item_id, audio_end_ms)

    async def _truncate_output(self, item_id: str, audio_end_ms: int) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.OUTPUT_TRUNCATION)

    async def add_context(self, item_id: str, text: str) -> None:
        """Insert host-authored text into the conversation without a response."""
        _validate_identifier(item_id, "item_id")
        _validate_text(text, "text")
        self._require(RealtimeCapability.DYNAMIC_CONTEXT)
        await self._add_context(item_id, text)

    async def _add_context(self, item_id: str, text: str) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.DYNAMIC_CONTEXT)

    async def remove_context(self, item_id: str) -> None:
        _validate_identifier(item_id, "item_id")
        self._require(RealtimeCapability.DYNAMIC_CONTEXT)
        await self._remove_context(item_id)

    async def _remove_context(self, item_id: str) -> None:
        raise UnsupportedRealtimeCapability(RealtimeCapability.DYNAMIC_CONTEXT)

    # -- close -------------------------------------------------------------

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


# -- provider ----------------------------------------------------------------


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


__all__ = [
    "MAX_IDENTIFIER_LENGTH",
    "PCM16_24K",
    "REALTIME_VOICE_PROVIDER_API_VERSION",
    "InputAudioCommitted",
    "InputSpeechStarted",
    "InputSpeechStopped",
    "InputTranscript",
    "OutputAudio",
    "OutputTranscript",
    "RealtimeAudioFormat",
    "RealtimeCapability",
    "RealtimeTool",
    "RealtimeToolResult",
    "RealtimeVoiceEvent",
    "RealtimeVoiceProvider",
    "RealtimeVoiceSession",
    "RealtimeVoiceSetup",
    "ResponseCompleted",
    "ResponseStarted",
    "SessionClosed",
    "SessionFailure",
    "SessionReady",
    "SessionResumptionUpdate",
    "ToolCall",
    "ToolCallCancelled",
    "UnsupportedRealtimeCapability",
]

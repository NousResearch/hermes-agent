"""
Realtime Voice Provider ABC
===========================

Defines the provider-neutral contract for low-latency, bidirectional voice
sessions. Realtime voice is intentionally separate from
:mod:`agent.transports`: model transports execute one request/response turn,
while a realtime session is a long-lived async channel carrying audio, text,
tool calls, interruptions, and lifecycle events.

Providers own their wire protocol and SDK. OpenAI Realtime, Gemini Live, and
future backends translate native events into the small event envelope below;
provider-specific state stays under ``provider_data`` instead of leaking into
Hermes callers.

Event contract
--------------
:meth:`RealtimeVoiceSession.events` yields dictionaries with a required
``type`` string. Shared event names are:

``session.started`` / ``session.updated`` / ``session.closed``
    Session lifecycle.
``input.transcript.delta`` / ``input.transcript.done``
    User speech transcription in ``text``.
``output.text.delta`` / ``output.text.done``
    Assistant text in ``text``.
``output.audio.delta`` / ``output.audio.done``
    Assistant audio bytes in ``audio``.
``tool.call``
    Tool request in ``tool_call`` with ``id``, ``name``, and ``arguments``.
``turn.done`` / ``interrupted`` / ``error``
    Turn completion, barge-in, and failure signals.

Unknown provider-native fields belong in ``provider_data``. New normalized
event names may be added without changing the provider API version.
"""

from __future__ import annotations

import abc
from typing import Any, AsyncIterator, Dict, List, Optional

REALTIME_VOICE_PROVIDER_API_VERSION = 1

RealtimeVoiceEvent = Dict[str, Any]


class RealtimeVoiceSession(abc.ABC):
    """One connected bidirectional voice session.

    Implementations may use WebSocket, WebRTC, an SDK-managed connection, or
    another transport. Callers interact only through normalized methods and
    events, so transport changes remain provider-local.
    """

    async def __aenter__(self) -> "RealtimeVoiceSession":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    @abc.abstractmethod
    async def send_audio(
        self,
        audio: bytes,
        *,
        mime_type: Optional[str] = None,
    ) -> None:
        """Send one audio chunk.

        ``mime_type`` describes the chunk when the provider needs an explicit
        format (for example ``audio/pcm;rate=16000``). Providers with a
        session-level or negotiated format may ignore it.
        """

    @abc.abstractmethod
    async def send_text(self, text: str, *, end_of_turn: bool = True) -> None:
        """Send text input, optionally marking the end of the user turn."""

    async def commit_audio(self) -> None:
        """Commit buffered audio when the provider requires it.

        Default is a no-op for server-VAD and continuously streamed sessions.
        """

    @abc.abstractmethod
    async def send_tool_result(
        self,
        call_id: str,
        output: Any,
        *,
        name: Optional[str] = None,
    ) -> None:
        """Return a tool result to the active realtime conversation."""

    async def interrupt(self) -> None:
        """Cancel current assistant output when explicit cancellation exists.

        Providers that only support automatic voice-activity interruption may
        keep the default and let incoming audio trigger barge-in.
        """
        raise NotImplementedError(
            f"Realtime voice session {type(self).__name__!r} does not support "
            "explicit interruption"
        )

    @abc.abstractmethod
    def events(self) -> AsyncIterator[RealtimeVoiceEvent]:
        """Return the normalized async event stream for this session."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the session and release all transport resources."""


class RealtimeVoiceProvider(abc.ABC):
    """Abstract factory and metadata surface for realtime voice sessions."""

    api_version: int = REALTIME_VOICE_PROVIDER_API_VERSION

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier used by the realtime voice registry."""

    @property
    def display_name(self) -> str:
        """Human-readable provider label."""
        return self.name.title()

    def is_available(self) -> bool:
        """Return whether credentials and optional dependencies are ready.

        Must not raise; setup and picker surfaces call this for diagnostics.
        """
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        """Return model catalog entries; each entry requires an ``id``."""
        return []

    def list_voices(self) -> List[Dict[str, Any]]:
        """Return voice catalog entries; each entry requires an ``id``."""
        return []

    def default_model(self) -> Optional[str]:
        """Return the default model id, or ``None`` when provider-defined."""
        models = self.list_models()
        if models:
            return models[0].get("id")
        return None

    def default_voice(self) -> Optional[str]:
        """Return the default voice id, or ``None`` when provider-defined."""
        voices = self.list_voices()
        if voices:
            return voices[0].get("id")
        return None

    def get_capabilities(self) -> Dict[str, Any]:
        """Return stable feature metadata for host capability negotiation.

        Providers override only supported features. ``transports`` describes
        provider-owned wire transports, not Hermes model transports.
        """
        return {
            "input_modalities": [],
            "output_modalities": [],
            "transports": [],
            "tool_calling": False,
            "input_transcription": False,
            "output_transcription": False,
            "interruption": False,
            "session_resumption": False,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        """Return provider metadata for a future setup/picker surface."""
        return {
            "name": self.display_name,
            "badge": "",
            "tag": "",
            "env_vars": [],
        }

    @abc.abstractmethod
    async def open_session(
        self,
        *,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **extra: Any,
    ) -> RealtimeVoiceSession:
        """Open and return a connected realtime voice session.

        Providers translate host tool definitions and session options into
        their native protocol. Provider-specific options travel through
        ``extra`` so adding a vendor feature does not widen the shared ABC.
        """

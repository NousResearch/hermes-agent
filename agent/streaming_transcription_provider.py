"""Streaming transcription provider protocol (partial-transcript STT).

Complements :class:`agent.transcription_provider.TranscriptionProvider` (one-shot file
transcription) with a session-based streaming interface: ``start``/``feed``/``finish``/``cancel``.
``feed`` returns the current partial transcript so a client can render incremental text before
the final result. Plugins register implementations through the default scoped provider channel
(``register_streaming_transcription_provider``); the live WebSocket endpoint
``/api/audio/transcribe-stream`` resolves the active provider by ``stt.provider``.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

from agent.provider_base import ProviderBase


class StreamingTranscriptionProvider(ProviderBase):
    """Streaming counterpart of :class:`~agent.transcription_provider.TranscriptionProvider`.

    A provider may accept one or many concurrent sessions; each ``start`` returns an opaque
    stream key used by the subsequent calls. Implementations own model lifecycle (including
    idle unload) internally.
    """

    @abc.abstractmethod
    def start(self, *, language: Optional[str] = None,
              config: Optional[Dict[str, Any]] = None) -> str:
        """Begin a streaming session; returns the stream key for ``feed``/``finish``/``cancel``.

        ``config`` carries the provider's own ``stt.<provider.name>`` section (already looked
        up by the caller), so a provider never hard-codes a config key.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def feed(self, stream_key: str, pcm: bytes) -> str:
        """Feed 16 kHz mono int16 PCM; return the current partial transcript text.

        May raise ``ValueError`` for invalid audio or stale/unknown keys.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def finish(self, stream_key: str) -> str:
        """Finalize the stream and return the full transcript text."""
        raise NotImplementedError

    @abc.abstractmethod
    def cancel(self, stream_key: str) -> None:
        """Abort and release the stream; must be idempotent and safe on unknown keys."""
        raise NotImplementedError
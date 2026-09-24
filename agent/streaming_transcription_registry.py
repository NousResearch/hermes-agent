"""Streaming Transcription Provider Registry.

Central map of partial-transcript STT providers, populated by plugins via
:meth:`PluginContext.register_streaming_transcription_provider` and consumed by
:mod:`tools.transcription_stream` (the live ``/api/audio/transcribe-stream`` WebSocket) to
resolve the active streaming backend by ``stt.provider``. There are no built-in streaming
providers today; ``stt_streaming`` capability reveals presence of a registered match.
"""

from __future__ import annotations

import logging

from agent.provider_registry import ProviderRegistry, lower_key
from agent.streaming_transcription_provider import StreamingTranscriptionProvider

logger = logging.getLogger(__name__)


def _warn_builtin_collision(key: str) -> None:
    logger.warning(
        "Streaming transcription provider '%s' shadows a built-in name; registration ignored.",
        key,
    )


_registry: ProviderRegistry[StreamingTranscriptionProvider] = ProviderRegistry(
    label="StreamingTranscription",
    provider_cls=StreamingTranscriptionProvider,
    logger=logger,
    normalize=lower_key,
    builtin_names=frozenset(),
    on_builtin_collision=_warn_builtin_collision,
)
_registry.export(globals())
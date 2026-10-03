"""Transcription Provider ABC — pluggable speech-to-text backends.

Providers register via :meth:`PluginContext.register_transcription_provider`; the one named
by ``stt.provider`` services :func:`tools.transcription_tools.transcribe_audio` **when that
name is not a built-in** (built-ins always win; ``HERMES_LOCAL_STT_COMMAND`` stays on the
built-in ``local_command`` path). :meth:`TranscriptionProvider.transcribe` envelope:
``success`` bool, ``transcript`` str (empty on failure), ``provider`` str, ``error`` str
(only when success=False).
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

from agent.provider_base import CatalogProviderBase


class TranscriptionProvider(CatalogProviderBase):
    """Abstract base class for a speech-to-text backend.

    Subclasses must implement :attr:`name` (rejected at registration if it
    collides with a built-in STT name) and :meth:`transcribe`.
    """

    @abc.abstractmethod
    def transcribe(
        self, file_path: str, *, model: Optional[str] = None, language: Optional[str] = None, **extra: Any,
    ) -> Dict[str, Any]:
        """Transcribe ``file_path`` (existence + size already validated) into the module envelope.

        Must NOT raise — convert exceptions to the error envelope. ``model`` None →
        :meth:`default_model`; ``language`` is an optional BCP-47 hint; ``extra`` may carry
        ``prompt`` (from ``stt.prompt`` or a ``pre_transcription`` hook) as a vocabulary hint;
        unknown keys must be ignored.
        """

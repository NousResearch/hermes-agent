"""Transcription Provider Registry.

Central map of registered STT providers, populated by plugins via
:meth:`PluginContext.register_transcription_provider` and consumed by
:mod:`tools.transcription_tools` to dispatch :func:`transcribe_audio` to the active plugin
backend **when** ``stt.provider`` is not a built-in. Built-ins always win: a colliding name is
rejected at registration with a warning (re-checked at dispatch time).
"""

from __future__ import annotations

import logging

from agent.provider_registry import ProviderRegistry, lower_key
from agent.transcription_provider import TranscriptionProvider

logger = logging.getLogger(__name__)


# Native built-in STT handlers. **Kept in sync with ``BUILTIN_STT_PROVIDERS`` in
# :mod:`tools.transcription_tools`** (TestBuiltinSync fails on drift); importing it
# directly would be a circular import.
_BUILTIN_NAMES = frozenset({
    "local", "local_command", "groq", "openai", "mistral", "xai", "elevenlabs", "deepinfra",
})


def _warn_builtin_collision(key: str) -> None:
    logger.warning(
        "Transcription provider '%s' shadows a built-in name; registration ignored. "
        "Built-in STT providers (%s) always win — pick a different name.",
        key, ", ".join(sorted(_BUILTIN_NAMES)),
    )


# Case-insensitive, whitespace-tolerant keys mirror ``tools.transcription_tools``.
_registry: ProviderRegistry[TranscriptionProvider] = ProviderRegistry(
    label="Transcription",
    provider_cls=TranscriptionProvider,
    logger=logger,
    normalize=lower_key,
    builtin_names=_BUILTIN_NAMES,
    on_builtin_collision=_warn_builtin_collision,
)
_registry.export(globals())

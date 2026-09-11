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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Dict  # noqa: F401,E402
from typing import List  # noqa: F401,E402
from typing import Optional  # noqa: F401,E402
import threading  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'hermes_home_key': ('hermes_constants', 'hermes_home_key'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
